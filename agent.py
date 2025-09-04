# agent.py
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai import Agent as PydAgent, RunContext
from claudeagent import ClaudeToolAgent  # Your Claude wrapper; must be compatible with .run_sync
from dotenv import load_dotenv
import tools
import threading
from pathlib import Path
from dataclasses import dataclass
import os, base64, imghdr
from types import SimpleNamespace
from openai import OpenAI
import re, json, time, hashlib, os
from typing import Optional, Tuple
# Load environment variables
load_dotenv()

# ========== Utility functions ==========

def _load_text(path: str) -> str:
    """
    Read text from a file (UTF-8). If missing, raise a clear error.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"System prompt file not found: {p}")
    return p.read_text(encoding="utf-8")

def _load_prompt(path, fallback):
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else fallback



def _guess_mime_from_b64(b64: str, default: str = "image/jpeg") -> str:
    try:
        data = base64.b64decode(b64, validate=True)
        kind = imghdr.what(None, data)
        return {
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "gif": "image/gif",
            "bmp": "image/bmp",
        }.get(kind, default)
    except Exception:
        return default

class VisionResponse:
    def __init__(self, output: str, history=None, raw=None):
        self.output = output
        self._history = history or []
        self.raw = raw
    def all_messages(self):
        return list(self._history)

class VisionAgent:
    """
    Vision agent:
    - run/run_sync(image_b64: Optional[str], user_text: str) supports image+text multimodality
    - Default model can be overridden by env OPENAI_VISION_MODEL (default 'gpt-4o-mini')
    - Works with text-only inputs as well
    """
    def __init__(self, system_prompt: str, model: str | None = None):
        self.system_prompt = system_prompt
        self.model = model or os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
        self.client = OpenAI()
    
    def _normalize_b64(self, s: str) -> tuple[str, str]:
        """Return (plain base64, mime). If 's' has a data: prefix, parse mime; also strip newlines and validate."""
        s = (s or "").strip()
        mime = None
        if s.startswith("data:"):
            m = re.match(r"^data:(image/[a-z0-9.+-]+);base64,(.*)$", s, re.I | re.S)
            if m:
                mime = m.group(1).lower()
                s = m.group(2)
        s = s.replace("\n", "").replace("\r", "")
        # Strict validation; raise if invalid; outer layer decides to continue or error
        base64.b64decode(s, validate=True)
        return s, (mime or _guess_mime_from_b64(s))

    def _build_messages(self, user_text: str = "", image_b64: str | None = None):
        content = []
        if user_text:
            content.append({"type": "text", "text": user_text})
        if image_b64:
            try:
                b64, mime = self._normalize_b64(image_b64)
            except Exception:
                # Provide clearer errors; alternatively return text-only message
                raise ValueError("Invalid base64 passed to VisionAgent.")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content or [{"type": "text", "text": "(no user text)"}]}]
    

    def run_sync(self, msg: str | dict | None = None, *,
                 user_text: str | None = None,
                 image_b64: str | None = None,
                 message_history=None):
        # Compat: legacy dict payload from app.py
        if isinstance(msg, dict):
            user_text = user_text or msg.get("text") or msg.get("user_text") or ""
            image_b64 = image_b64 or msg.get("image") or msg.get("image_b64")

        force_clause = (
            "MANDATORY: Never ask for a clearer photo. "
            "Always output the JSON + payload even if uncertain."
        )
        messages = self._build_messages((user_text or msg or "") + "\n\n" + force_clause,
                                        image_b64)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=1400,
        )

        # Extract text content and return
        content = ""
        try:
            content = (resp.choices[0].message.content or "").strip()
        except Exception:
            # Fallback to avoid re-raising
            content = str(resp)

        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (user_text or msg or "")},
            {"role": "assistant", "content": content},
        ]
        return VisionResponse(output=content, history=history, raw=resp)
    
    def run(self, *args, **kwargs):
        return self.run_sync(*args, **kwargs)



class ControllerAgent:
    """
    LLM-driven controller:
    -  pydantic-ai Agent (self.agent) ""
    - Utility functions Vision/File/Dialog  operator
    - handle_meal() 
    """
    # ---------- / ----------
    @staticmethod
    def _extract_payload_lines(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """ JSON  bracket_list / totals_line / CUISINES / CONFIDENCES"""
        bracket, totals, cuisines, confs = None, None, None, None
        # 
        for ln in (text or "").splitlines():
            s = ln.strip()
            if not s:
                continue
            if (s.startswith("[") and s.endswith("]") and "," in s):
                bracket = bracket or s
            elif s.upper().startswith("TOTALS:"):
                totals = totals or s
            elif s.upper().startswith("CUISINES:"):
                cuisines = cuisines or s
            elif s.upper().startswith("CONFIDENCES:"):
                confs = confs or s
        if bracket and totals:
            return bracket, totals, cuisines, confs
        # JSON 
        try:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(text[start:end+1])
                payload = None
                if isinstance(obj, dict):
                    payload = (obj.get("payload_for_file_agent") or
                               obj.get("payload") or
                               obj.get("file_payload"))
                    if payload is None:
                        for v in obj.values():
                            if isinstance(v, dict) and any(k in v for k in ("payload_for_file_agent","payload","file_payload")):
                                payload = v.get("payload_for_file_agent") or v.get("payload") or v.get("file_payload")
                                break
                if isinstance(payload, dict):
                    bl = payload.get("bracket_list")
                    tl = payload.get("totals_line")
                    cu = payload.get("cuisines_line") or payload.get("CUISINES")
                    cf = payload.get("confidences_line") or payload.get("CONFIDENCES")
                    if isinstance(bl, list):
                        bl = "[" + ", ".join(str(x) for x in bl) + "]"
                    if isinstance(tl, dict):
                        items = [f"{k}={v}" for k, v in tl.items() if v is not None]
                        if items:
                            tl = "TOTALS: " + "; ".join(items)
                    if isinstance(bl, str) and isinstance(tl, str) and tl.upper().startswith("TOTALS:"):
                        return bl.strip(), tl.strip(), (cu if isinstance(cu, str) else None), (cf if isinstance(cf, str) else None)
        except Exception:
            pass
        return bracket, totals, cuisines, confs
    
    def _pick_valid_image_b64(self, candidate: str | None) -> str | None:
        """
         _image_b64_tmp; If,  LLM  candidate. 
         strip/  base64 ,  'invalid_base64 image_url'. 
        """
        # 1) ( LLM )
        s = (self._image_b64_tmp or "").strip()
        if len(s) < 1000:   # : 
            s = (candidate or "").strip()

        # 2) 
        s = s.replace("\n", "").replace("\r", "")

        # 3) base64 ( None)
        if not s:
            return None
        try:
            base64.b64decode(s, validate=True)
        except Exception:
            return None
        return s
    @staticmethod
    def _parse_totals_line(totals_line: str) -> dict:
        """ 'TOTALS: Key=val unit; Key=val unit; ...'  dict"""
        line = (totals_line or "").strip()
        if not line.upper().startswith("TOTALS:"):
            return {}
        body = line.split(":", 1)[1]
        items = [x.strip() for x in body.split(";") if x.strip()]
        out = {}
        for it in items:
            if "=" in it:
                k, v = it.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    @staticmethod
    def _core_ok(totals: dict) -> bool:
        """(/////)"""
        if not isinstance(totals, dict):
            return False
        txt = " ".join(totals.keys()).lower()
        for kw in ["energy", "carbohydrate", "protein", "fat", "fiber", "sodium"]:
            if kw not in txt:
                return False
        return True

    # ---------- Initialize ----------
    
    def __init__(self, llm=None, system_prompt: Optional[str]=None, tools=None,
                dialog=None, vision=None, filea=None):
        self.llm = llm or OpenAIModel("gpt-4o-mini")
        
        

        # Added: early-stop event & final text cache
        self._final_reply_event = threading.Event()
        self._final_reply_text = ""


        # hint
        if system_prompt is None:
            if Path("./controller_agent_system_prompt.txt").exists():
                system_prompt = Path("./controller_agent_system_prompt.txt").read_text(encoding="utf-8")
            elif Path("./prompts/controller_agent_system_prompt.txt").exists():
                system_prompt = Path("./prompts/controller_agent_system_prompt.txt").read_text(encoding="utf-8")
            else:
                system_prompt = "You are the Controller Agent."
        self.system_prompt = system_prompt

        # ()
        self.dialog = dialog
        self.vision = vision
        self.filea  = filea

        # ""(/user_id)
        self._image_b64_tmp: Optional[str] = None
        self._user_id_tmp: Optional[str] = None

        #  pydantic-ai Agent( tools)
        self.agent = PydAgent(self.llm, system_prompt=self.system_prompt)

        # ---------- :  __init__  ----------

        @self.agent.tool  # Use .tool when context is needed (1st arg is ctx)
        def op_vision_analyze_meal_image(ctx: RunContext[None],
                                        image_b64: Optional[str] = None,
                                        user_text: Optional[str] = None) -> dict:
            """
            Vision.analyze_meal_image -> { ok, raw_text, bracket_list, totals_line, cuisines_line?, confidences_line? }
            """
            v = self.vision or create_vision_agent()
            img = image_b64 or self._image_b64_tmp  # Allow fetching from the slot
            raw = (v.run_sync(user_text=user_text or "analyze meal photo",
                            image_b64=img).output or "").strip()
            br, tl, cu, cf = self._extract_payload_lines(raw)
            return {
                "ok": bool(br and tl),
                "raw_text": raw,
                "bracket_list": br,
                "totals_line": tl,
                "cuisines_line": cu,
                "confidences_line": cf,
            }

        @self.agent.tool_plain  #  ctx  .tool_plain
        def op_file_ingest_meal(payload_for_file_agent: dict,
                                user_id: Optional[str] = None) -> dict:
            """
            File.ingest_meal(payload_for_file_agent) -> JSON
            NOTE: If FILE_AGENT_DRY_RUN=1, . 
            """
            fa = self.filea or create_file_agent()

            #  token()
            token = hashlib.md5(f"{payload_for_file_agent}|{int(time.time()/60)}".encode("utf-8")).hexdigest()
            payload_for_file_agent = dict(payload_for_file_agent or {})
            payload_for_file_agent["client_token"] = token

            uid = user_id or (self._user_id_tmp or "default")
            prefix = "DRY_RUN=1\n" if os.getenv("FILE_AGENT_DRY_RUN", "0") == "1" else ""
            msg = (
                prefix +
                "Please perform op=ingest_meal for the following:\n" +
                json.dumps({"user_id": uid, "payload": payload_for_file_agent}, ensure_ascii=False) +
                "\nReturn ONLY the JSON per the Output Contract."
            )
            out = fa.run_sync(msg).output or "{}"
            try:
                return json.loads(out)
            except Exception:
                return {"ok": False, "error": f"File agent returned non-JSON: {out[:200]}"}
        @self.agent.tool_plain
        def op_fast_vf_ingest(user_id: Optional[str] = None,
                            image_b64: Optional[str] = None,
                            user_text: Optional[str] = "") -> dict:
            """
            Single-call pipeline: Vision(image_b64, user_text) -> extract payload -> File.ingest_meal(payload, user_id)
            When FILE_AGENT_DRY_RUN=1, validate only (no write).
            Return a unified JSON with Vision text, payload line(s), and File results.
            """
            # 1) Vision
            v = self.vision or create_vision_agent()
            img = self._pick_valid_image_b64(image_b64)
            if not img:
                return {"ok": False, "error": "no_image"}
            v_out = (v.run_sync(user_text=user_text or "", image_b64=img).output or "").strip()
            vision_json = {}
            start, end = v_out.find("{"), v_out.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    vision_json = json.loads(v_out[start:end+1])
                except Exception:
                    vision_json = {}
            meal_totals_json = vision_json.get("meal_totals") or {}
            # 2)  payload ()
            br = tl = cu = cf = None
            if not meal_totals_json:
                br, tl, cu, cf = self._extract_payload_lines(v_out)
            if not meal_totals_json and not (br and tl):
                return {"ok": False, "error": "invalid_payload_from_vision", "vision_output": v_out[:800]}

            # 3) File.ingest_meal( token + DRY-RUN )
            fa = self.filea or create_file_agent()
            token_src = json.dumps(meal_totals_json, sort_keys=True) if meal_totals_json else f"{br}|{tl}"
            token = hashlib.md5((token_src + str(int(time.time()/60))).encode("utf-8")).hexdigest()
            payload = {
                "client_token": token,
                "totals_json": meal_totals_json,   # 40+()
                "vision_json": vision_json         # (items/meal_id/timestamp...)
            }
            if br: payload["bracket_list"] = br
            if tl: payload["totals_line"] = tl
            if cu: payload["cuisines_line"] = cu
            if cf: payload["confidences_line"] = cf
            uid = user_id or (self._user_id_tmp or "default")

            prefix = "DRY_RUN=1\n" if os.getenv("FILE_AGENT_DRY_RUN", "0") == "1" else ""
            msg = (
                prefix +
                "Please perform op=ingest_meal for the following:\n" +
                json.dumps({"user_id": uid, "payload": payload}, ensure_ascii=False) +
                "\nReturn ONLY the JSON per the Output Contract."
            )
            f_raw = fa.run_sync(msg).output or "{}"
            try:
                f_json = json.loads(f_raw)
            except Exception:
                f_json = {"ok": False, "error": f"file_agent_non_json:{f_raw[:200]}"}

            return {
                "ok": bool(f_json.get("ok", True)),
                "vision_output": v_out,
                "payload_for_file_agent": {"bracket_list": br, "totals_line": tl,
                                        **({"cuisines_line": cu} if cu else {}),
                                        **({"confidences_line": cf} if cf else {})},
                "file_result": f_json,
            }

        @self.agent.tool_plain
        def op_dialog_answer(user_query: str) -> str:
            print(f"[CTRLDIALOG] op_dialog_answer CALLED (user_id={self._user_id_tmp})", flush=True)
            print(f"[CTRLDIALOG] brief: {user_query[:200].replace(chr(10),' ')}", flush=True)

            d = self.dialog or create_dialog_agent()
            t0 = time.perf_counter()
            try:
                out = d.run_sync(user_query).output or ""
            except Exception as e:
                out = f"[DIALOG_ERROR] {type(e).__name__}: {e}"
            print(f"[CTRLDIALOG] total_dialog_time={time.perf_counter()-t0:.3f}s", flush=True)

            # early stop
            if not self._final_reply_event.is_set():
                self._final_reply_text = out
                self._final_reply_event.set()
                print("[CTRL] dialog set FINAL reply (early-stop event ON)", flush=True)

            # ; handle_text ()
            return "[FINAL_BY_DIALOG]" + out

    

        @self.agent.tool_plain
        def op_file_close_day(user_id: str, date: Optional[str] = None) -> dict:
            fa = self.filea or create_file_agent()
            out = fa.run_sync(f"op=close_day user_id={user_id} date={date or 'today'}").output or "{}"
            try:
                return json.loads(out)
            except Exception:
                return {"ok": False, "error": out[:200]}

        @self.agent.tool_plain
        def op_file_gen_next_day_plan(user_id: str, date: Optional[str] = None) -> dict:
            fa = self.filea or create_file_agent()
            out = fa.run_sync(f"op=gen_next_day_plan user_id={user_id} date={date or 'tomorrow'}").output or "{}"
            try:
                return json.loads(out)
            except Exception:
                return {"ok": False, "error": out[:200]}


    # ----------  ----------
    def run(self, user_query: str, image_b64: Optional[str] = None, user_id: Optional[str] = None):
        # ()
        self._image_b64_tmp = image_b64
        self._user_id_tmp = user_id

        #  LLM hint, /
        if image_b64:
            user_query = f"{user_query}\n\n[IMAGE_B64_ATTACHED]\n"

        return self.agent.run_sync(user_query)

    def run_sync(self, user_query: str, image_b64: Optional[str] = None, user_id: Optional[str] = None):
        res = self.run(user_query, image_b64=image_b64, user_id=user_id)
        class _Wrap:
            def __init__(self, inner):
                self.inner = inner
                self.output = getattr(inner, "output", None) or str(inner)
            def all_messages(self):
                return []
        return _Wrap(res)


    def handle_meal(self, user_text: str, image_b64: str, user_id: str):
        """
         FAST PIPELINE(Vision -> File),  LLM . 
        . 
        """
        self._image_b64_tmp = image_b64 or ""
        self._user_id_tmp = user_id

        # --- FAST PIPELINE ---
        if str(os.getenv("FAST_PIPELINE", "")).strip().lower() in {"1", "true", "yes", "on"}:
            v = self.vision or create_vision_agent()
            raw_text = (v.run_sync(user_text=user_text, image_b64=image_b64).output or "").strip()

            # /JSON payload 
            br, tl, cu, cf = self._extract_payload_lines(raw_text)

            file_json = None
            if br and tl:
                payload = {"bracket_list": br, "totals_line": tl}
                if cu: payload["cuisines_line"] = cu
                if cf: payload["confidences_line"] = cf

                fa = self.filea or create_file_agent()

                #  token(), DRY-RUN 
                token = hashlib.md5(f"{payload}|{int(time.time()/60)}".encode("utf-8")).hexdigest()
                payload["client_token"] = token
                prefix = "DRY_RUN=1\n" if os.getenv("FILE_AGENT_DRY_RUN", "0") == "1" else ""

                msg = (
                    prefix +
                    "Please perform op=ingest_meal for the following:\n" +
                    json.dumps({"user_id": user_id or "default", "payload": payload}, ensure_ascii=False) +
                    "\nReturn ONLY the JSON per the Output Contract."
                )
                out = fa.run_sync(msg).output or "{}"
                try:
                    file_json = json.loads(out)
                except Exception:
                    file_json = {"ok": False, "error": f"File agent returned non-JSON: {out[:200]}"}

            dialog_text = None
            if file_json and file_json.get("ok"):
                # ( file_json /)
                dlg = self.dialog or create_dialog_agent()
                hints = {
                    "meal_preview": file_json.get("preview", {}),
                    "top_remaining_after": (file_json.get("preview") or {}).get("top_remaining_after"),
                }
                dialog_text = dlg.run_sync(
                    "The user just logged a meal. Based on the remaining targets below, "
                    "give concise guidance for the next meal (2-3 sentences), friendly but specific.\n"
                    + json.dumps(hints, ensure_ascii=False)
                ).output

            return {
                "route": "controller-fast",
                "vision_output": raw_text,
                "file_result": file_json,
                "dialog_reply": dialog_text or {"route": "fast-path", "note": "dialog skipped"},
            }

        # --- : LLM () ---
        # --- : LLM ( handle_text early stop) ---
        self._user_id_tmp = user_id
        # early stop
        self._final_reply_event.clear()
        self._final_reply_text = ""

        result_box, err_box = {}, {}

        def _runner():
            try:
                result_box["res"] = self.agent.run_sync(
                    f"{user_text}\n\n[IMAGE_B64_READY]\nuser_id={user_id}"
                )
            except Exception as e:
                err_box["e"] = e

        th = threading.Thread(target=_runner, daemon=True)
        th.start()

        # early stop;  Dialog ,  HTTP
        while th.is_alive():
            if self._final_reply_event.is_set():
                txt = self._final_reply_text or ""
                print("[CTRL] EARLY-FINISH via dialog event (image path) -> return to HTTP", flush=True)
                return {"route": "controller-early", "reply": txt}
            time.sleep(0.03)

        # , early stop
        if "e" in err_box:
            err = err_box["e"]
            print(f"[CTRL] controller error (image path): {type(err).__name__}: {err}", flush=True)
            return {"route": "controller-error", "reply": f"[ERROR] {type(err).__name__}: {err}"}

        res = result_box.get("res")
        out = getattr(res, "output", "") or ""
        calls = getattr(res, "tool_calls", []) or []

        def _name(c): 
            return c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
        print(f"[CTRL] tool_calls (image path): {[_name(c) for c in calls]}", flush=True)

        vision_text, file_json, dialog_text = None, None, None
        for c in calls:
            if (isinstance(c, dict) and c.get("name") == "op_vision_analyze_meal_image") or getattr(c, "name", "") == "op_vision_analyze_meal_image":
                r = c.get("result") or {}
                vision_text = (r.get("raw_text") if isinstance(r, dict) else None) or vision_text
            elif (isinstance(c, dict) and c.get("name") == "op_file_ingest_meal") or getattr(c, "name", "") == "op_file_ingest_meal":
                file_json = c.get("result") or file_json
            elif (isinstance(c, dict) and c.get("name") == "op_dialog_answer") or getattr(c, "name", "") == "op_dialog_answer":
                dialog_text = c.get("result")

        return {
            "route": "controller-llm",
            "vision_output": vision_text,
            "file_result": file_json,
            "dialog_reply": dialog_text or out,
        }

    
    # agent.py   in class ControllerAgent
    def handle_text(self, user_text: str, user_id: str = "default"):
        """Run controller; if dialog returns, immediately early-finish and return to HTTP."""
        self._user_id_tmp = user_id
        print(f"[CTRL] handle_text user={user_id} msg={user_text}", flush=True)

        # reset early-stop latch
        self._final_reply_event.clear()
        self._final_reply_text = ""

        result_box, err_box = {}, {}

        #  controller ()
        def _runner():
            try:
                result_box["res"] = self.agent.run_sync(user_text)
            except Exception as e:
                err_box["e"] = e

        th = threading.Thread(target=_runner, daemon=True)
        th.start()

        # : ""
        while th.is_alive():
            if self._final_reply_event.is_set():
                txt = self._final_reply_text or ""
                print("[CTRL] EARLY-FINISH via dialog event -> return to HTTP", flush=True)
                return {"route": "controller-early", "reply": txt}
            time.sleep(0.03)  # , 

        #  controller , early stop
        if "e" in err_box:
            err = err_box["e"]
            print(f"[CTRL] controller error: {type(err).__name__}: {err}", flush=True)
            return {"route": "controller-error", "reply": f"[ERROR] {type(err).__name__}: {err}"}

        res = result_box.get("res")
        out = getattr(res, "output", "") or ""
        calls = getattr(res, "tool_calls", []) or []

        def _name(c): 
            return c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
        print(f"[CTRL] tool_calls: {[_name(c) for c in calls]}", flush=True)

        # If op_dialog_answer,  result
        dialog_text = None
        for c in calls:
            if isinstance(c, dict) and c.get("name") == "op_dialog_answer":
                dialog_text = c.get("result")
                break
            if getattr(c, "name", None) == "op_dialog_answer":
                dialog_text = getattr(c, "result", None)
                break

        return {"route": "controller-llm", "reply": dialog_text or out}





# agent.py

def create_controller_agent(dialog=None, vision=None, filea=None, model_name: str = "gpt-4o-mini"):
    llm = OpenAIModel(model_name)
    # hint(Default)
    if Path("./controller_agent_system_prompt.txt").exists():
        sp = Path("./controller_agent_system_prompt.txt").read_text(encoding="utf-8")
    elif Path("./prompts/controller_agent_system_prompt.txt").exists():
        sp = Path("./prompts/controller_agent_system_prompt.txt").read_text(encoding="utf-8")
    else:
        sp = "You are the Controller Agent."
    return ControllerAgent(llm=llm, system_prompt=sp, dialog=dialog, vision=vision, filea=filea)



def create_dialog_agent() -> Agent:
    # 
    model = OpenAIModel("gpt-4o-mini")

    #  prompts hint
    candidates = [
        Path("prompts/dialog_agent_system_prompt.txt"),
    ]
    for p in candidates:
        if p.exists():
            sys_prompt = p.read_text(encoding="utf-8")
            break
    else:
        sys_prompt = "You are the Dialog Agent."

    return Agent(
        model,
        system_prompt=sys_prompt,
        tools=[
            tools.read_user_csv_tail,
            tools.read_food_csv,
            tools.read_user_csv,    
            tools.list_user_files,  
            tools.read_txt_file,   
            tools.write_txt_file, 
        ],
    )


def create_file_agent():
    """File editing/linked-computation Agent (Claude)"""
    
    file_agent_prompt = _load_text("prompts/file_agent_system_prompt.txt")
    return ClaudeToolAgent(
        system_prompt=file_agent_prompt
    )

def create_vision_agent():
    """ Agent(OpenAI ,  base64 )"""
    prompt_path = Path("prompts/vision_agent_system_prompt.txt")
    sys_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else \
        "Agent, . JSONpayload_for_file_agent. "
    return VisionAgent(system_prompt=sys_prompt, model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"))

# ==========  ==========

_AVAILABLE_ROLES = ("controller", "dialog", "file", "vision")

def get_available_agents():
    """ Agent """
    return list(_AVAILABLE_ROLES)

def create_agent_by_type(role: str):
    role = (role or "").lower()
    if role == "controller":
        return create_controller_agent()
    if role == "dialog":
        return create_dialog_agent()
    if role == "file":
        return create_file_agent()
    if role == "vision":
        return create_vision_agent()
    raise ValueError(f" Agent : {role}(: {', '.join(_AVAILABLE_ROLES)})")

# ========== () ==========

def console_agent(role: str = "controller"):
    """Interactive console to test a specific role agent"""
    try:
        agent = create_agent_by_type(role)
        print(f" {role.upper()} Agent started (type exit/quit to exit)")
    except Exception as e:
        print(f"Initialize {role} agent : {e}")
        return

    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            resp = agent.run_sync(user_input, message_history=history)
            # pydantic_ai.Agent &  ClaudeToolAgent 
            history = list(resp.all_messages())
            print("Assistant: ", resp.output)
        except Exception as e:
            print("Error: ", e)

def start_console_agent(role: str = "controller"):
    """Start console test in a separate thread and return a Thread instance"""
    thread = threading.Thread(target=lambda: console_agent(role), daemon=True)
    thread.start()
    return thread



def test_agents():
    """ Agent Initialize"""
    for role in _AVAILABLE_ROLES:
        try:
            print(f"\n[check] Testing agent: {role}")
            agent = create_agent_by_type(role)
            result = agent.run_sync("Hello, please introduce yourself.")
            print(f"[ok] {role} First 80 characters of reply:{(result.output or '')[:80]}...")
        except Exception as e:
            print(f"[fail] {role} Load/Invoke failed:{e}")


