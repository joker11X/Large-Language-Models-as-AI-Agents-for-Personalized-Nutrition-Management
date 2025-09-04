import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
import tools
import inspect
import json

load_dotenv()

class ClaudeToolAgent:
    def __init__(self, system_prompt="You are a nutrition assistant with tool-use capabilities."):
        self.client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        self.model = "claude-sonnet-4-20250514"
        self.system_prompt = system_prompt
        self.tools = self._load_tools()

    def _load_tools(self):
        """
        """
        Dynamically register functions in tools.py as Claude-callable tools.
        Each tool includes: name, description, and input_schema.
        def describe(fn):
            doc = fn.__doc__ or ""
            sig = inspect.signature(fn)
            props, required = {}, []
            for name, param in sig.parameters.items():
                # default string
                schema = {"type": "string"}
                # Use object type for append_to_user_csv's 'row' argument
                if fn.__name__ == "append_to_user_csv" and name == "row":
                    schema = {
                        "type": "object",
                        "additionalProperties": {"type": ["string", "number", "null"]}
                    }
                props[name] = schema
                if param.default is inspect.Parameter.empty:
                    required.append(name)
            return {
                "name": fn.__name__,
                "description": doc.strip(),
                "input_schema": {"type": "object", "properties": props, "required": required}
            }

        # tools
        function_names = [
            "read_user_csv",
            "append_to_user_csv",
            "list_recommend_files",
            "list_user_files",
            "read_txt_file",
            "write_txt_file",
            "upsert_food_csv_fuzzy"
        ]

        tool_list = []
        self.func_map = {}
        for name in function_names:
            fn = getattr(tools, name, None)
            if fn:
                tool_list.append(describe(fn))
                self.func_map[name] = fn
        return tool_list

    def run_sync(self, msg: str, message_history=None, max_hops: int = 16):
        if message_history is None:
            message_history = []

        def _to_blocks(c):
            return c if isinstance(c, list) else [{"type": "text", "text": c}]

        # Restore history + current user input
        messages = []
        for m in message_history:
            messages.append({"role": m["role"], "content": _to_blocks(m["content"])})
        messages.append({"role": "user", "content": [{"type": "text", "text": msg}]})

        for _ in range(max_hops):
            resp = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
                max_tokens=1024,
            )

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]

            if not tool_uses:
                final = "".join(texts).strip()
                new_history = message_history + [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": final},
                ]
                return ClaudeResponse(final, new_history)

            # Important: append the entire assistant message (with tool_use) to history
            messages.append({"role": "assistant", "content": resp.content})

            # Execute all tools; build proper tool_result blocks; support fast-exit


            tool_results = []
            executed = []  # record executed tools (for fast-exit summary)
            for tu in tool_uses:
                name = tu.name
                args = tu.input or {}
                fn = self.func_map.get(name)
                try:
                    print(f"File agent called the tool function: {name}({args})")
                    out = fn(**args) if fn else {"error": f"Unknown tool: {name}"}
                except Exception as e:
                    out = {"error": repr(e)}

                # Stringify output to avoid block schema errors
                if not isinstance(out, str):
                    out = json.dumps(out, ensure_ascii=False)

                # (Optional) truncate large outputs to avoid slow responses
                if len(out) > 8000:
                    out = out[:8000] + " ...(truncated)"

                executed.append({"name": name, "args": args, "out": out})

                # Proper tool_result block (content must be a list of blocks)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": [ { "type": "text", "text": out } ],
                })

            # Fast exit: if upsert_food_csv_fuzzy ran, finalize and return
            if any(x["name"] == "upsert_food_csv_fuzzy" for x in executed):
                # Approximate which files were written for Dialog to reference
                files_touched = set()
                for x in executed:
                    if x["name"] == "append_to_user_csv":
                        n = (x["args"].get("name") or "").strip()
                        if n: files_touched.add(n)
                    if x["name"] == "upsert_food_csv_fuzzy":
                        files_touched.add("food.csv")
                if not files_touched:
                    files_touched = {"every_meal.csv", "plan_completion_rate.csv", "food.csv"}

                # Use the upsert summary as a preview string
                upsert_out = next((x["out"] for x in executed if x["name"] == "upsert_food_csv_fuzzy"), "food.csv updated.")
                final_obj = {
                    "ok": True,
                    "files_touched": sorted(files_touched),
                    "preview": { "message": upsert_out }
                }
                final_text = json.dumps(final_obj, ensure_ascii=False)
                print(" Fast-exit: File agent finalized after upsert_food_csv_fuzzy")
                # End run_sync here; let the Controller pass results to Dialog
                return ClaudeResponse(final_text, message_history + [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": final_text},
                ])

            # Normal path: send tool_result as the next user message per protocol
            print(f"posting {len(tool_results)} tool_result(s) back to Claude")
            messages.append({ "role": "user", "content": tool_results })



        # Fallback when max hops reached
        final = "Stopped after max tool hops without a final answer."
        new_history = message_history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": final},
        ]
        return ClaudeResponse(final, new_history)



class ClaudeResponse:
    def __init__(self, output, history):
        self.output = output
        self._history = history

    def all_messages(self):
        return self._history