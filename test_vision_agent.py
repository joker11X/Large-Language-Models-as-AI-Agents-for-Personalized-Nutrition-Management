# -*- coding: utf-8 -*-
"""
E2E (FAST PIPELINE) - Vision -> Controller(_fast_ingest_meal) -> File(DRY-RUN)
- 仅用 dotenv 读取 API Key
- 测试图片路径在代码里设置 TEST_IMAGE_PATH(改这一行)
- 若图片无效: 自动启用" Vision", 仍走快速通道, 确保 File(DRY-RUN) 收到合规载荷
- 同时给 ctrl 挂一个最简 .agent 适配器, 避免回退路径 AttributeError
"""
import os, csv, base64, shutil, importlib, imghdr
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

# 1) Load API Key only
load_dotenv()  
DRY_RUN = os.getenv("FILE_AGENT_DRY_RUN", "1")
os.environ["FILE_AGENT_DRY_RUN"] = DRY_RUN

# 2) Set the test image path in code (edit here)
TEST_IMAGE_PATH = r"./_tmp_user_data_file_agent/2.jpg"  # 

# 3) Temporary user_data (no pollution even in DRY-RUN)
TEST_ROOT = Path("./_tmp_user_data_vision_fast")
UD = TEST_ROOT / "user_data"
if TEST_ROOT.exists():
    shutil.rmtree(TEST_ROOT)
UD.mkdir(parents=True, exist_ok=True)

# 4) Import tools and point write directory to a temp folder (double safety)
import tools
tools.user_dir = UD

# 5) Minimal daily_nutrition_plan (some logic may read it)
EVERY_MEAL_HEADER = [
    "Timestamp","food_names_volumes",
    "Calcium (mg/d)","Chromium (mcg/d)","Copper (mcg/d)","Fluoride (mg/d)",
    "Iodine (mcg/d)","Iron (mg/d)","Magnesium (mg/d)","Manganese (mg/d)",
    "Molybdenum (mcg/d)","Phosphorus (mg/d)","Selenium (mcg/d)","Zinc (mg/d)",
    "Potassium (mg/d)","Sodium (mg/d)","Chloride (g/d)","Total Water (L/d)",
    "Carbohydrate (g/d)","Total Fiber (g/d)","Fat (g/d)","Linoleic Acid (g/d)",
    "α-Linolenic Acid (g/d)","Protein (g/d)","Vitamin A (mcg/d)","Vitamin C (mg/d)",
    "Vitamin D (mcg/d)","Vitamin E (mg/d)","Vitamin K (mcg/d)","Thiamin (mg/d)",
    "Riboflavin (mg/d)","Niacin (mg/d)","Vitamin B6 (mg/d)","Folate (mcg/d)",
    "Vitamin B12 (mcg/d)","Pantothenic Acid (mg/d)","Biotin (mcg/d)","Choline (mg/d)",
    "total_energy(kcal)"
]
META_COLS = {"Timestamp","food_names_volumes"}
NUTRIENT_COLS = [c for c in EVERY_MEAL_HEADER if c not in META_COLS]

with open(UD / "daily_nutrition_plan.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Date","user_id","Plan_Category"] + NUTRIENT_COLS)
    row = ["2099-01-01","u1","Base"]
    for c in NUTRIENT_COLS:
        lc = c.lower()
        if "kcal" in lc: row.append("2200")
        elif "(g/d)" in c: row.append("100")
        elif "(mg/d)" in c: row.append("1000")
        elif "mcg" in lc or "mcg" in lc: row.append("1000")
        elif "(l/d)" in lc: row.append("2000")  # ml
        else: row.append("0")
    w.writerow(row)

# Stub files (DRY-RUN does not write; safeguard reads)
for name in ["plan_completion_rate.csv","every_meal.csv","food.csv","user_habits.txt"]:
    (UD / name).write_text("", encoding="utf-8")

# 6) Image read + validation
def image_to_b64_if_valid(path_str: str) -> tuple[str|None, str]:
    p = Path(path_str)
    if not p.exists():
        return None, "image path does not exist"
    kind = imghdr.what(p)  # jpeg/png/webp/gif/bmp/...
    if kind not in {"jpeg","png","webp","gif","bmp"}:
        return None, f"unsupported image format: {kind}"
    if p.stat().st_size < 10 * 1024:  # 
        return None, f"image file too small({p.stat().st_size} bytes)"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return b64, ""

# 7)  DRY-RUN File Agent
import agent
importlib.reload(agent)

def build_dry_run_file_agent():
    fa = agent.create_file_agent()
    if DRY_RUN == "1":   # ←
        dry_clause = """
[TEST MODE - DRY RUN]
FILE_AGENT_DRY_RUN=1
Rules:
- Do NOT write files in this mode.
- When op == "ingest_meal" and payload contains valid JSON with keys:
    {"bracket_list": ".", "totals_line": "."}
  then return exactly:
    {"ok": true, "op": "ingest_meal", "message": "OK (dry-run)"}
- If payload is missing or invalid, return:
    {"ok": false, "op":"ingest_meal", "error": "<why>"}
"""
        try:
            fa.system_prompt = (getattr(fa, "system_prompt", "") or "") + "\n" + dry_clause
        except Exception:
            pass
    return fa

# 8) If image invalid, patch: replace agent.create_vision_agent with a Fake Vision
class FakeVisionForTest:
    def run_sync(self, *_, **__):
        # Build visible text that Controller._extract_payload_lines can parse:
        # 1) 1) a bracket_list line; 2) a TOTALS line; optional: CUISINES / CONFIDENCES
        txt = (
            '{"foods":[{"name":"noodles","vol":"150 g"}]}\n'
            "[noodles, 150 g, tomato, 60 g]\n"
            "TOTALS: Total Energy (kcal)=640 kcal; Protein=25 g; Carbohydrate=85 g; "
            "Fat=18 g; Total Fiber=6 g; Sodium (mg)=900 mg\n"
            "CUISINES: noodles=chinese; tomato=western\n"
            "CONFIDENCES: noodles=0.86; tomato=0.72\n"
        )
        return SimpleNamespace(output=txt)
    # compat: run() alias
    run = run_sync

def maybe_patch_fake_vision_if_needed(img_b64: str|None, reason: str):
    if img_b64 is None:
        print(f"[WARN] enabling "Fake Vision" (reason: {reason}). ")
        agent.create_vision_agent = lambda *a, **k: FakeVisionForTest()

# 9) Main flow: force fast pipeline + attach fallback adapter to avoid AttributeError
def main():
    os.environ["FAST_PIPELINE"] = "1"  # Use Vision->File fast pipeline
    if not os.getenv("OPENAI_API_KEY"):
        print("[STOP] OPENAI_API_KEY not detected(dotenv loaded). please set it in .env. ")
        return

    dialog = agent.create_dialog_agent()
    vision = agent.create_vision_agent()  # attach;  fast-path
    filea  = build_dry_run_file_agent()
    ctrl   = agent.create_controller_agent(dialog=dialog, vision=vision, filea=filea)

    # ctrl  .agent(,  AttributeError)
     # class _LegacyBypass:
     #     def run_sync(self, payload, *a, **k):
      #        return SimpleNamespace(output={"route":"legacy-bypass","error":"Disabled in test"})
     # setattr(ctrl, "agent", _LegacyBypass())

    # Vision prompt (require output payload_for_file_agent)
    header_hint = ", ".join([c for c in NUTRIENT_COLS if "total_energy" not in c.lower()])
    vision_user_text = (
        "Detect foods in the photo and output strict JSON plus `payload_for_file_agent`.\n"
        "For `totals_line`, use EXACT labels from every_meal.csv. "
        "Unit mapping: (g/d)->g, (mg/d)->mg, (mcg|mcg/d)->mcg, (L/d)->ml. "
        f"Include these keys: {header_hint}, and `Total Energy (kcal)` explicitly.\n"
        "Also include `CUISINES:` and `CONFIDENCES:` lines.\n"
        "NOTE: Downstream File Agent is in DRY-RUN mode; only validate, no writes."
    )

    img_b64, reason = image_to_b64_if_valid(TEST_IMAGE_PATH)
    maybe_patch_fake_vision_if_needed(img_b64, reason)

    res = ctrl.handle_meal(user_text=vision_user_text, image_b64=(img_b64 or ""), user_id="u1")

    print("\n=== FAST PIPELINE RESULT ===")
    print("route:", res.get("route"))
    print("\n--- VISION OUTPUT (head) ---")
    vo = (res.get("vision_output") or "")
    print(vo[:1200] + ("..." if len(vo) > 1200 else ""))
    print("\n--- FILE AGENT RESULT (DRY-RUN) ---")
    print(res.get("file_result"))
    if res.get("dialog_reply"):
        print("\n--- DIALOG REPLY ---")
        print(res.get("dialog_reply"))
    print("\n[user_data dir]", UD.resolve())
    print("DRY-RUN enabled: File Agent validates only, no writes.")

if __name__ == "__main__":
    main()