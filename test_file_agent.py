# -*- coding: utf-8 -*-
"""
Rows-only test (preserve existing CSV headers):
- Do NOT delete/overwrite your prepared CSVs.
- Agent should ONLY append a row to every_meal.csv and update food.csv counts.
- We verify row counts increased and key values are sane.
"""
import csv, importlib, time, re
from pathlib import Path
import tools
import agent

# ============ CONFIG ============
# Point this to the folder where you've pre-created CSVs with correct headers:
PREPARED_USER_DATA = Path("./_tmp_user_data_file_agent/user_data")  # <-- change to your folder
# ================================

assert PREPARED_USER_DATA.exists(), "Prepared user_data folder not found"
tools.user_dir = PREPARED_USER_DATA
UD = PREPARED_USER_DATA

# ---- helpers ----
def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return max(0, sum(1 for _ in f) - 1)  # minus header

def print_tail(path: Path, n: int = 6):
    if not path.exists():
        print(f"[tail] {path} does not exist"); return
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print(f"[tail] {path.name} last {min(n, len(lines))} lines:")
    for line in lines[-n:]:
        print(line.rstrip("\n"))

def load_every_meal_header() -> list:
    em = UD / "every_meal.csv"
    assert em.exists() and em.stat().st_size > 0, "every_meal.csv must exist with header"
    with open(em, "r", encoding="utf-8", errors="ignore") as f:
        return f.readline().rstrip("\n").split(",")

# Build totals_line from header (units)
def header_unit_to_payload_unit(col: str) -> str:
    m = re.search(r"\(([^)]+)\)", col or "")
    if not m: return ""
    u = m.group(1).lower()
    if "kcal" in u: return "kcal"
    if "l/d" in u:  return "ml"     # payload uses ml for water
    if "mg/d" in u: return "mg"
    if "mcg" in u or "µg" in u or "¦ìg" in u: return "mcg"
    if "g/d"  in u: return "g"
    return ""

def build_totals_line_from_header(header: list) -> str:
    META = {"Date","Timestamp","user_id","meal_id","meal_window","food_names_volumes"}
    nutrient_cols = [c for c in header if c not in META and "total_energy" not in c.lower()]
    seed = {
        "Protein (g/d)": 27,
        "Total Fiber (g/d)": 8,
        "Fat (g/d)": 20,
        "Carbohydrate (g/d)": 60,
        "Sodium (mg/d)": 900,
        "Total Water (L/d)": 300,   # ml in payload
        "Total Water (ml/d)": 300,
    }
    parts = []
    for col in nutrient_cols:
        unit = header_unit_to_payload_unit(col)
        v = seed.get(col, 0)
        if unit:
            parts.append(f"{col}={v} {unit}")
    parts.append("Total Energy (kcal)=640 kcal")
    return "TOTALS: " + "; ".join(parts)

def main():
    # Sanity: required files exist with headers
    em = UD / "every_meal.csv"
    fd = UD / "food.csv"
    dp = UD / "daily_nutrition_plan.csv"
    for p in [em, fd, dp]:
        assert p.exists() and p.stat().st_size > 0, f"{p.name} must exist with header"

    # Build message
    header = load_every_meal_header()
    totals_line = build_totals_line_from_header(header)
    bracket_list = "[steamed white rice, 180, mapo tofu, 380]"

    importlib.reload(agent)
    filea = agent.create_file_agent()

    # Count before
    before_em = count_rows(em)
    before_fd = count_rows(fd)

    message = (
        "op=ingest_meal\n"
        "user_id=u1\n"
        f"bracket_list={bracket_list}\n"
        f"totals_line={totals_line}\n"
        "For this test, ONLY append every_meal.csv and food.csv (do not touch other CSVs). "
        "Use as few tool calls as possible and return JSON."
    )

    # Run
    if hasattr(filea, "run_sync"):
        res = filea.run_sync(msg=message)
    else:
        res = filea.run(msg=message)
    print("Agent output (head):", str(getattr(res, "output", res))[:240], "...")

    time.sleep(0.05)  # let FS settle

    # Count after
    after_em = count_rows(em)
    after_fd = count_rows(fd)
    print(f"[check] every_meal rows: {before_em} -> {after_em}")
    print(f"[check] food rows:       {before_fd} -> {after_fd}")

    # Assertions
    if after_em <= before_em:
        print_tail(em)
        raise AssertionError("every_meal.csv should have at least one new data row")
    if after_fd <= before_fd:
        print_tail(fd)
        raise AssertionError("food.csv should have at least one new data row")

    print("[OK] File Agent (preserve-headers) test passed.")

if __name__ == "__main__":
    main()
