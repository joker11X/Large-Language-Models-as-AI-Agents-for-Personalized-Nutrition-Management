from pathlib import Path
import os
import pandas as pd
import csv
import base64
from difflib import SequenceMatcher
import re, unicodedata, json
import time

recommend_dir = Path("./nutrition_data")
user_dir = Path("./user_data")

def list_recommend_files() -> str:
    """
    List all CSV files in the recommendation directory.
    Returns a string listing all available files, or an error message.
    """
    try:
        files = [f.name for f in recommend_dir.glob("*.csv")]
        if not files:
            return "No CSV files found in the recommendation directory."
        return "Available recommendation files:\n" + "\n".join(files)
    except Exception as e:
        return f"Failed to list files: {type(e).__name__}: {e}"
    

def list_user_files() -> str:
    """
    List all CSV files in the user directory.
    Returns a string listing all available files, or an error message.
    """
    try:
        files = [f.name for f in user_dir.glob("*.csv")]
        if not files:
            return "No CSV files found in the user directory."
        return "Available user files:\n" + "\n".join(files)
    except Exception as e:
        return f"Failed to list files: {type(e).__name__}: {e}"
    
def rename_file(name: str, new_name: str) -> str:
    print(f"(rename_file {name} -> {new_name})")
    try:
        new_path = user_dir / new_name
        if not str(new_path).startswith(str(user_dir)):
            return "Error: new_name is outside base_dir."

        os.makedirs(new_path.parent, exist_ok=True)
        os.rename(user_dir / name, new_path)
        return f"File '{name}' successfully renamed to '{new_name}'."
    except Exception as e:
        return f"An error occurred: {e}"

def create_user_data_subfolder(subfolder_name: str) -> str:
    """
    Create a subfolder inside the ./user_data directory.

    Parameters:
        subfolder_name (str): Name of the subfolder to create (e.g., 'u001' or 'u001/logs').

    Returns:
        str: Success or error message.
    """
    try:
        # 防止目录穿越
        if ".." in subfolder_name or subfolder_name.startswith("/"):
            return "Invalid subfolder name: directory traversal is not allowed."

        # 组合完整路径
        target_path = user_dir / subfolder_name

        # 检查是否仍在 user_data 范围内（安全检查）
        if not target_path.resolve().is_relative_to(user_dir.resolve()):
            return "Error: Folder must be within the './user_data' directory."

        # 创建文件夹
        if target_path.exists():
            return f"Folder '{target_path}' already exists."
        
        target_path.mkdir(parents=True, exist_ok=True)
        return f"Folder '{target_path}' created successfully."

    except Exception as e:
        return f"Failed to create folder: {type(e).__name__}: {e}"
    
def read_image_for_gpt(date: str, filename: str) -> bytes:
    """
    Read an image file under ./user_data/{date}/{filename} and return its bytes for GPT-4o input.

    Parameters:
        date (str): Date folder under user_data, e.g., "2024-06-21"
        filename (str): Image file name, e.g., "lunch.jpg"

    Returns:
        bytes: Image content in bytes, suitable for GPT-4o's files=[("image", ...)] input
    """
    try:
        # 安全性检查
        if ".." in date or ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid path components.")

        image_path = Path("./user_data") / date / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return image_path.read_bytes()
    
    except Exception as e:
        raise RuntimeError(f"Failed to read image for GPT-4o: {type(e).__name__}: {e}")
    

import chardet

def read_csv_file(name: str) -> str:
    """Return content of csv file. If not exist, return error message.
    """
    try:
        path = recommend_dir / name
        with open(path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']

        df = pd.read_csv(path, encoding=encoding)
        return df.to_markdown(index=False)
    except Exception as e:
        return f"Failed to read CSV '{name}': {type(e).__name__} - {str(e)}"

def read_user_csv(name: str) -> str:
    """
    Return content of a CSV under ./user_data as markdown.
    """
    try:
        file_path = user_dir / name
        if not file_path.exists():
            return f"Error: user CSV '{name}' does not exist."
        import chardet, pandas as pd
        with open(file_path, 'rb') as f:
            enc = chardet.detect(f.read())['encoding'] or 'utf-8'
        df = pd.read_csv(file_path, encoding=enc)
        return df.to_markdown(index=False)
    except Exception as e:
        return f"Failed to read user CSV '{name}': {type(e).__name__} - {e}"
    
CSV_ORDER = {
    "food.csv": [
        "food_name", "cuisine_category", "appearance_times"
    ],
    "every_meal.csv": [
        "Timestamp", "food_names_volumes",
        "Calcium (mg/d)", "Chromium (mcg/d)", "Copper (mg/d)", "Fluoride (mg/d)",
        "Iodine (mcg/d)", "Iron (mg/d)", "Magnesium (mg/d)", "Manganese (mg/d)",
        "Molybdenum (mcg/d)", "Phosphorus (mg/d)", "Selenium (mcg/d)", "Zinc (mg/d)",
        "Potassium (mg/d)", "Sodium (mg/d)", "Chloride (g/d)", "Total Water (L/d)",
        "Carbohydrate (g/d)", "Total Fiber (g/d)", "Fat (g/d)",
        "Linoleic Acid (g/d)", "alpha-Linolenic Acid (g/d)",
        "Protein (g/d)", "Vitamin A (mcg/d)", "Vitamin C (mg/d)", "Vitamin D (mcg/d)",
        "Vitamin E (mg/d)", "Vitamin K (mcg/d)", "Thiamin (mg/d)", "Riboflavin (mg/d)",
        "Niacin (mg/d)", "Vitamin B6 (mg/d)", "Folate (mcg/d)", "Vitamin B12 (mcg/d)",
        "Pantothenic Acid (mg/d)", "Biotin (mcg/d)", "Choline (mg/d)",
        "total_energy(kcal)"
    ],
    # 如需也盲写其它文件，按同样方式把表头贴进来：
    # "daily_nutrition_plan.csv": [...],
    # "plan_completion_rate.csv": [...]
}

def append_to_user_csv(name: str, row) -> str:
    """
    Append a row to the specified CSV file in the user directory.
    
    Parameters:
        name (str): The name of the CSV file (e.g., "adult.csv").
        row (dict): A dictionary representing one row to append (key = column name).
    
    Returns:
        str: Success or error message.
    """
    try:
        path = user_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)

        # 首次创建时写表头（若配置了）
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                header = CSV_ORDER.get(name)
                if header:
                    w.writerow(header)

        # 计算要写入的一行
        if isinstance(row, (list, tuple)):
            values = list(row)
        elif isinstance(row, dict):
            order = CSV_ORDER.get(name, list(row.keys()))  # 没配置就按传入顺序
            # 常见别名简单规整（可按需扩展）
            alias = {
                    "α-Linolenic Acid (g/d)": "alpha-Linolenic Acid (g/d)",
                    "Vitamin A (mcg RAE)": "Vitamin A (mcg/d)",
                    "Vitamin D (IU or mcg)": "Vitamin D (mcg/d)",
                    "Vitamin K (mcg)": "Vitamin K (mcg/d)",
                    "Sodium (mg)": "Sodium (mg/d)",
                    "Chloride (mg)": "Chloride (g/d)",
                    "Total Water (ml)": "Total Water (L/d)",
                }

            values = [row.get(col, row.get(alias.get(col, ""), "")) for col in order]
        else:
            return f"Failed to write to CSV: unsupported row type {type(row).__name__}"

        # 直接追加
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(values)

        return f"Row successfully added to {name}."
    except Exception as e:
        return f"Failed to write to CSV: {type(e).__name__}: {e}"

    

def read_txt_file(name: str) -> str:
    """
    读取 ./user_data/ 目录下的 txt 文件内容。
    
    参数:
        name (str): 文件名，例如 "notes.txt"
    
    返回:
        str: 文件内容，或错误信息
    """
    try:
        file_path = user_dir / name
        if not file_path.exists():
            return f"Error: TXT file '{name}' does not exist."
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read TXT '{name}': {type(e).__name__} - {str(e)}"


    
def write_txt_file(name: str, content: str):
    # 本函数禁止用于写 .csv
    """
    在 ./user_data/ 目录下写入 txt 文件内容（覆盖写入）。
    
    参数:
        name (str): 文件名，例如 "notes.txt"
        content (str): 要写入的文本内容
    
    返回:
        str: 成功或错误信息
    """
    if name.lower().endswith(".csv"):
        raise ValueError("write_txt_file cannot write CSV. Use append_to_user_csv instead.")
    path = user_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")
    return {"ok": True, "name": name}


def read_image_for_gemini(date: str, filename: str) -> str:
    """
    读取 ./user_data/{date}/{filename} 下的图片，并返回 base64 编码字符串，
    适合传给 Gemini API 作为 image input。
    """
    try:
        if ".." in date or ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid path components.")

        image_path = Path("./user_data") / date / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded

    except Exception as e:
        raise RuntimeError(f"Failed to read image for Gemini: {type(e).__name__}: {e}")
    
def _norm_food_name(s: str) -> str:
    """Normalize a food name for fuzzy matching (ASCII-safe, lower, space-collapse)."""
    s = unicodedata.normalize("NFKC", (s or "")).lower().strip()
    s = re.sub(r"[\u2019'’`\"]", "", s)            # drop quotes
    s = re.sub(r"[^a-z0-9\s\-/&]+", " ", s)       # keep ascii letters/digits/basic punct
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _parse_bracket_list(bracket_list: str) -> list[str]:
    """
    Parse Vision's bracket_list like "[mapo tofu, 380, steamed rice, 180]" → ["mapo tofu","steamed rice"].
    Assumes alternating: name, number, name, number...
    """
    if not bracket_list:
        return []
    b = bracket_list.strip()
    if b.startswith("[") and b.endswith("]"):
        b = b[1:-1]
    tokens = [t.strip() for t in b.split(",")]
    names = []
    for i in range(0, len(tokens), 2):   # even index → name
        name = tokens[i]
        if name:
            names.append(name)
    # de-dup within this meal (keep order)
    seen, out = set(), []
    for n in names:
        key = _norm_food_name(n)
        if key in seen: 
            continue
        seen.add(key)
        out.append(n)
    return out

def _parse_cuisines_line(line: str) -> dict:
    """
    "CUISINES: name=cuisine; name=cuisine" → {name: cuisine}
    """
    out = {}
    if not line or ":" not in line:
        return out
    body = line.split(":", 1)[1]
    for seg in body.split(";"):
        seg = seg.strip()
        if not seg or "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def upsert_food_csv_fuzzy(bracket_list: str, cuisines_line: str = "", threshold: str = "0.86") -> str:
    """
    Upsert long-term stats into ./user_data/food.csv with fuzzy name matching (no local mapping table).
    - bracket_list: REQUIRED, e.g. "[mapo tofu, 380, steamed rice, 180]".
    - cuisines_line: OPTIONAL, e.g. "CUISINES: mapo tofu=chinese food; steamed rice=global common".
    - threshold: OPTIONAL string float in [0,1], default "0.86".
    Returns: JSON string summary {threshold, matched:[...], created:[...], rows:<int>}
    """
    try:
        thr = float(threshold or "0.86")
    except Exception:
        thr = 0.86

    names = _parse_bracket_list(bracket_list or "")
    cuisines_map = _parse_cuisines_line(cuisines_line or "")

    # load current food.csv (if any)
    fpath = user_dir / "food.csv"
    if fpath.exists():
        import chardet
        with open(fpath, "rb") as f:
            enc = chardet.detect(f.read()).get("encoding") or "utf-8"
        df = pd.read_csv(fpath, encoding=enc)
    else:
        df = pd.DataFrame(columns=["food_name", "cuisine_category", "appearance_times"])

    # ensure columns & dtypes
    if "food_name" not in df.columns: df["food_name"] = ""
    if "cuisine_category" not in df.columns: df["cuisine_category"] = "unknown"
    if "appearance_times" not in df.columns: df["appearance_times"] = 0
    df["appearance_times"] = pd.to_numeric(df["appearance_times"], errors="coerce").fillna(0).astype(int)
    if "_norm" not in df.columns:
        df["_norm"] = df["food_name"].map(_norm_food_name)
    else:
        df["_norm"] = df["_norm"].fillna(df["food_name"].map(_norm_food_name))

    summary = {"threshold": thr, "matched": [], "created": []}

    for name in names:
        norm = _norm_food_name(name)

        # best match among existing rows
        best_idx, best_score = -1, 0.0
        for idx, row in df.iterrows():
            score = SequenceMatcher(None, norm, row["_norm"]).ratio()
            # containment bonus
            if norm and (norm in row["_norm"] or row["_norm"] in norm):
                score = max(score, 0.92)
            if score > best_score:
                best_score, best_idx = score, idx

        if best_idx != -1 and best_score >= thr:
            # merge with existing item
            df.at[best_idx, "appearance_times"] = int(df.at[best_idx, "appearance_times"]) + 1
            new_cuisine = cuisines_map.get(name)
            if new_cuisine and str(df.at[best_idx, "cuisine_category"]).strip().lower() in {"", "unknown", "null", "none"}:
                df.at[best_idx, "cuisine_category"] = new_cuisine
            summary["matched"].append({
                "name": name,
                "matched_to": str(df.at[best_idx, "food_name"]),
                "similarity": round(best_score, 3),
                "cuisine_used": new_cuisine or ""
            })
        else:
            # create a new row
            cuisine = cuisines_map.get(name, "unknown")
            df = pd.concat([df, pd.DataFrame([{
                "food_name": name,
                "cuisine_category": cuisine,
                "appearance_times": 1,
                "_norm": norm
            }])], ignore_index=True)
            summary["created"].append({"name": name, "cuisine": cuisine})

    # drop helper column & save
    if "_norm" in df.columns:
        df = df.drop(columns=["_norm"])
    fpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fpath, index=False, encoding="utf-8")

    summary["rows"] = int(len(df))
    return json.dumps(summary, ensure_ascii=False)


def read_user_csv_tail(name: str, user_id: str | None = None) -> str:
    import json, time, re
    from pathlib import Path
    import pandas as pd

    t0 = time.perf_counter()
    print(f"[TOOL] read_user_csv_tail start name={name} user_id={user_id}", flush=True)

    try:
        file_path = user_dir / name
        if not file_path.exists():
            print(f"[TOOL] read_user_csv_tail miss  name={name}", flush=True)
            return json.dumps({"error": f"user CSV '{name}' does not exist."})

        # 读文件（自动探测编码）
        try:
            import chardet
            with open(file_path, "rb") as f:
                enc = chardet.detect(f.read())["encoding"] or "utf-8"
        except Exception:
            enc = "utf-8"
        df = pd.read_csv(file_path, encoding=enc)

        # 可选：按 user_id 过滤
        if user_id and "user_id" in df.columns:
            df = df[df["user_id"].astype(str) == str(user_id)]

        # 若存在时间列，先排序
        sort_col = next((c for c in ["Date","Timestamp","datetime","time","date","timestamp"] if c in df.columns), None)
        if sort_col:
            df[sort_col] = pd.to_datetime(df[sort_col], errors="coerce")
            df = df.sort_values(sort_col, ascending=True)

        tail_df = df.tail(1)

        # 仅对 plan_completion_rate.csv 做“核心15项”的筛选；其余文件维持原样
        if name.strip().lower() == "plan_completion_rate.csv":
            # 丢弃所有时间相关列
            drop_cols = [c for c in tail_df.columns if any(k in str(c).lower() for k in ("timestamp","datetime","time","date"))]
            tail_df = tail_df.drop(columns=drop_cols, errors="ignore")

            # 目标15项（以“基础名”匹配，忽略单位/括号/大小写）
            CORE15 = [
                "Protein","Carbohydrate","Fat","Total Fiber","Total Water",
                "Sodium","Potassium","Calcium","Magnesium","Iron","Zinc",
                "Vitamin A","Vitamin C","Vitamin D","Vitamin B12"
            ]
            def norm_col(s: str) -> str:
                s = re.sub(r"\(.*?\)", "", str(s))      # 去单位括号
                s = s.replace("/d", "")                 # 诸如 g/d 的 /d
                s = re.sub(r"[^a-z0-9]+", " ", s.lower())
                return s.strip()

            cols = list(tail_df.columns)
            selected = {}
            for base in CORE15:
                b = norm_col(base)
                cands = [c for c in cols if b in norm_col(c)]
                if not cands:
                    continue
                # 优先 remaining/gap → intake/consumed → target/goal
                def _score(c):
                    nc = norm_col(c)
                    if "remain" in nc or "gap" in nc: return (3, -len(c))
                    if "intake" in nc or "consum" in nc: return (2, -len(c))
                    if "target" in nc or "goal" in nc: return (1, -len(c))
                    return (0, -len(c))
                best = sorted(cands, key=_score, reverse=True)[0]
                val = tail_df.iloc[0][best]
                # 尝试转数值
                try:
                    val = float(val)
                except Exception:
                    pass
                selected[base] = val

            records = [selected]
            dt = time.perf_counter() - t0
            print(f"[TOOL] read_user_csv_tail done  name={name} selected={list(selected.keys())} time={dt:.3f}s", flush=True)
            return json.dumps({"file": name, "rows": 1, "data": records}, ensure_ascii=False)

        # —— 其他文件：保持“最新一行 + 全列”，并把时间转 ISO，NaN→null —— 
        records = json.loads(tail_df.to_json(orient="records", date_format="iso"))
        dt = time.perf_counter() - t0
        print(f"[TOOL] read_user_csv_tail done  name={name} shape=1x{tail_df.shape[1]} time={dt:.3f}s", flush=True)
        return json.dumps({"file": name, "rows": 1, "data": records}, ensure_ascii=False)

    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[TOOL] read_user_csv_tail error name={name} time={dt:.3f}s err={type(e).__name__}: {e}", flush=True)
        return json.dumps({"error": f"Failed to read tail of user CSV '{name}': {type(e).__name__} - {e}"})
    
def read_food_csv(limit: int = 10, user_id: str | None = None) -> str:
    """
    Read the first `limit` rows from ./user_data/food.csv and return compact JSON.
    - Prints start/done logs with file size, shape, and elapsed time.
    - If 'user_id' column exists and user_id is provided, filter first.
    """
    import json, time
    import pandas as pd

    t0 = time.perf_counter()
    print(f"[TOOL] read_food_csv start name=food.csv limit={limit} user_id={user_id}", flush=True)

    try:
        file_path = user_dir / "food.csv"
        if not file_path.exists():
            print(f"[TOOL] read_food_csv miss  name=food.csv", flush=True)
            return json.dumps({"error": "user CSV 'food.csv' does not exist."})

        size = file_path.stat().st_size

        # autodetect encoding (fallback to utf-8)
        try:
            import chardet
            with open(file_path, "rb") as f:
                enc = chardet.detect(f.read())["encoding"] or "utf-8"
        except Exception:
            enc = "utf-8"

        df = pd.read_csv(file_path, encoding=enc)

        if user_id and "user_id" in df.columns:
            df = df[df["user_id"].astype(str) == str(user_id)]

        n = 10 if (limit is None or limit <= 0) else int(limit)
        head_df = df.head(n)

        # JSON-safe (dates -> ISO, NaN -> null)
        records = json.loads(head_df.to_json(orient="records", date_format="iso"))

        dt = time.perf_counter() - t0
        print(
            f"[TOOL] read_food_csv done  name=food.csv size={size}B "
            f"shape={head_df.shape[0]}x{head_df.shape[1]} time={dt:.3f}s "
            f"cols={list(head_df.columns)}",
            flush=True,
        )
        return json.dumps({"file": "food.csv", "rows": head_df.shape[0], "data": records}, ensure_ascii=False)

    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[TOOL] read_food_csv error name=food.csv time={dt:.3f}s err={type(e).__name__}: {e}", flush=True)
        return json.dumps({"error": f"Failed to read head of 'food.csv': {type(e).__name__} - {e}"})


