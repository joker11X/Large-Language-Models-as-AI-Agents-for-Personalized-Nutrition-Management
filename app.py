from flask import Flask, request, jsonify, send_from_directory
import csv
import os
from datetime import datetime
import pandas as pd
from agent import create_dialog_agent, create_vision_agent, create_file_agent, create_controller_agent
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import base64
import re
import time

DEMO_DELAY_MODE = True
DEMO_DELAY_SECONDS = 90   # 90 seconds (~1m30s). For demos, set to 3 for quick testing


# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
#debug
app.logger.setLevel('DEBUG')

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
UPLOAD_FOLDER = './picture'

# Global storage for chat histories
chat_histories = {}

# Initialize agent - default OpenAI; can be changed via parameter
agent = None

def initialize_agent(agent_type="dialog"):
    """Initialize the specified type of agent"""
    global agent
    if agent_type == "dialog":
        from agent import create_dialog_agent
        agent = create_dialog_agent()
    elif agent_type == "vision":
        from agent import create_vision_agent
        agent = create_vision_agent()
    elif agent_type == "file":
        from agent import create_file_agent
        agent = create_file_agent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

# Initialize agents (could be placed in app factory)
_dialog = create_dialog_agent()
_vision = create_vision_agent()
_file   = create_file_agent()
_ctrl   = create_controller_agent(_dialog, _vision, _file)

@app.post("/api/controller/ask")
def controller_ask():
    data = request.get_json(force=True)
    msg = data.get("message","").strip()
    user_id = data.get("user_id","default")
    print(f"[HTTP] /api/controller/ask user_id={user_id} msg={msg}", flush=True)

    try:
        res = _ctrl.handle_text(msg, user_id)
        # Important: this should print; if missing, likely stuck in handle_text
        print(f"[HTTP] /api/controller/ask result_keys={list(res.keys())} "
              f"route={res.get('route')} len={len(res.get('reply',''))}", flush=True)
        return jsonify({"ok": True, "data": res})
    except Exception as e:
        print(f"[HTTP] /api/controller/ask ERROR {type(e).__name__}: {e}", flush=True)
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


@app.post("/api/controller/meal")
def controller_meal():
    # multipart: image + optional message + user_id
    user_id = request.form.get("user_id","default")
    msg = request.form.get("message","").strip()
    file = request.files.get("image")
    if not file:
        return jsonify({"ok": False, "error":"no image"}), 400
    b64 = base64.b64encode(file.read()).decode("utf-8")
    res = _ctrl.handle_meal(msg, b64, user_id)
    return jsonify({"ok": True, "data": res})



def allowed_file(filename):
    """Check whether the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_daily_folder():
    """Create today's date folder"""
    today = datetime.now().strftime('%Y_%m_%d')
    daily_folder = os.path.join(UPLOAD_FOLDER, today)
    
    if not os.path.exists(daily_folder):
        os.makedirs(daily_folder, exist_ok=True)
        print(f"Create a new date folder: {daily_folder}")
    
    return daily_folder, today

def get_next_image_number(daily_folder, date_str):
    """Get next image index for today"""
    if not os.path.exists(daily_folder):
        return 1
    
    # Find today's existing image files
    existing_files = [f for f in os.listdir(daily_folder) 
                     if f.startswith(date_str) and os.path.isfile(os.path.join(daily_folder, f))]
    
    if not existing_files:
        return 1
    
    #  NOTE
    max_num = 0
    for filename in existing_files:
        try:
            # ,  "2025_08_12_01.jpg"
            parts = filename.split('_')
            if len(parts) >= 4:
                num_part = parts[3].split('.')[0]  #  NOTE
                num = int(num_part)
                max_num = max(max_num, num)
        except (ValueError, IndexError):
            continue
    
    return max_num + 1

def save_uploaded_image(file):
    """saveimage"""
    try:
        if not file or not allowed_file(file.filename):
            return None, ""
        
        #  NOTE
        daily_folder, date_str = create_daily_folder()
        
        #  NOTE
        next_num = get_next_image_number(daily_folder, date_str)
        
        #  NOTE
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        #  NOTE
        filename = f"{date_str}_{next_num:02d}.{file_extension}"
        filepath = os.path.join(daily_folder, filename)
        
        #  NOTE
        file.save(filepath)
        
        #  NOTE
        try:
            with Image.open(filepath) as img:
                # ,
                max_size = (1920, 1920)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    img.save(filepath, optimize=True, quality=85)
                    print(f"image: {filename}")
                    
        except Exception as e:
            print(f"image: {str(e)}")
        
        print(f"imagesaveSuccess: {filepath}")
        return filepath, None
        
    except Exception as e:
        print(f"saveimage: {str(e)}")
        return None, f"saveimageFailed: {str(e)}"

def image_to_base64(image_path):
    """imagebase64"""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"imagebase64: {str(e)}")
        return None

def match_nutrition_plan(user_data):
    """
    Pure-English matching:
      - category: 'Infant' | 'Pregnancy' | 'Lactation' | 'Other' | '' (empty means regular case)
      - gender:   'Male' | 'Female'
      - age:      years (float, can be <1)
    CSV columns expected: 'Category', 'Life-Stage Group' (English)
    """
    try:
        nutrition_file = './nutrition_data/merged_table.csv'
        if not os.path.exists(nutrition_file):
            print(f"[match] nutrition file not found: {nutrition_file}")
            return None

        import pandas as pd
        df = pd.read_csv(nutrition_file, encoding='utf-8')
        print(f"[match] loaded nutrition table: {len(df)} rows")

        cat = (user_data.get('category') or '').strip()        # 'Infant' / 'Pregnancy' / 'Lactation' / 'Other' / ''
        gender = (user_data.get('gender') or '').strip()       # 'Male' / 'Female'
        try:
            age = float(user_data.get('age', 0))
        except Exception:
            age = 0.0

        gender_category = 'Males' if gender == 'Male' else ('Females' if gender == 'Female' else '')
        matched = None

        # ---- Special cases first ----
        if cat == 'Infant' or age < 1:
            # 06 mo vs 712 mo by Life-Stage Group text
            if age <= 0.5:
                cond = (df['Category'] == 'Infants') & (df['Life-Stage Group'].str.contains('0-6', na=False))
            else:
                cond = (df['Category'] == 'Infants') & (df['Life-Stage Group'].str.contains('7-12', na=False))
            tmp = df[cond]
            matched = tmp.iloc[0] if not tmp.empty else None
            print("[match] case: Infant")

        elif cat == 'Pregnancy':
            if 14 <= age <= 18:
                cond = (df['Category'] == 'Pregnancy') & (df['Life-Stage Group'].str.contains('14-18', na=False))
            elif 19 <= age <= 30:
                cond = (df['Category'] == 'Pregnancy') & (df['Life-Stage Group'].str.contains('19-30', na=False))
            else:
                cond = (df['Category'] == 'Pregnancy') & (df['Life-Stage Group'].str.contains('31-50', na=False))
            tmp = df[cond]
            matched = tmp.iloc[0] if not tmp.empty else None
            print("[match] case: Pregnancy")

        elif cat == 'Lactation':
            if 14 <= age <= 18:
                cond = (df['Category'] == 'Lactation') & (df['Life-Stage Group'].str.contains('14-18', na=False))
            elif 19 <= age <= 30:
                cond = (df['Category'] == 'Lactation') & (df['Life-Stage Group'].str.contains('19-30', na=False))
            else:
                cond = (df['Category'] == 'Lactation') & (df['Life-Stage Group'].str.contains('31-50', na=False))
            tmp = df[cond]
            matched = tmp.iloc[0] if not tmp.empty else None
            print("[match] case: Lactation")

        else:
            # ---- Regular cases by age + gender ----
            if 1 <= age <= 3:
                cond = (df['Category'] == 'Children') & (df['Life-Stage Group'].str.contains('1-3', na=False))
                print("[match] auto: Children 13")
            elif 4 <= age <= 8:
                cond = (df['Category'] == 'Children') & (df['Life-Stage Group'].str.contains('4-8', na=False))
                print("[match] auto: Children 48")
            else:
                # 9+ use Males/Females groups
                if not gender_category:
                    print("[match] missing gender for 9+ years")
                    return None
                if 9 <= age <= 13:
                    cond = (df['Category'] == gender_category) & (df['Life-Stage Group'].str.contains('9-13', na=False))
                elif 14 <= age <= 18:
                    cond = (df['Category'] == gender_category) & (df['Life-Stage Group'].str.contains('14-18', na=False))
                elif 19 <= age <= 30:
                    cond = (df['Category'] == gender_category) & (df['Life-Stage Group'].str.contains('19-30', na=False))
                elif 31 <= age <= 50:
                    cond = (df['Category'] == gender_category) & (df['Life-Stage Group'].str.contains('31-50', na=False))
                elif 51 <= age <= 70:
                    cond = (df['Category'] == gender_category) & (df['Life-Stage Group'].str.contains('51-70', na=False))
                else:
                    cond = (df['Category'] == gender_category) & (df['Life-Stage Group'].str.contains('>70', na=False))
                tmp = df[cond]
                matched = tmp.iloc[0] if not tmp.empty else None
                print(f"[match] auto: {gender_category}")

        if matched is not None:
            print("[match] success")
            return matched

        print("[match] no match found (check column values in merged_table.csv)")
        return None

    except Exception as e:
        print(f"[match] error: {e}")
        return None


# app.py -- REPLACE the whole function
def calculate_EER(user_data):
    """
    Calculate EER (Estimated Energy Requirement) with ENGLISH category keys.
      category: 'Infant' | 'Pregnancy' | 'Lactation' | 'Other' | ''(none)
      gender:   'Male' | 'Female'   (Chinese '/' tolerated)
      activity_level: 'inactive' | 'low active' | 'active' | 'very active'
    """
    try:
        # normalize gender/activity
        gender_map = {'': 'Male', '': 'Female'}
        gender_raw = user_data.get('gender', '')
        gender = gender_map.get(gender_raw, gender_raw)

        age    = float(user_data.get('age', 0))      # years (can be <1)
        height = float(user_data.get('height', 0))   # cm
        weight = float(user_data.get('weight', 0))   # kg

        activity_level_map = {
            '': 'inactive',
            '': 'low active',
            '': 'active',
            '': 'very active'
        }
        activity_level_raw = user_data.get('activity_level', 'inactive')
        activity_level = activity_level_map.get(activity_level_raw, activity_level_raw)

        # category now uses ENGLISH keys only
        category = (user_data.get('category') or '').strip()

        print(f"[EER] params gender={gender}, age={age}, height={height}, weight={weight}, "
              f"activity={activity_level}, category={category}")

        # map to equation strings used below
        activity_map = {
            'inactive': 'Inactive',
            'low active': 'Low active',
            'active': 'Active',
            'very active': 'Very active'
        }
        activity_type = activity_map.get(activity_level, 'Inactive')

        # ---------- cohorts & special cases ----------
        # Infant/toddler: honor explicit 'Infant', also treat age<3 as infant/toddler block
        if category == 'Infant' or age < 3:
            if age <= 2.99/12:  # 02.99 months
                if gender == 'Male':
                    eer = -716.45 - (1.00 * age) + (17.82 * height) + (15.06 * weight) + 200
                else:
                    eer = -69.15 + (80.0 * age) + (2.65 * height) + (54.15 * weight) + 180
            elif age <= 5.99/12:  # 35.99 months
                if gender == 'Male':
                    eer = -716.45 - (1.00 * age) + (17.82 * height) + (15.06 * weight) + 50
                else:
                    eer = -69.15 + (80.0 * age) + (2.65 * height) + (54.15 * weight) + 60
            else:  # 6 months  2.99 years
                if gender == 'Male':
                    eer = -716.45 - (1.00 * age) + (17.82 * height) + (15.06 * weight) + 20
                else:
                    if age <= 11.99/12:
                        eer = -69.15 + (80.0 * age) + (2.65 * height) + (54.15 * weight) + 20
                    else:
                        eer = -69.15 + (80.0 * age) + (2.65 * height) + (54.15 * weight) + 15

        elif category == 'Pregnancy' and gender == 'Female':
            # Pregnancy (assume mid/late gestation ~25 weeks as before)
            gestation = 25
            energy_deposition = 200
            if activity_type == 'Inactive':
                eer = 1131.20 - (2.04 * age) + (0.34 * height) + (12.15 * weight) + (9.16 * gestation) + energy_deposition
            elif activity_type == 'Low active':
                eer = 693.35 - (2.04 * age) + (5.73 * height) + (10.20 * weight) + (9.16 * gestation) + energy_deposition
            elif activity_type == 'Active':
                eer = -223.84 - (2.04 * age) + (13.23 * height) + (8.15 * weight) + (9.16 * gestation) + energy_deposition
            else:  # Very active
                eer = -779.72 - (2.04 * age) + (18.45 * height) + (8.73 * weight) + (9.16 * gestation) + energy_deposition

        elif category == 'Lactation' and gender == 'Female':
            # Lactation (06 months exclusively breastfeeding)
            milk_energy = 540
            energy_mobilization = 140
            if age >= 19:  # adult female
                if activity_type == 'Inactive':
                    eer = 584.90 - (7.01 * age) + (5.72 * height) + (11.71 * weight) + milk_energy - energy_mobilization
                elif activity_type == 'Low active':
                    eer = 575.77 - (7.01 * age) + (6.60 * height) + (12.14 * weight) + milk_energy - energy_mobilization
                elif activity_type == 'Active':
                    eer = 710.25 - (7.01 * age) + (6.54 * height) + (12.34 * weight) + milk_energy - energy_mobilization
                else:
                    eer = 511.83 - (7.01 * age) + (9.07 * height) + (12.56 * weight) + milk_energy - energy_mobilization
            else:  # adolescent lactation (<19 years)
                if activity_type == 'Inactive':
                    eer = 55.59 - (22.25 * age) + (8.43 * height) + (17.07 * weight) + milk_energy - energy_mobilization
                elif activity_type == 'Low active':
                    eer = -297.54 - (22.25 * age) + (12.77 * height) + (14.73 * weight) + milk_energy - energy_mobilization
                elif activity_type == 'Active':
                    eer = -189.55 - (22.25 * age) + (11.74 * height) + (18.34 * weight) + milk_energy - energy_mobilization
                else:
                    eer = -709.59 - (22.25 * age) + (18.22 * height) + (14.25 * weight) + milk_energy - energy_mobilization

        elif 3 <= age < 19:
            # Children & adolescents (318 years)
            if gender == 'Male':
                if activity_type == 'Inactive':
                    # keep small growth adjustments like original
                    growth_energy = 20 if age == 3 else (15 if 4 <= age <= 8 else 25)
                    eer = -447.51 + (3.68 * age) + (13.01 * height) + (13.15 * weight) + growth_energy
                elif activity_type == 'Low active':
                    eer = 19.12 + (3.68 * age) + (8.62 * height) + (20.28 * weight)
                elif activity_type == 'Active':
                    eer = -388.19 + (3.68 * age) + (12.66 * height) + (20.46 * weight)
                else:  # Very active
                    eer = -671.75 + (3.68 * age) + (15.38 * height) + (23.25 * weight)
            else:  # Female
                if activity_type == 'Inactive':
                    if 3 <= age <= 13.99:
                        growth_energy = 15 if 3 <= age <= 8 else 30
                        eer = 55.59 - (22.25 * age) + (8.43 * height) + (17.07 * weight) + growth_energy
                    else:
                        eer = 55.59 - (22.25 * age) + (8.43 * height) + (17.07 * weight) + 20
                elif activity_type == 'Low active':
                    eer = -297.54 - (22.25 * age) + (12.77 * height) + (14.73 * weight) + (15 if 3 <= age <= 13.99 else 20)
                elif activity_type == 'Active':
                    eer = -189.55 - (22.25 * age) + (11.74 * height) + (18.34 * weight) + (15 if 3 <= age <= 13.99 else 20)
                else:
                    eer = -709.59 - (22.25 * age) + (18.22 * height) + (14.25 * weight) + (15 if 3 <= age <= 13.99 else 20)

        else:
            # Adults (19 years)
            if gender == 'Male':
                if activity_type == 'Inactive':
                    eer = 753.07 - (10.83 * age) + (6.50 * height) + (14.10 * weight)
                elif activity_type == 'Low active':
                    eer = 581.47 - (10.83 * age) + (8.30 * height) + (14.94 * weight)
                elif activity_type == 'Active':
                    eer = 1004.82 - (10.83 * age) + (6.52 * height) + (15.91 * weight)
                else:  # Very active
                    eer = -517.88 - (10.83 * age) + (15.61 * height) + (19.11 * weight)
            else:
                if activity_type == 'Inactive':
                    eer = 584.90 - (7.01 * age) + (5.72 * height) + (11.71 * weight)
                elif activity_type == 'Low active':
                    eer = 575.77 - (7.01 * age) + (6.60 * height) + (12.14 * weight)
                elif activity_type == 'Active':
                    eer = 710.25 - (7.01 * age) + (6.54 * height) + (12.34 * weight)
                else:
                    eer = 511.83 - (7.01 * age) + (9.07 * height) + (12.56 * weight)

        eer = round(eer)
        print(f"[EER] computed: {eer} kcal/day")
        return eer

    except Exception as e:
        print(f"[EER] error: {e}")
        return 2000


def save_nutrition_plan(user_id, matched_plan, user_data):
    """Docstring in English only."""
    try:
        #  NOTE
        os.makedirs('user_data', exist_ok=True)
        
        #  NOTE
        plan_file = './user_data/daily_nutrition_plan.csv'
        file_exists = os.path.isfile(plan_file)
        
        #  NOTE
        current_date = datetime.now().strftime('%Y-%m-%d')
        eer = calculate_EER(user_data)
        
        #  NOTE
        plan_data = {
            'Date': current_date,
            'Plan_Category': 'Base',
            'user_id': user_id,
            'total_energy(kcal)': eer
        }
        
        # (3)
        for column in matched_plan.index[2:]:  # Category  Life-Stage Group
            plan_data[column] = matched_plan[column]
        
        # CSV
        with open(plan_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=plan_data.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(plan_data)
        
        print(f"Successfully saved nutrition plan to {plan_file}")
        return True
        
    except Exception as e:
        print(f"Error occurred while saving nutrition plan: {str(e)}")
        return False


@app.route('/')
def serve_frontend():
    """"""
    # :
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)
    print(f"Current directory: {current_dir}")
    print(f"Files in directory: {files_in_dir}")
    
    # HTML
    html_files = [f for f in files_in_dir if f.endswith('.html')]
    print(f"HTML files found: {html_files}")
    
    # HTML
    possible_files = ['webpage.html', 'index.html', 'main.html']
    
    for filename in possible_files:
        if filename in files_in_dir:
            print(f"Serving file: {filename}")
            return send_from_directory('.', filename)
    
    # HTML, HTML
    return """Docstring in English only.""".format(current_dir, ', '.join(files_in_dir))

@app.route('/manifest.webmanifest')
def manifest():
    return send_from_directory('.', 'manifest.webmanifest',
                               mimetype='application/manifest+json')

@app.route('/sw.js')
def service_worker():
    # Service Worker ,
    return send_from_directory('.', 'sw.js',
                               mimetype='application/javascript')

@app.route('/icons/<path:filename>')
def icons(filename):
    # /icons/**
    return send_from_directory('icons', filename)


@app.route('/api/nutrition-plan', methods=['GET'])
def get_nutrition_plan():
    """Docstring in English only."""
    plan_file = './user_data/daily_nutrition_plan.csv'
    user_id = request.args.get('user_id', '').strip()

    if not os.path.exists(plan_file) or os.path.getsize(plan_file) == 0:
        return jsonify({'success': False, 'error': 'nutrition plan not found'})

    # ( user_id )
    latest = None
    with open(plan_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if user_id and row.get('user_id') != user_id:
                continue
            latest = row  # ->

    if not latest:
        return jsonify({'success': False, 'error': 'no plan for this user'})

    def base_name(col):
        # ,  "Calcium (mg)" -> "Calcium"
        return re.sub(r'\s*\(.*?\)\s*', '', col).strip()

    # ("", )
    macros = {
        'Carbohydrate', 'Total Fiber', 'Protein', 'Fat',
        'Saturated fatty acids', 'Trans fatty acids',
        '-Linolenic Acid', 'Linoleic Acid',
        'Dietary Cholesterol', 'Total Water'
    }
    vitamins = {
        'Vitamin A', 'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K',
        'Thiamin', 'Riboflavin', 'Niacin', 'Vitamin B6', 'Vitamin B12',
        'Folate', 'Choline', 'Pantothenic Acid', 'Biotin', 'Carotenoids'
    }
    minerals = {
        'Calcium', 'Chloride', 'Chromium', 'Copper', 'Fluoride', 'Iodine',
        'Iron', 'Magnesium', 'Manganese', 'Molybdenum', 'Phosphorus',
        'Potassium', 'Selenium', 'Sodium', 'Zinc'
    }

    #  NOTE
    def pick(items):
        out = []
        for k, v in latest.items():
            if k in ('Date', 'Plan_Category', 'user_id'):  #  NOTE
                continue
            name = base_name(k)
            if name in items and v not in (None, '', 'NA', 'ND'):
                # ()
                m = re.search(r'\((.*?)\)', k)
                unit = m.group(1) if m else ''
                out.append({'name': name, 'value': v, 'unit': unit})
        # :  name
        out.sort(key=lambda x: x['name'])
        return out

    payload = {
        'date': latest.get('Date'),
        'user_id': latest.get('user_id'),
        'eer': latest.get('total_energy(kcal)'),  # :contentReference[oaicite:3]{index=3}
        'groups': {
            'macronutrients': pick(macros),
            'vitamins': pick(vitamins),
            'minerals': pick(minerals),
        }
    }
    return jsonify({'success': True, 'data': payload})

@app.route('/api/clear-user-data', methods=['POST'])
def clear_user_data():
    """, """
    try:
        #  NOTE
        user_csv = 'user_data/user.csv'
        plan_csv = 'user_data/daily_nutrition_plan.csv'
        
        if os.path.exists(user_csv):
            os.remove(user_csv)
            print("")
            
        if os.path.exists(plan_csv):
            os.remove(plan_csv)
            print("nutrition plan")
        
        #  NOTE
        global chat_histories
        chat_histories.clear()
        
        return jsonify({'success': True, 'message': 'User data cleared'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/check-user-data', methods=['GET'])
def check_user_data():
    """"""
    try:
        csv_file = 'user_data/user.csv'
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            #  NOTE
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                user_data = None
                for row in reader:
                    user_data = row  #  NOTE
                
                if user_data:
                    return jsonify({'exists': True, 'user_data': user_data})
        
        return jsonify({'exists': False})
    
    except Exception as e:
        return jsonify({'exists': False, 'error': str(e)})


@app.route('/api/save-user-data', methods=['POST'])
def save_user_data():
    """savenutrition plan"""
    try:
        data = request.json
        
        os.makedirs('user_data', exist_ok=True)
        
        #  NOTE
        csv_file = 'user_data/user.csv'
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as file:
            fieldnames = ['user_id', 'user_name', 'gender', 'category', 'age', 'weight', 'height', 'BMI', 'activity_level', 'registration_time']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'user_id': data['user_id'],
                'user_name': data['user_name'],
                'gender': data['gender'],
                'category': data['category'],
                'age': data['age'],
                'weight': data['weight'],
                'height': data['height'],
                'BMI': data['BMI'],  
                'activity_level': data['activity_level'],
                'registration_time': data['registration_time']
            })
        
        print(f": {data['user_id']}")
        
        #  NOTE
        matched_plan = match_nutrition_plan(data)
        nutrition_plan_saved = False
        
        if matched_plan is not None:
            nutrition_plan_saved = save_nutrition_plan(data['user_id'], matched_plan, data)
            if nutrition_plan_saved:
                print("nutrition plansaveSuccess")
            else:
                print("nutrition plansaveFailed")
        else:
            print("nutrition plan, save")
        
        response_data = {
            'success': True, 
            'user_id': data['user_id'],
            'nutrition_plan_matched': matched_plan is not None,
            'nutrition_plan_saved': nutrition_plan_saved
        }
        
        # ,
        if matched_plan is not None:
            response_data['matched_plan_info'] = {
                'category': matched_plan['Category'],
                'life_stage': matched_plan['Life-Stage Group']
            }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"save: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    """AI - image"""
    try:
        #  NOTE
        if request.content_type and 'multipart/form-data' in request.content_type:
            #  NOTE
            user_message = request.form.get('message', '')
            user_id = request.form.get('user_id', 'default')
            uploaded_file = request.files.get('image')
            
            print(f" - ID: {user_id}, : {user_message}, image: {uploaded_file is not None}")
            
            #  NOTE
            image_info = None
            if uploaded_file and uploaded_file.filename:
                image_path, error = save_uploaded_image(uploaded_file)
                if image_path:
                    image_info = {
                        'path': image_path,
                        'filename': os.path.basename(image_path)
                    }
                    print(f"imagesaveSuccess: {image_path}")
                else:
                    print(f"imagesaveFailed: {error}")
                    return jsonify({
                        'success': False,
                        'error': f'imageFailed: {error}'
                    })
            
            # AI
            ai_message = user_message
            if image_info and not user_message.strip():
                ai_message = "analysisimage, . "
            elif image_info:
                ai_message = f"[imageanalysis] {user_message}"
            
            #  NOTE
            if user_id not in chat_histories:
                chat_histories[user_id] = []
            
            # ,
            if image_info:
                try:
                    #  NOTE
                    if agent.__class__.__name__ != 'VisionAgent':  #  NOTE
                        initialize_agent("vision")
                    
                    # base64
                    image_b64 = image_to_base64(image_info['path'])
                    if image_b64:
                        #  NOTE
                        vision_message = {
                            'text': ai_message,
                            'image': image_b64,
                            'image_path': image_info['path']
                        }
                        resp = agent.run_sync(vision_message, message_history=chat_histories[user_id])
                    else:
                        resp = agent.run_sync(f", imageFailed. {ai_message}", message_history=chat_histories[user_id])
                        
                except Exception as vision_error:
                    print(f"Failed, : {str(vision_error)}")
                    #  NOTE
                    if agent.__class__.__name__ != 'DialogAgent':
                        initialize_agent("dialog")
                    resp = agent.run_sync(f"({image_info['filename']}), {ai_message}", 
                                        message_history=chat_histories[user_id])
            else:
                #  NOTE
                resp = agent.run_sync(ai_message, message_history=chat_histories[user_id])
            
        else:
            # JSON()
            data = request.json
            user_message = data.get('message', '')
            user_id = data.get('user_id', 'default')
            
            print(f" - ID: {user_id}, : {user_message}")
            
            #  NOTE
            if user_id not in chat_histories:
                chat_histories[user_id] = []
            
            #  NOTE
            if agent.__class__.__name__ != 'DialogAgent':
                initialize_agent("dialog")
            
            # AI
            resp = agent.run_sync(user_message, message_history=chat_histories[user_id])
        
        #  NOTE
        chat_histories[user_id] = list(resp.all_messages())
        
        return jsonify({
            'success': True,
            'response': resp.output
        })
        
    except Exception as e:
        print(f": {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """imageAPI"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': ''})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': ''})
        
        image_path, error = save_uploaded_image(file)
        if image_path:
            return jsonify({
                'success': True, 
                'image_path': image_path,
                'filename': os.path.basename(image_path)
            })
        else:
            return jsonify({'success': False, 'error': error})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/get-image/<path:image_path>')
def get_image(image_path):
    """saveimage"""
    try:
        #  NOTE
        safe_path = secure_filename(image_path)
        full_path = os.path.join(UPLOAD_FOLDER, safe_path)
        
        if os.path.exists(full_path):
            return send_from_directory(UPLOAD_FOLDER, safe_path)
        else:
            return jsonify({'error': 'Image not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-nutrition-plan/<user_id>', endpoint='get_nutrition_plan_legacy')
def get_nutrition_plan_legacy(user_id):
    """Get user's nutrition plan"""
    try:
        plan_file = './user_data/daily_nutrition_plan.csv'
        if not os.path.exists(plan_file):
            return jsonify({'success': False, 'error': 'Nutrition plan file does not exist'})
        
        #  NOTE
        df = pd.read_csv(plan_file, encoding='utf-8')
        user_plans = df[df['user_id'] == user_id]
        
        if len(user_plans) == 0:
            return jsonify({'success': False, 'error': 'No nutrition plan found for this user'})
        
        # Return the latest plan
        latest_plan = user_plans.iloc[-1]
        plan_data = latest_plan.to_dict()
        
        return jsonify({
            'success': True,
            'nutrition_plan': plan_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/get-image-history/<user_id>')
def get_image_history(user_id):
    """Get user's uploaded image history"""
    try:
        today = datetime.now().strftime('%Y_%m_%d')
        daily_folder = os.path.join(UPLOAD_FOLDER, today)
        
        if not os.path.exists(daily_folder):
            return jsonify({'success': True, 'images': []})
        
        # Get all images for today
        images = []
        for filename in os.listdir(daily_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                file_path = os.path.join(daily_folder, filename)
                file_stat = os.stat(file_path)
                images.append({
                    'filename': filename,
                    'upload_time': datetime.fromtimestamp(file_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                    'size': file_stat.st_size,
                    'path': os.path.join(today, filename)
                })
        
        # Sort by upload time
        images.sort(key=lambda x: x['upload_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'images': images
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def run_server(clean_data=False, agent_type="dialog"):
    """Start Flask server"""
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize the specified type of agent
    initialize_agent(agent_type)
    
    if clean_data:
        print("Starting web server with clean user data...")
        # Clear existing user data
        user_csv = 'user_data/user.csv'
        plan_csv = 'user_data/daily_nutrition_plan.csv'
        
        if os.path.exists(user_csv):
            os.remove(user_csv)
            print("User data cleared")
            
        if os.path.exists(plan_csv):
            os.remove(plan_csv)
            print("Nutrition plan data cleared")
            
        chat_histories.clear()
    
    print(f"Starting web server with {agent_type.upper()} agent...")
    print("Web interface available at http://localhost:5000")
    print(f"Image upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)


if __name__ == "__main__":
    run_server()