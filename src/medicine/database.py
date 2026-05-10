import sqlite3

DB_NAME = "medicine.db"

def connect():
    return sqlite3.connect(DB_NAME)


def create_tables():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS compartments(
        section_id TEXT PRIMARY KEY,
        medicine_name TEXT NOT NULL,
        medicine_type TEXT,
        servo_angle INTEGER,
        expiry_date TEXT,
        current_quantity INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS schedule (
        schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
        section_id TEXT,
        time TEXT,
        dosage INTEGER,
        before_after_food TEXT,
        water_ml INTEGER,
        FOREIGN KEY (section_id) REFERENCES compartments(section_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS intake_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        schedule_id INTEGER,
        actual_time TEXT,
        status TEXT,
        FOREIGN KEY (schedule_id) REFERENCES schedule(schedule_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS water_profile (
        user_id        INTEGER PRIMARY KEY,
        age            INTEGER NOT NULL,
        gender         TEXT NOT NULL,
        weight_kg      REAL NOT NULL,
        height_cm      REAL NOT NULL,
        activity_level TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS water_alert_config (
        config_id          INTEGER PRIMARY KEY,
        alarm_enabled      INTEGER NOT NULL DEFAULT 1,
        mode               TEXT    NOT NULL DEFAULT 'smart',
        doctor_ml_per_slot INTEGER NOT NULL DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()


# COMPARTMENT FUNCTIONS

def add_compartment(section_id, medicine_name, medicine_type, servo_angle, expiry_date, current_quantity):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO compartments VALUES(?,?,?,?,?,?)",
                   (section_id, medicine_name, medicine_type, servo_angle, expiry_date, current_quantity))
    conn.commit()
    conn.close()


def get_all_compartments():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM compartments")
    data = cursor.fetchall()
    conn.close()
    return data


def get_compartment_by_id(section_id):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM compartments WHERE section_id=?", (section_id,))
    data = cursor.fetchone()
    conn.close()
    return data


def delete_compartment(section_id):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM compartments WHERE section_id=?", (section_id,))
    conn.commit()
    conn.close()


def update_compartment(section_id, medicine_name, medicine_type, servo_angle, expiry_date, current_quantity):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE compartments
        SET medicine_name=?, medicine_type=?, servo_angle=?,
            expiry_date=?, current_quantity=?
        WHERE section_id=?
    """, (medicine_name, medicine_type, servo_angle, expiry_date, current_quantity, section_id))
    conn.commit()
    conn.close()


# SCHEDULE FUNCTIONS

def get_schedule_by_id(schedule_id):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM schedule WHERE schedule_id=?", (schedule_id,))
    data = cursor.fetchone()
    conn.close()
    return data


def update_schedule(schedule_id, section_id, time, dosage, before_after_food, water_ml):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE schedule
        SET section_id=?, time=?, dosage=?, before_after_food=?, water_ml=?
        WHERE schedule_id=?
    """, (section_id, time, dosage, before_after_food, water_ml, schedule_id))
    conn.commit()
    conn.close()


def delete_schedule(schedule_id):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM schedule WHERE schedule_id=?", (schedule_id,))
    conn.commit()
    conn.close()


def add_schedule(section_id, time, dosage, before_after_food, water_ml):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO schedule (section_id, time, dosage, before_after_food, water_ml)
        VALUES (?,?,?,?,?)
    """, (section_id, time, dosage, before_after_food, water_ml))
    conn.commit()
    conn.close()


def get_schedule_by_time(current_time):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM schedule WHERE time=?", (current_time,))
    data = cursor.fetchall()
    conn.close()
    return data


# INTAKE LOG FUNCTIONS

def log_intake(schedule_id, actual_time, status):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO intake_log (schedule_id, actual_time, status)
        VALUES (?,?,?)
    """, (schedule_id, actual_time, status))
    conn.commit()
    conn.close()


# WATER PROFILE FUNCTIONS

def get_water_profile():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM water_profile WHERE user_id=1")
    data = cursor.fetchone()
    conn.close()
    return data


def save_water_profile(age, gender, weight_kg, height_cm, activity_level):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO water_profile (user_id, age, gender, weight_kg, height_cm, activity_level)
        VALUES (1,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            age=excluded.age, gender=excluded.gender,
            weight_kg=excluded.weight_kg, height_cm=excluded.height_cm,
            activity_level=excluded.activity_level
    """, (age, gender, weight_kg, height_cm, activity_level))
    conn.commit()
    conn.close()


def calculate_daily_water_ml(age, gender, weight_kg, height_cm, activity_level):
    if age <= 5:
        base = weight_kg * 100
    elif age >= 65:
        base = weight_kg * 30
    else:
        base = weight_kg * 35

    multipliers = {"light": 1.00, "moderate": 1.15, "high": 1.30}
    adjusted = base * multipliers.get(activity_level.lower(), 1.00)

    if gender.lower() == "male":
        adjusted += 200

    return int(round(adjusted / 50) * 50)


# WATER ALERT CONFIG FUNCTIONS

ALERT_SLOTS = [
    {"time": "06:00", "label": "Wake-up",        "weight": 0.10},
    {"time": "08:00", "label": "After breakfast", "weight": 0.10},
    {"time": "10:00", "label": "Mid-morning",     "weight": 0.15},
    {"time": "12:00", "label": "Before lunch",    "weight": 0.20},
    {"time": "14:00", "label": "After lunch",     "weight": 0.20},
    {"time": "16:00", "label": "Peak afternoon",  "weight": 0.15},
    {"time": "18:00", "label": "Early evening",   "weight": 0.10},
]

SLOT_REASONS = {
    "06:00": "You lose 400-600 ml overnight through breathing and perspiration. Morning rehydration restores blood volume and supports kidney function after the overnight fast.",
    "08:00": "Water with breakfast improves nutrient absorption and helps the body process morning medications more effectively.",
    "10:00": "Cortisol peaks around 8-9 AM and begins declining. A mid-morning drink sustains energy and prevents early dehydration that often goes unnoticed in elderly patients.",
    "12:00": "Pre-meal hydration supports digestion and helps regulate appetite. The body's metabolic demand is near its daily peak around midday.",
    "14:00": "Post-lunch digestion requires fluid. Afternoon hours also coincide with rising ambient temperature, increasing insensible fluid loss through the skin.",
    "16:00": "Research shows urine is least concentrated between 4-6 PM, reflecting the body's peak fluid turnover. This is the highest-priority hydration window of the day.",
    "18:00": "A moderate early-evening intake completes the day's target. Kept smaller than other slots to reduce nighttime bathroom trips — especially important for elderly patients.",
}


def get_alert_config():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM water_alert_config WHERE config_id=1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "alarm_enabled":      bool(row[1]),
            "mode":               row[2],
            "doctor_ml_per_slot": row[3],
        }
    return {"alarm_enabled": True, "mode": "smart", "doctor_ml_per_slot": 0}


def save_alert_config(alarm_enabled, mode, doctor_ml_per_slot):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO water_alert_config (config_id, alarm_enabled, mode, doctor_ml_per_slot)
        VALUES (1,?,?,?)
        ON CONFLICT(config_id) DO UPDATE SET
            alarm_enabled=excluded.alarm_enabled,
            mode=excluded.mode,
            doctor_ml_per_slot=excluded.doctor_ml_per_slot
    """, (1 if alarm_enabled else 0, mode, int(doctor_ml_per_slot)))
    conn.commit()
    conn.close()


def build_alert_schedule(recommended_ml, config):
    slots = []
    for slot in ALERT_SLOTS:
        if config["mode"] == "custom":
            ml = int(config["doctor_ml_per_slot"])
        else:
            ml = int(round((recommended_ml * slot["weight"]) / 10) * 10)
            ml = max(ml, 50)
        slots.append({
            "time":    slot["time"],
            "label":   slot["label"],
            "ml":      ml,
            "reason":  SLOT_REASONS[slot["time"]],
            "enabled": config["alarm_enabled"],
        })
    return slots


# UTILITY

def clear_compartments():
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM compartments")
    print("All compartments deleted")
    conn.commit()
    conn.close()


def reduce_quantity(section_id, dosage):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE compartments
        SET current_quantity = current_quantity - ?
        WHERE section_id = ?
    """, (dosage, section_id))
    conn.commit()
    conn.close()