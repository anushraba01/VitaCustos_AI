import sqlite3
import time
import os
import subprocess
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler

from gpiozero import AngularServo, OutputDevice

import board
import busio
from PIL import Image, ImageDraw, ImageFont

import Adafruit_DHT
import adafruit_ssd1306

PIN_SERVO_SIGNAL = 12   
PIN_DHT22_DATA   = 4    # 1-wire data line
PIN_PUMP_RELAY   = 23   # Relay coil: switches pump motor power

DEF_POS = 0
S1_WATER = 45    # Servo angle for water dispensing position
S2_MED   = 135   # Servo angle for medicine compartment (override per-compartment)

PUMP_ML_PER_SEC = 5   # ml delivered per second of pump run time

DHT_WARMUP_MINUTES = 3

pump_relay  = OutputDevice(PIN_PUMP_RELAY,  active_high=False, initial_value=False)

servo = AngularServo(PIN_SERVO_SIGNAL, min_angle=0, max_angle=180)

_oled  = None
_draw  = None
_font  = None
_image = None

def power_on_pump():
    pump_relay.on()
    print("[PUMP] Power ON")

def power_off_pump():
    pump_relay.off()
    print("[PUMP] Power OFF")


#oled
def init_oled():
    
    global _oled, _draw, _font, _image
    if _oled is not None:
        return
    
    i2c    = busio.I2C(board.SCL, board.SDA)
    _oled  = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)
    _image = Image.new("1", (128, 64))
    _draw  = ImageDraw.Draw(_image)
    _font  = ImageFont.load_default()

def oled_display(line1="", line2="", line3="", line4=""):
   
    if _oled is None:
        print(f"[OLED-OFF] {line1} | {line2} | {line3} | {line4}")
        return

    _draw.rectangle((0, 0, 128, 64), outline=0, fill=0)
    if line1: _draw.text((0, 0),  line1, font=_font, fill=255)
    if line2: _draw.text((0, 14), line2, font=_font, fill=255)
    if line3: _draw.text((0, 28), line3, font=_font, fill=255)
    if line4: _draw.text((0, 42), line4, font=_font, fill=255)
    _oled.image(_image)
    _oled.show()

def oled_clear():
    """Clears the OLED screen."""
    init_oled()
    _oled.fill(0)
    _oled.show()

#dht22
def read_dht22():
    for attempt in range(3):
        humidity, temp = Adafruit_DHT.read(Adafruit_DHT.DHT22, PIN_DHT22_DATA)
        if humidity is not None and temp is not None:
            temp     = max(10.0, min(float(temp),     50.0))
            humidity = max(10.0, min(float(humidity), 100.0))
            print(f"[DHT22] T:{temp:.1f}C  H:{humidity:.1f}%")
            return round(temp, 1), round(humidity, 1)
        time.sleep(2)
 
    print("[DHT22] Read failed 3 times — using fallback (27.0, 55.0)")
    return 27.0, 55.0

#audio- max98375A
def speak(text):
    
    print(f"[SPEAKER] {text}")
    os.system(f'espeak -s 130 -v en "{text}"')
    
def play_beep(count=1):
    
    for _ in range(count):
        try:
            subprocess.run(
                ["play", "-n", "synth", "0.3", "sine", "880"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            time.sleep(0.2)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass   # sox not installed — skip beep silently

def move_servo(angle):
    
    print(f"[SERVO] Moving to {angle}°")
    servo.angle = angle
    time.sleep(5)       # wait for servo to reach position
 
def fill_water(water_ml):
    
    run_seconds = max(1, round(water_ml / PUMP_ML_PER_SEC))
    print(f"[PUMP] Target: {water_ml} ml → running {run_seconds}s at {PUMP_ML_PER_SEC} ml/s")

    power_on_pump()
    time.sleep(run_seconds)
    power_off_pump()

    print(f"[PUMP] Done — ~{water_ml} ml dispensed.")

def dispense_medicine_and_water(section_id,water_ml):
    
    comp = get_compartment(section_id)
    # if not comp:
    #     print(f"[SERVO] Compartment {section_id} not found in DB.")
    #     return

    med_angle    = comp[3]   # servo_angle column
    medicine = comp[1]   # medicine_name column

    move_servo(DEF_POS)
 
    # oled_display("Preparing water", f"For {medicine}", "Please wait...")
    move_servo(S1_WATER)
 
    oled_display("Filling water...", f"{water_ml} ml", "Take the glass")
    fill_water(water_ml)
 
    speak("Water is ready. Please take the glass.")
    # oled_display("Take the glass", "Waiting 10s...", "Then medicine")
    time.sleep(10)
 
    oled_display("Take medicine:", medicine, f"Section {section_id}")
    speak(f"Please take your medicine. {medicine}.")
    move_servo(med_angle)
     
    oled_display("Take medicine now", medicine, "45 sec to close")
    time.sleep(45)
 
    move_servo(S1_WATER)
    time.sleep(1)

    move_servo(DEF_POS)
    print("complete.")

def dispense_water_only(water_ml, slot_label):
        
    move_servo(DEF_POS)
 
    oled_display(slot_label, "Moving to water...", "Please wait")
    move_servo(S1_WATER)
 
    # print(f"[SEQUENCE] Step 3 — dispensing {water_ml}ml")
    oled_display(slot_label, f"Filling: {water_ml} ml", "Take the glass")
    fill_water(water_ml)
 
    # print("[SEQUENCE] Step 4 — waiting for user to drink")
    speak(f"Water is ready. Please drink {water_ml} millilitres.")
    # oled_display("Drink up!", f"{water_ml} ml", "Take your time")
    time.sleep(20)
 
    move_servo(DEF_POS)
 
    

DB_NAME = "medicine.db"

def connect():
    return sqlite3.connect(DB_NAME)

def get_all_schedules():
    conn   = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM schedule")
    data   = cursor.fetchall()
    conn.close()
    return data

def get_compartment(section_id):
    conn   = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM compartments WHERE section_id=?", (section_id,))
    data   = cursor.fetchone()
    conn.close()
    return data

def log_intake(schedule_id):
    conn   = connect()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO intake_log VALUES (NULL,?,?,?)",
        (schedule_id, datetime.now().strftime("%H:%M"), "taken")
    )
    conn.commit()
    conn.close()

def get_water_profile():
    conn   = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM water_profile WHERE user_id=1")
    data   = cursor.fetchone()
    conn.close()
    return data

def get_alert_config():
    conn   = connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM water_alert_config WHERE config_id=1")
    row    = cursor.fetchone()
    conn.close()
    if row:
        return {
            "alarm_enabled":      bool(row[1]),
            "mode":               row[2],
            "doctor_ml_per_slot": row[3],
        }
    return {"alarm_enabled": True, "mode": "smart", "doctor_ml_per_slot": 0}

def log_water_intake(slot_time, base_ml, final_ml, temp_c, humidity_pct, supplement_ml):
    conn   = connect()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS water_intake_log (
            log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_time     TEXT,
            actual_time   TEXT,
            base_ml       INTEGER,
            final_ml      INTEGER,
            supplement_ml INTEGER,
            temp_c        REAL,
            humidity_pct  REAL
        )
    """)
    cursor.execute("""
        INSERT INTO water_intake_log
        (slot_time, actual_time, base_ml, final_ml, supplement_ml, temp_c, humidity_pct)
        VALUES (?,?,?,?,?,?,?)
    """, (
        slot_time,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        base_ml, final_ml, supplement_ml, temp_c, humidity_pct
    ))
    conn.commit()
    conn.close()


def calculate_daily_water_ml(age, gender, weight_kg, activity_level):
    
    if age <= 5:
        base = weight_kg * 100
    elif age >= 65:
        base = weight_kg * 30
    else:
        base = weight_kg * 35

    multipliers = {"light": 1.00, "moderate": 1.15, "high": 1.30}
    adjusted    = base * multipliers.get(activity_level.lower(), 1.00)

    if gender.lower() == "male":
        adjusted += 200

    return int(round(adjusted / 50) * 50)

def adjust_ml_for_climate(base_slot_ml, temp_c, humidity_pct, age, weight_kg):
    
    T_NEUTRAL  = 22.0
    RH_NEUTRAL = 50.0

    # Temperature supplement
    if temp_c <= T_NEUTRAL:
        temp_daily_extra = 0.0
    elif temp_c <= 37.0:
        temp_daily_extra = (temp_c - T_NEUTRAL) * 11.4
    else:
        temp_daily_extra = (37.0 - T_NEUTRAL) * 11.4 + (temp_c - 37.0) * 2.5 * weight_kg

    temp_slot_extra = temp_daily_extra / 12.0   # 1 slot = 2hr = 1/12 day

    # Humidity factor
    if humidity_pct <= RH_NEUTRAL:
        humidity_factor = 1.0 + min((RH_NEUTRAL - humidity_pct) * 0.008, 0.30)
    else:
        humidity_factor = 1.0 + min((humidity_pct - RH_NEUTRAL) * 0.003, 0.15)

    raw_supplement = temp_slot_extra * humidity_factor

    # Age-specific safety cap
    if age <= 5:
        max_supplement = min(base_slot_ml * 0.30, 60)
    elif age >= 65:
        max_supplement = min(base_slot_ml * 0.35, 120)
    else:
        max_supplement = min(base_slot_ml * 0.40, 150)

    supplement_ml = round(max(0.0, min(raw_supplement, max_supplement)) / 10) * 10
    final_ml      = round((base_slot_ml + supplement_ml) / 10) * 10

    if temp_c <= T_NEUTRAL and humidity_pct <= 60:
        condition = "Comfortable"
    elif temp_c <= 27 and humidity_pct <= 70:
        condition = "Slightly warm"
    elif temp_c <= 32 or (humidity_pct > 70 and temp_c > 27):
        condition = "Warm/humid"
    elif temp_c <= 37:
        condition = "Hot"
    else:
        condition = "Very hot"

    return {
        "final_ml":      int(final_ml),
        "supplement_ml": int(supplement_ml),
        "condition":     condition,
    }

WATER_SLOTS = [
    {"time": "06:00", "label": "Wake-up"},
    {"time": "08:00", "label": "After breakfast"},
    {"time": "10:00", "label": "Mid-morning"},
    {"time": "12:00", "label": "Before lunch"},
    {"time": "14:00", "label": "After lunch"},
    {"time": "16:00", "label": "Peak afternoon"},
    {"time": "18:00", "label": "Early evening"},
]

WATER_WEIGHTS = {
    "06:00": 0.15, "08:00": 0.10, "10:00": 0.10,
    "12:00": 0.15, "14:00": 0.15, "16:00": 0.20,
    "18:00": 0.15,
}

def get_base_slot_ml(daily_total, slot_time, config):
    if config["mode"] == "custom":
        return int(config["doctor_ml_per_slot"])
    weight = WATER_WEIGHTS.get(slot_time, 0.15)
    return max(int(round((daily_total * weight) / 10) * 10), 50)

def water_job(slot_time, slot_label):
    
    print(f"\n{'='*48}")
    print(f"  WATER SLOT: {slot_time} — {slot_label}")
    print(f"{'='*48}")

    profile = get_water_profile()
    config  = get_alert_config()
  
    # if not profile:
        
    #     oled_display("Water Reminder", "No profile set.", "Visit web app")
    #     play_beep(2)
    #     speak("Water profile not configured. Please set up on the web app.")
    #     time.sleep(20)
    #     return
    
    if not config["alarm_enabled"]:
        return

    _, age, gender, weight_kg, height_cm, activity_level = profile

    oled_display("Water time!", f"{slot_label}")

    temp_c, humidity_pct = read_dht22()

    daily_total  = calculate_daily_water_ml(age, gender, weight_kg, activity_level)
    base_slot_ml = get_base_slot_ml(daily_total, slot_time, config)
    result       = adjust_ml_for_climate(base_slot_ml, temp_c, humidity_pct, age, weight_kg)
    final_ml     = result["final_ml"]
    supplement   = result["supplement_ml"]
    condition    = result["condition"]

    print(f"  Base: {base_slot_ml}ml | +{supplement}ml supplement | Final: {final_ml}ml")
    print(f"  Condition: {condition} | T:{temp_c}°C H:{humidity_pct}%")

    
    oled_display(
        slot_label,
        f"Drink: {final_ml} ml",
        # f"T:{temp_c}C  H:{humidity_pct}%",
        # f"+{supplement}ml ({condition})" if supplement > 0 else condition
    )
    
    play_beep(1)
    time.sleep(0.5)

    if supplement > 0:
        speak(
            f"Water time. {slot_label}. "
            f"Room is {condition}. "
            f"Please drink {final_ml} millilitres. "
            f"This includes {supplement} millilitres extra for the current room temperature."
        )
    else:
        speak(
            f"Water time. {slot_label}. "
            f"Room is comfortable. "
            f"Please drink {final_ml} millilitres of water."
        )

    dispense_water_only(final_ml, slot_label)

    log_water_intake(slot_time, base_slot_ml, final_ml, temp_c, humidity_pct, supplement)
    
    oled_display("Water complete!", f"{final_ml} ml done.")
    time.sleep(20)
    oled_clear()

    print(f"  Water slot complete. OLED + DHT22 OFF.")

def medicine_job(schedule):
    
    schedule_id = schedule[0]
    section_id  = schedule[1]
    water_ml    = schedule[5]

    print(f"\n{'='*48}")
    print(f"  MEDICINE JOB — ID:{schedule_id}  Section:{section_id}")
    print(f"{'='*48}")

    oled_display("Medicine Time!", "Preparing...")

    play_beep(2)
    time.sleep(0.3)
    speak("Time to take your medicine")

    dispense_medicine_and_water(section_id,water_ml)

    log_intake(schedule_id)
    speak("Medicine complete. Well done.")

    oled_display("All done!", "Medicine taken.")
    time.sleep(10)
    oled_clear()
    
    

scheduler = BackgroundScheduler()

def load_medicine_jobs():
    schedules = get_all_schedules()
    for s in schedules:
        hour, minute = map(int, s[2].split(":"))
        scheduler.add_job(
            medicine_job, 'cron',
            hour=hour, minute=minute,
            args=[s],
            id=f"med_{s[0]}",
            replace_existing=True
        )
    print(f"[SCHEDULER] {len(schedules)} medicine job(s) loaded.")

def load_water_jobs():
    config = get_alert_config()
    if not config["alarm_enabled"]:
        print("[SCHEDULER] Water alarms OFF — no water jobs loaded.")
        return
    for slot in WATER_SLOTS:
        hour, minute = map(int, slot["time"].split(":"))
        scheduler.add_job(
            water_job, 'cron',
            hour=hour, minute=minute,
            args=[slot["time"], slot["label"]],
            id=f"water_{slot['time']}",
            replace_existing=True
        )
    print(f"[SCHEDULER] {len(WATER_SLOTS)} water slot(s) loaded.")

load_medicine_jobs()
load_water_jobs()
scheduler.start()

move_servo(DEF_POS)

time.sleep(1)
speak("Smart caretaker robot is ready.")

while True:
    time.sleep(30)