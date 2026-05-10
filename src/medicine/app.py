from flask import Flask, render_template, request, redirect
import database

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


# COMPARTMENT ROUTES

@app.route("/add_compartment")
def add_compartment():
    return render_template("add_compartment.html")


@app.route("/insert_compartment", methods=["POST"])
def insert_compartment():
    section_id       = request.form["section_id"]
    medicine_name    = request.form["medicine_name"]
    medicine_type    = request.form["medicine_type"]
    servo_angle      = request.form["servo_angle"]
    expiry_date      = request.form["expiry_date"]
    current_quantity = request.form["current_quantity"]
    database.add_compartment(section_id, medicine_name, medicine_type,
                             servo_angle, expiry_date, current_quantity)
    return redirect("/view_compartment")


@app.route("/view_compartment")
def view_compartment():
    data = database.get_all_compartments()
    return render_template("view_compartment.html", data=data)


@app.route("/delete_compartment/<section_id>")
def delete_compartment(section_id):
    database.delete_compartment(section_id)
    return redirect("/view_compartment")


@app.route("/edit_compartment/<section_id>")
def edit_compartment(section_id):
    data = database.get_compartment_by_id(section_id)
    return render_template("edit_compartment.html", data=data)


@app.route("/update_compartment", methods=["POST"])
def update_compartment():
    section_id       = request.form["section_id"]
    medicine_name    = request.form["medicine_name"]
    medicine_type    = request.form["medicine_type"]
    servo_angle      = request.form["servo_angle"]
    expiry_date      = request.form["expiry_date"]
    current_quantity = request.form["current_quantity"]
    database.update_compartment(section_id, medicine_name, medicine_type,
                                servo_angle, expiry_date, current_quantity)
    return redirect("/view_compartment")


# SCHEDULE ROUTES

@app.route("/add_schedule")
def add_schedule():
    return render_template("add_schedule.html")


@app.route("/insert_schedule", methods=["POST"])
def insert_schedule():
    section_id        = request.form["section_id"]
    time              = request.form["time"]
    dosage            = request.form["dosage"]
    before_after_food = request.form["before_after_food"]
    water_ml          = request.form["water_ml"]
    database.add_schedule(section_id, time, dosage, before_after_food, water_ml)
    return redirect("/view_schedule")


@app.route("/view_schedule")
def view_schedule():
    conn = database.connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM schedule")
    data = cursor.fetchall()
    conn.close()
    return render_template("view_schedule.html", data=data)


@app.route("/delete_schedule/<int:schedule_id>")
def delete_schedule(schedule_id):
    database.delete_schedule(schedule_id)
    return redirect("/view_schedule")


@app.route("/edit_schedule/<int:schedule_id>")
def edit_schedule(schedule_id):
    data = database.get_schedule_by_id(schedule_id)
    return render_template("edit_schedule.html", data=data)


@app.route("/update_schedule", methods=["POST"])
def update_schedule():
    schedule_id       = request.form["schedule_id"]
    section_id        = request.form["section_id"]
    time              = request.form["time"]
    dosage            = request.form["dosage"]
    before_after_food = request.form["before_after_food"]
    water_ml          = request.form["water_ml"]
    database.update_schedule(schedule_id, section_id, time, dosage,
                             before_after_food, water_ml)
    return redirect("/view_schedule")


# WATER PROFILE ROUTES

@app.route("/water_profile")
def water_profile():
    profile        = database.get_water_profile()
    recommended_ml = None
    alert_slots    = []
    config         = database.get_alert_config()

    if profile:
        _, age, gender, weight_kg, height_cm, activity_level = profile
        recommended_ml = database.calculate_daily_water_ml(
            age, gender, weight_kg, height_cm, activity_level)
        alert_slots = database.build_alert_schedule(recommended_ml, config)

    return render_template("water_profile.html",
                           profile=profile,
                           recommended_ml=recommended_ml,
                           alert_slots=alert_slots,
                           config=config)


@app.route("/edit_water_profile")
def edit_water_profile():
    profile = database.get_water_profile()
    return render_template("edit_water_profile.html", profile=profile)


@app.route("/save_water_profile", methods=["POST"])
def save_water_profile():
    age            = int(request.form["age"])
    gender         = request.form["gender"]
    weight_kg      = float(request.form["weight_kg"])
    height_cm      = float(request.form["height_cm"])
    activity_level = request.form["activity_level"]
    database.save_water_profile(age, gender, weight_kg, height_cm, activity_level)
    return redirect("/water_profile")


@app.route("/edit_water_alerts")
def edit_water_alerts():
    config = database.get_alert_config()
    return render_template("edit_water_alerts.html", config=config)


@app.route("/save_water_alerts", methods=["POST"])
def save_water_alerts():
    alarm_enabled      = 1 if request.form.get("alarm_enabled") else 0
    mode               = request.form["mode"]
    doctor_ml_per_slot = int(request.form.get("doctor_ml_per_slot") or 0)
    database.save_alert_config(alarm_enabled, mode, doctor_ml_per_slot)
    return redirect("/water_profile")


if __name__ == "__main__":
    database.create_tables()
    app.run(debug=True)