"""
Tools the healthcare agent can call autonomously.

  - search_medical_knowledge   : RAG lookup in MedQuAD
  - check_availability         : free appointment slots for a specialty
  - book_appointment / list_appointments / cancel_appointment
  - summarize_health_report    : parse lab values and flag abnormal results
  - get_preventive_recommendations
"""
import json
import os
import re
import uuid
from datetime import date, datetime, timedelta

from .knowledge_base import ROOT

APPOINTMENTS_FILE = os.path.join(ROOT, "data", "appointments.json")

# ----------------------------------------------------------------------------
# Doctors / appointment scheduling
# ----------------------------------------------------------------------------
DOCTORS = {
    "General Medicine": "Dr. Anjali Mehta",
    "Cardiology": "Dr. Rohan Kapoor",
    "Dermatology": "Dr. Sneha Iyer",
    "Endocrinology": "Dr. Vikram Rao",
    "Pediatrics": "Dr. Priya Nair",
    "Orthopedics": "Dr. Arjun Singh",
    "Neurology": "Dr. Kavita Joshi",
    "Gynecology": "Dr. Meera Desai",
    "ENT": "Dr. Sameer Khan",
    "Psychiatry": "Dr. Neha Gupta",
}

SPECIALTY_KEYWORDS = {
    "Cardiology": ["heart", "cardio", "chest", "blood pressure", "bp", "hypertension", "cholesterol"],
    "Dermatology": ["skin", "derma", "rash", "acne", "eczema", "hair"],
    "Endocrinology": ["diabetes", "sugar", "thyroid", "endocrin", "hormone", "glucose"],
    "Pediatrics": ["child", "kid", "baby", "pediatric", "infant"],
    "Orthopedics": ["bone", "joint", "knee", "back pain", "fracture", "ortho", "spine"],
    "Neurology": ["neuro", "headache", "migraine", "seizure", "brain", "nerve"],
    "Gynecology": ["gyne", "pregnan", "period", "menstrua", "women"],
    "ENT": ["ear", "nose", "throat", "ent", "sinus", "tonsil"],
    "Psychiatry": ["mental", "anxiety", "depress", "stress", "psychiat", "sleep"],
}

SLOTS = ["09:00", "10:00", "11:00", "12:00", "14:00", "15:00", "16:00", "17:00"]


def _load_appointments():
    if not os.path.exists(APPOINTMENTS_FILE):
        return []
    with open(APPOINTMENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_appointments(items):
    os.makedirs(os.path.dirname(APPOINTMENTS_FILE), exist_ok=True)
    with open(APPOINTMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def resolve_specialty(text: str) -> str:
    t = (text or "").lower()
    for spec in DOCTORS:
        if re.search(r"\b" + re.escape(spec.lower()) + r"\b", t):
            return spec
    for spec, kws in SPECIALTY_KEYWORDS.items():
        # short keywords (ear, ent, bp...) must match a whole word
        if any(re.search(r"\b" + re.escape(k) + (r"\b" if len(k) <= 3 else ""), t) for k in kws):
            return spec
    return "General Medicine"


def parse_date(text: str):
    """Understands 'today', 'tomorrow', weekday names and YYYY-MM-DD / DD-MM-YYYY."""
    t = (text or "").lower()
    today = date.today()
    if "day after tomorrow" in t:
        return today + timedelta(days=2)
    if "tomorrow" in t:
        return today + timedelta(days=1)
    if "today" in t:
        return today
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", t)
    if m:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, d in enumerate(days):
        if d in t:
            ahead = (i - today.weekday()) % 7 or 7
            return today + timedelta(days=ahead)
    return None


def parse_time(text: str):
    t = (text or "").lower()
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        h, mins = int(m.group(1)), int(m.group(2) or 0)
        if m.group(3) == "pm" and h != 12:
            h += 12
        if m.group(3) == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mins:02d}"
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    if "morning" in t:
        return "10:00"
    if "afternoon" in t:
        return "14:00"
    if "evening" in t:
        return "17:00"
    return None


def check_availability(specialty: str, day: str = "") -> dict:
    specialty = resolve_specialty(specialty)
    d = parse_date(day) if day else None
    days = [d] if d else [date.today() + timedelta(days=i) for i in range(1, 4)]
    booked = {(a["date"], a["time"], a["specialty"]) for a in _load_appointments()
              if a["status"] == "confirmed"}
    free = {}
    for dd in days:
        if dd.weekday() == 6:  # closed on Sundays
            continue
        ds = dd.isoformat()
        free[ds] = [s for s in SLOTS if (ds, s, specialty) not in booked]
    return {"specialty": specialty, "doctor": DOCTORS[specialty], "available_slots": free}


def book_appointment(patient_name: str, specialty: str, day: str = "", time: str = "",
                     reason: str = "") -> dict:
    specialty = resolve_specialty(specialty)
    d = parse_date(day) or (date.today() + timedelta(days=1))
    if d.weekday() == 6:
        d += timedelta(days=1)
    t = parse_time(time) or time or ""
    avail = check_availability(specialty, d.isoformat())["available_slots"].get(d.isoformat(), [])
    if not avail:
        return {"status": "failed", "message": f"No free slots for {specialty} on {d}."}
    if t not in avail:
        # pick the closest free slot to the requested time
        wanted = t or "10:00"
        t = min(avail, key=lambda s: abs(int(s[:2]) * 60 + int(s[3:]) -
                                         (int(wanted[:2]) * 60 + int(wanted[3:5]))))
    appt = {
        "id": uuid.uuid4().hex[:6].upper(),
        "patient": patient_name or "Patient",
        "specialty": specialty,
        "doctor": DOCTORS[specialty],
        "date": d.isoformat(),
        "time": t,
        "reason": reason,
        "status": "confirmed",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    items = _load_appointments()
    items.append(appt)
    _save_appointments(items)
    return {"status": "confirmed", "appointment": appt}


def list_appointments(patient_name: str = "") -> dict:
    items = [a for a in _load_appointments() if a["status"] == "confirmed"]
    if patient_name:
        items = [a for a in items if a["patient"].lower() == patient_name.lower()]
    return {"appointments": items}


def cancel_appointment(appointment_id: str) -> dict:
    items = _load_appointments()
    for a in items:
        if a["id"].upper() == appointment_id.upper().strip() and a["status"] == "confirmed":
            a["status"] = "cancelled"
            _save_appointments(items)
            return {"status": "cancelled", "appointment": a}
    return {"status": "failed", "message": f"No active appointment with id {appointment_id}."}


# ----------------------------------------------------------------------------
# Health report summarisation
# ----------------------------------------------------------------------------
# name -> (aliases, unit, low, high, advice if low, advice if high)
LAB_RANGES = {
    "HbA1c": (["hba1c", "a1c", "glycated"], "%", 4.0, 5.6,
              "Low.", "5.7-6.4% suggests prediabetes, >=6.5% suggests diabetes."),
    "Hemoglobin": (["hemoglobin", "haemoglobin", "hb", "hgb"], "g/dL", 12.0, 17.5,
                   "May indicate anaemia - iron-rich diet and a doctor review are advised.",
                   "Can be due to dehydration or smoking; discuss with your doctor."),
    "WBC Count": (["wbc", "white blood cell", "total leucocyte count", "tlc"], "x10^3/uL", 4.0, 11.0,
                  "Low immunity cells - avoid infections and consult a physician.",
                  "Often a sign of infection or inflammation."),
    "Platelet Count": (["platelet"], "x10^3/uL", 150, 450,
                       "Low platelets increase bleeding risk - consult a doctor.",
                       "High platelets should be reviewed by a doctor."),
    "Fasting Blood Sugar": (["fasting blood sugar", "fasting glucose", "fbs", "fasting sugar",
                             "glucose fasting"], "mg/dL", 70, 99,
                            "Low sugar - eat regular meals.",
                            "Above normal - may indicate prediabetes/diabetes; limit sugar, exercise."),
    "Postprandial Blood Sugar": (["postprandial", "ppbs", "pp sugar"], "mg/dL", 70, 140,
                                 "Low sugar.", "Raised after-meal sugar - diabetes screening advised."),
    "LDL Cholesterol": (["ldl"], "mg/dL", 0, 100, "", "High 'bad' cholesterol raises heart risk."),
    "HDL Cholesterol": (["hdl"], "mg/dL", 40, 100, "Low 'good' cholesterol - exercise helps raise it.", ""),
    "Total Cholesterol": (["total cholesterol", "cholesterol total", "cholesterol"], "mg/dL", 0, 200,
                          "", "High cholesterol - reduce fried/fatty food, exercise regularly."),
    "Triglycerides": (["triglyceride", "tg"], "mg/dL", 0, 150, "", "High - cut sugar, refined carbs and alcohol."),
    "Creatinine": (["creatinine"], "mg/dL", 0.6, 1.3, "Usually not concerning.",
                   "May indicate reduced kidney function - consult a doctor."),
    "Urea": (["urea", "bun"], "mg/dL", 7, 20, "", "Raised urea - stay hydrated, check kidney function."),
    "TSH": (["tsh", "thyroid stimulating"], "mIU/L", 0.4, 4.0,
            "Low TSH may suggest an overactive thyroid.", "High TSH may suggest an underactive thyroid."),
    "Vitamin D": (["vitamin d", "vit d", "25-oh"], "ng/mL", 30, 100,
                  "Vitamin D deficiency - sunlight exposure and supplements may help.", "Too high - review supplements."),
    "Vitamin B12": (["vitamin b12", "b12"], "pg/mL", 200, 900, "Low B12 - consider dietary sources/supplements.", ""),
    "Systolic BP": (["systolic"], "mmHg", 90, 120, "Low blood pressure.", "Elevated blood pressure - reduce salt, exercise."),
    "Diastolic BP": (["diastolic"], "mmHg", 60, 80, "Low blood pressure.", "Elevated blood pressure."),
    "Heart Rate": (["heart rate", "pulse"], "bpm", 60, 100, "Slow heart rate.", "Fast heart rate."),
    "BMI": (["bmi", "body mass index"], "kg/m2", 18.5, 24.9, "Underweight.", "Overweight - balanced diet and activity advised."),
}


def _find_value(line: str):
    m = re.search(r"[:=\-\s]\s*(\d+(?:\.\d+)?)", line)
    return float(m.group(1)) if m else None


def summarize_health_report(report_text: str) -> dict:
    results, seen = [], set()
    lines = [l.strip() for l in (report_text or "").splitlines() if l.strip()]
    for line in lines:
        low = line.lower()
        # blood pressure written as 140/90
        bp = re.search(r"(blood pressure|bp)\D*(\d{2,3})\s*/\s*(\d{2,3})", low)
        if bp:
            for name, val in (("Systolic BP", bp.group(2)), ("Diastolic BP", bp.group(3))):
                seen.add(name)
                results.append(_classify(name, float(val)))
            continue
        for name, spec in LAB_RANGES.items():
            if name in seen:
                continue
            if any(re.search(r"\b" + re.escape(a) + r"\b", low) for a in spec[0]):
                val = _find_value(low.split(":", 1)[-1] if ":" in low else low)
                if val is not None:
                    seen.add(name)
                    results.append(_classify(name, val))
                break
    abnormal = [r for r in results if r["status"] != "Normal"]
    if not results:
        overview = "No recognisable lab values were found in the report."
    elif not abnormal:
        overview = f"All {len(results)} detected values are within the normal range. Keep it up!"
    else:
        names = ", ".join(f"{r['test']} ({r['status'].lower()})" for r in abnormal)
        overview = (f"{len(results)} values checked: {len(results) - len(abnormal)} normal, "
                    f"{len(abnormal)} need attention - {names}.")
    return {
        "overview": overview,
        "results": results,
        "abnormal_count": len(abnormal),
        "recommendations": [r["advice"] for r in abnormal if r["advice"]],
        "suggested_specialty": _specialty_for(abnormal),
    }


def _classify(name, value):
    _, unit, lo, hi, adv_lo, adv_hi = LAB_RANGES[name]
    if value < lo:
        status, advice = "Low", adv_lo
    elif value > hi:
        status, advice = "High", adv_hi
    else:
        status, advice = "Normal", ""
    return {"test": name, "value": value, "unit": unit, "normal_range": f"{lo}-{hi}",
            "status": status, "advice": advice}


def _specialty_for(abnormal):
    tests = {r["test"] for r in abnormal}
    if tests & {"Fasting Blood Sugar", "Postprandial Blood Sugar", "HbA1c", "TSH"}:
        return "Endocrinology"
    if tests & {"Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol", "Triglycerides",
                "Systolic BP", "Diastolic BP", "Heart Rate"}:
        return "Cardiology"
    return "General Medicine" if abnormal else None


# ----------------------------------------------------------------------------
# Preventive care recommendations
# ----------------------------------------------------------------------------
PREVENTIVE = {
    "general": [
        "Eat a balanced diet rich in vegetables, fruits, whole grains and lean protein.",
        "Do at least 150 minutes of moderate exercise (e.g. brisk walking) per week.",
        "Sleep 7-9 hours a night and keep a regular sleep schedule.",
        "Drink enough water; limit sugary drinks, alcohol and avoid tobacco.",
        "Get an annual health check-up and keep vaccinations up to date.",
    ],
    "diabetes": [
        "Limit refined sugar and white carbs; prefer high-fibre whole foods.",
        "Maintain a healthy weight - losing 5-7% body weight cuts diabetes risk significantly.",
        "Exercise 30 minutes a day, 5 days a week.",
        "Check fasting blood sugar / HbA1c yearly if you are over 35 or have a family history.",
    ],
    "heart": [
        "Reduce salt (<5 g/day), saturated and trans fats.",
        "Stay active and keep blood pressure below 120/80 mmHg.",
        "Check cholesterol every 4-6 years (yearly if at risk).",
        "Quit smoking - risk of heart disease halves within a year.",
    ],
    "hypertension": [
        "Follow the DASH diet: fruits, vegetables, low-fat dairy, less salt.",
        "Limit alcohol and caffeine; manage stress with breathing or yoga.",
        "Monitor blood pressure at home regularly.",
    ],
    "obesity": [
        "Aim for a gradual 0.5-1 kg weight loss per week.",
        "Use smaller portions, avoid late-night snacking and processed food.",
        "Combine cardio with strength training 2-3 times per week.",
    ],
    "cancer": [
        "Avoid tobacco and limit alcohol.",
        "Use sunscreen and avoid excessive sun exposure.",
        "Follow age-appropriate screenings (cervical, breast, colorectal).",
        "Get HPV and Hepatitis B vaccinations.",
    ],
    "infection": [
        "Wash hands frequently with soap for 20 seconds.",
        "Get the yearly flu vaccine and stay up to date with boosters.",
        "Cover coughs/sneezes and stay home when sick.",
    ],
    "mental health": [
        "Stay socially connected and talk about your feelings.",
        "Practise mindfulness or meditation 10 minutes a day.",
        "Limit screen time before bed and seek professional help early.",
    ],
}

TOPIC_KEYWORDS = {
    "diabetes": ["diabet", "sugar", "glucose", "insulin"],
    "heart": ["heart", "cardio", "cholesterol", "stroke"],
    "hypertension": ["hypertension", "blood pressure", "bp"],
    "obesity": ["obes", "weight", "fat", "bmi"],
    "cancer": ["cancer", "tumor", "tumour"],
    "infection": ["flu", "cold", "infect", "covid", "virus", "fever"],
    "mental health": ["mental", "stress", "anxiety", "depress", "sleep"],
}


def get_preventive_recommendations(topic: str = "general", age: int = 0) -> dict:
    t = (topic or "general").lower()
    topics = [k for k, kws in TOPIC_KEYWORDS.items() if any(w in t for w in kws)] or ["general"]
    tips = []
    for k in topics:
        tips.extend(PREVENTIVE[k])
    screenings = []
    try:
        age = int(age or 0)
    except (TypeError, ValueError):
        age = 0
    if age >= 18:
        screenings.append("Blood pressure check every year.")
    if age >= 35:
        screenings.append("Blood sugar and cholesterol screening every 1-3 years.")
    if age >= 45:
        screenings.append("Colorectal cancer screening and an ECG as advised by your doctor.")
    if age >= 60:
        screenings.append("Bone density, eye and hearing tests; pneumonia & shingles vaccines.")
    return {"topics": topics, "tips": tips, "age_based_screenings": screenings}


# ----------------------------------------------------------------------------
# Safety
# ----------------------------------------------------------------------------
EMERGENCY_TERMS = ["chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
                   "unconscious", "heart attack", "stroke symptoms", "severe bleeding",
                   "suicide", "kill myself", "overdose", "seizure now"]


def detect_emergency(text: str) -> bool:
    t = (text or "").lower()
    return any(term in t for term in EMERGENCY_TERMS)


EMERGENCY_MESSAGE = ("**This may be a medical emergency.** Please call your local emergency number "
                     "(e.g. 112 / 108 / 911) or go to the nearest hospital immediately. "
                     "If you are having thoughts of self-harm, please contact a crisis helpline right away.")
