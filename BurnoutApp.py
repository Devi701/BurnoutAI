import streamlit as st
import pandas as pd
from datetime import datetime
import uuid, hashlib
from pathlib import Path
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("burnout_xgb_model.pkl")
scaler = joblib.load("burnout_scaler.pkl")

# Session tracking
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()
if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())

st.title("Burnout Predictor (Demo)")

# Consent checkbox
st.markdown(
    "Privacy & analytics: To show engagement metrics to investors we can record anonymous session data "
    "(no names, emails, or IP addresses)."
)
consent = st.checkbox("I consent to anonymous analytics for this demo (see privacy notes).")

# ✅ User Inputs
work_hours = st.number_input("Work hours per week", min_value=0, max_value=100, value=40)
sleep_hours = st.number_input("Sleep hours per night", min_value=0.0, max_value=24.0, value=7.5)
stress_level = st.slider("Stress level (1–10)", 1, 10, 5)
job_satisfaction = st.slider("Job satisfaction (1–10)", 1, 10, 5)
support_level = st.slider("Support level from colleagues (1–10)", 1, 10, 5)
exercise_days = st.number_input("Exercise days per week", min_value=0, max_value=7, value=3)
exercise_minutes = st.number_input("Average exercise minutes per day", min_value=0, max_value=300, value=30)
remote_work = st.radio("Remote work (0 = No, 1 = Yes)", options=[0, 1], index=0)
caffeine_mg = st.number_input("Daily caffeine intake (mg)", min_value=0, max_value=1000, value=200)
screen_time_hours = st.number_input("Daily screen time (hours)", min_value=0.0, max_value=24.0, value=6.0)

# Predict button
if st.button("Predict Burnout Score"):
    new_data = pd.DataFrame([{
        "work_hours": work_hours,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "job_satisfaction": job_satisfaction,
        "support_level": support_level,
        "exercise_days": exercise_days,
        "exercise_minutes": exercise_minutes,
        "remote_work": remote_work,
        "caffeine_mg": caffeine_mg,
        "screen_time_hours": screen_time_hours
    }])

    # Scale and predict
    new_scaled = scaler.transform(new_data)
    pred = model.predict(new_scaled)[0]
    st.write(f"Predicted Burnout Score: {pred:.2f}")

    # Log engagement if consent is given
    if consent:
        sid_hash = hashlib.sha256(st.session_state.session_uuid.encode()).hexdigest()
        duration = (datetime.now() - st.session_state.start_time).total_seconds()
        row = {
            "timestamp": datetime.now().isoformat(),
            "session_hash": sid_hash,
            "duration_s": duration,
            "work_hours": work_hours,
            "sleep_hours": sleep_hours,
            "stress_level": stress_level,
            "job_satisfaction": job_satisfaction,
            "support_level": support_level,
            "exercise_days": exercise_days,
            "exercise_minutes": exercise_minutes,
            "remote_work": remote_work,
            "caffeine_mg": caffeine_mg,
            "screen_time_hours": screen_time_hours,
            "predicted_score": float(pred)
        }
        df = pd.DataFrame([row])
        df.to_csv(
            "engagement_data.csv",
            mode="a",
            header=not Path("engagement_data.csv").exists(),
            index=False
        )
        st.success("Anonymous engagement data recorded for the demo.")
    else:
        st.info("Analytics not recorded (consent not given).")

# Optional: delete own session data
if st.button("Delete my demo data"):
    if Path("engagement_data.csv").exists():
        sid_hash = hashlib.sha256(st.session_state.session_uuid.encode()).hexdigest()
        df_all = pd.read_csv("engagement_data.csv")
        df_all = df_all[df_all["session_hash"] != sid_hash]
        df_all.to_csv("engagement_data.csv", index=False)
        st.success("Your demo data has been deleted.")
    else:
        st.info("No data file found.")


