import streamlit as st
import pandas as pd
import numpy as np
import uuid
import joblib
from datetime import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import random

# ------------------------
# Helper functions
# ------------------------

# Load or initialize user history
USER_HISTORY_FILE = "user_history.csv"
def load_user_data():
    if pd.io.common.file_exists(USER_HISTORY_FILE):
        return pd.read_csv(USER_HISTORY_FILE)
    else:
        cols = ["user_id","date","work_hours","sleep_hours","stress_level",
                "job_satisfaction","support_level","exercise_days","exercise_minutes",
                "remote_work","caffeine_mg","screen_time_hours","predicted_score"]
        return pd.DataFrame(columns=cols)

def save_user_data(user_id, inputs, pred):
    df = load_user_data()
    row = {**inputs, "user_id": user_id, "date": datetime.now().strftime("%Y-%m-%d"), "predicted_score": float(pred)}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(USER_HISTORY_FILE, index=False)

def get_trends(user_id, df):
    user_data = df[df["user_id"]==user_id].sort_values("date")
    trends = {}
    if len(user_data) >= 2:
        trends["delta_work"] = (user_data["work_hours"].iloc[-1] - user_data["work_hours"].iloc[-2]) or 0
        trends["delta_sleep"] = (user_data["sleep_hours"].iloc[-1] - user_data["sleep_hours"].iloc[-2]) or 0
        trends["delta_stress"] = (user_data["stress_level"].iloc[-1] - user_data["stress_level"].iloc[-2]) or 0
    return trends

# Adaptive feedback logic
def generate_feedback(latest, trends):
    messages = []

    work = latest.get("work_hours") or 0
    sleep = latest.get("sleep_hours") or 0
    stress = latest.get("stress_level") or 0
    job_sat = latest.get("job_satisfaction") or 5
    delta_work = trends.get("delta_work") or 0
    delta_sleep = trends.get("delta_sleep") or 0
    delta_stress = trends.get("delta_stress") or 0

    # Workload
    if work > 45 or delta_work > 2:
        messages.append("⚠️ High workload detected or increasing — consider taking a break.")
    elif work > 35:
        messages.append("📌 Workload is moderate. Take short breaks.")
    else:
        messages.append("✅ Workload is manageable.")

    # Sleep
    if sleep < 6 or delta_sleep < -1:
        messages.append("😴 Sleep is low or decreasing — prioritize rest.")
    else:
        messages.append("✅ Sleep is adequate.")

    # Stress
    if stress > 7 or delta_stress > 1:
        messages.append("🔥 Stress levels are high — try mindfulness or a short break.")
    else:
        messages.append("✅ Stress is under control.")

    # Job satisfaction
    if job_sat < 4:
        messages.append("💡 Low job satisfaction — consider discussing workload or tasks with your manager.")
    else:
        messages.append("👍 Job satisfaction looks good.")

    return messages

# Risk category coloring
def risk_category(score):
    if score < 3.5:
        return "Low", "green"
    elif score < 6.5:
        return "Medium", "orange"
    else:
        return "High", "red"

# Generate PDF report
def generate_pdf(user_id, pred, messages):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 800, f"Burnout Report - User: {user_id}")
    c.drawString(100, 780, f"Predicted Score: {pred:.2f} ({risk_category(pred)[0]})")
    y = 760
    for msg in messages:
        c.drawString(100, y, msg)
        y -= 20
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------
# Load model
# ------------------------
model = joblib.load("burnout_xgb_model.pkl")
scaler = joblib.load("burnout_scaler.pkl")

# ------------------------
# Sidebar: User login / guest
# ------------------------
st.sidebar.title("User / Device")
username = st.sidebar.text_input("Username (leave blank for guest device)")

if not username:
    if "guest_uuid" not in st.session_state:
        st.session_state.guest_uuid = str(uuid.uuid4())
    user_id = st.session_state.guest_uuid
    st.sidebar.write(f"Using device UUID (guest): {user_id}")
else:
    user_id = username

# Optional LLM API key
st.sidebar.subheader("LLM Recommendations (optional)")
api_key = st.sidebar.text_input("OpenAI API key", type="password")

# ------------------------
# Main Inputs
# ------------------------
st.title("🔥 Burnout Predictor Dashboard")

work_hours = st.number_input("Work hours per week", 0, 100, 40)
sleep_hours = st.number_input("Sleep hours per night", 0.0, 24.0, 7.5)
stress_level = st.slider("Stress level (1–10)", 1, 10, 5)
job_satisfaction = st.slider("Job satisfaction (1–10)", 1, 10, 5)
support_level = st.slider("Support from colleagues (1–10)", 1, 10, 5)
exercise_days = st.number_input("Exercise days per week", 0, 7, 3)
exercise_minutes = st.number_input("Average exercise minutes per day", 0, 300, 30)
remote_work = st.radio("Remote work", [0,1], index=0)
caffeine_mg = st.number_input("Daily caffeine intake (mg)", 0, 1000, 200)
screen_time_hours = st.number_input("Daily screen time (hours)", 0.0, 24.0, 6.0)

# ------------------------
# Predict button
# ------------------------
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
    scaled = scaler.transform(new_data)
    pred = float(model.predict(scaled)[0])
    st.subheader(f"Predicted Burnout Score: {pred:.2f}")
    category, color = risk_category(pred)
    st.markdown(f"**Risk Category:** <span style='color:{color}'>{category}</span>", unsafe_allow_html=True)

    # Save history
    inputs = new_data.iloc[0].to_dict()
    save_user_data(user_id, inputs, pred)

    # Load history & trends
    df = load_user_data()
    trends = get_trends(user_id, df)
    latest_row = df[df["user_id"]==user_id].sort_values("date").iloc[-1].to_dict()

    # Adaptive feedback
    messages = generate_feedback(latest_row, trends)
    st.subheader("📋 Adaptive Feedback")
    for msg in messages:
        st.write("- " + msg)

    # ------------------------
    # Burnout score history
    # ------------------------
    st.subheader("📊 Burnout History")
    user_history = df[df["user_id"]==user_id]
    fig, ax = plt.subplots()
    dates = pd.to_datetime(user_history['date'])
    scores = user_history['predicted_score']
    ax.plot(dates, scores, marker='o', color='red')
    for d,s in zip(dates, scores):
        ax.text(d,s,f"{s:.1f}", ha='center', va='bottom')
    ax.set_ylabel("Burnout Score")
    ax.set_xlabel("Date")
    ax.set_title("Burnout Score Over Time")
    st.pyplot(fig)

    # ------------------------
    # Your habits vs peers
    # ------------------------
    st.subheader("📊 Your Habits vs Peers (Average)")

    # Mock peer averages
    peer_data = {
        "Sleep Hours": random.uniform(6.5, 8),
        "Work Hours": random.uniform(35, 45),
        "Exercise Minutes": random.uniform(20, 60),
        "Caffeine (mg)": random.uniform(150, 300),
        "Screen Time Hours": random.uniform(5, 8),
    }

    # User's data
    user_data = {
        "Sleep Hours": sleep_hours,
        "Work Hours": work_hours,
        "Exercise Minutes": exercise_minutes,
        "Caffeine (mg)": caffeine_mg,
        "Screen Time Hours": screen_time_hours,
    }

    compare_df = pd.DataFrame([user_data, peer_data], index=["You", "Peers"]).T

    # Plot bar chart
    fig2, ax2 = plt.subplots(figsize=(8,5))
    compare_df.plot(kind='bar', ax=ax2)
    ax2.set_ylabel("Amount / Hours")
    ax2.set_title("Your Habits vs Peers (Average)")
    ax2.legend(loc='upper right')
    st.pyplot(fig2)

    # ------------------------
    # PDF Export
    # ------------------------
    pdf_buffer = generate_pdf(user_id, pred, messages)
    st.download_button("Download PDF Report", pdf_buffer, file_name="burnout_report.pdf", mime="application/pdf")
