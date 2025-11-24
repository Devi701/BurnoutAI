import streamlit as st
import pandas as pd
import numpy as np
import uuid
import joblib
from datetime import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt

# ========================
# FILES
# ========================
USER_HISTORY_FILE = "user_history.csv"
ALL_SCORES_FILE = "all_burnout_scores.csv"

# ========================
# SAFE LOAD FUNCTIONS (no more EmptyDataError!)
# ========================
def load_all_scores():
    if not pd.io.common.file_exists(ALL_SCORES_FILE):
        return pd.DataFrame(columns=["score", "timestamp"])
    try:
        df = pd.read_csv(ALL_SCORES_FILE)
        if df.empty or "score" not in df.columns:
            return pd.DataFrame(columns=["score", "timestamp"])
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["score", "timestamp"])

def load_user_data():
    cols = ["user_id","date","work_hours","sleep_hours","stress_level",
            "job_satisfaction","support_level","exercise_days","exercise_minutes",
            "remote_work","caffeine_mg","screen_time_hours","predicted_score"]
    if not pd.io.common.file_exists(USER_HISTORY_FILE):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(USER_HISTORY_FILE)
        if df.empty:
            return pd.DataFrame(columns=cols)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)

# ========================
# SAVE FUNCTIONS
# ========================
def save_score(score):
    df = load_all_scores()
    new_row = pd.DataFrame([{"score": float(score), "timestamp": datetime.now()}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ALL_SCORES_FILE, index=False)

def save_user_data(user_id, inputs, pred):
    df = load_user_data()
    row = {**inputs, "user_id": user_id, "date": datetime.now().strftime("%Y-%m-%d"), "predicted_score": float(pred)}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(USER_HISTORY_FILE, index=False)

# ========================
# Your original helpers (unchanged)
# ========================
def get_trends(user_id, df):
    user_data = df[df["user_id"] == user_id].sort_values("date")
    trends = {}
    if len(user_data) >= 2:
        trends["delta_work"] = (user_data["work_hours"].iloc[-1] - user_data["work_hours"].iloc[-2]) or 0
        trends["delta_sleep"] = (user_data["sleep_hours"].iloc[-1] - user_data["sleep_hours"].iloc[-2]) or 0
        trends["delta_stress"] = (user_data["stress_level"].iloc[-1] - user_data["stress_level"].iloc[-2]) or 0
    return trends

def generate_feedback(latest, trends):
    messages = []
    work = latest.get("work_hours") or 0
    sleep = latest.get("sleep_hours") or 0
    stress = latest.get("stress_level") or 0
    job_sat = latest.get("job_satisfaction") or 5
    delta_work = trends.get("delta_work") or 0
    delta_sleep = trends.get("delta_sleep") or 0
    delta_stress = trends.get("delta_stress") or 0

    if work > 45 or delta_work > 2:
        messages.append("High workload — consider cutting hours or delegating.")
    elif work > 35:
        messages.append("Moderate workload. Keep an eye on it.")
    else:
        messages.append("Workload looks sustainable.")

    if sleep < 6 or delta_sleep < -1:
        messages.append("Sleep is low — this is the #1 burnout accelerator.")
    else:
        messages.append("Sleep is in a good range.")

    if stress > 7 or delta_stress > 1:
        messages.append("Stress is climbing — try breathing exercises or a walk.")
    else:
        messages.append("Stress levels are manageable.")

    if job_sat < 4:
        messages.append("Job satisfaction is low — maybe it’s time to update the résumé… just in case.")
    else:
        messages.append("Job satisfaction is solid.")

    return messages

def risk_category(score):
    if score < 3.5:
        return "Low", "#2ecc71"
    elif score < 6.5:
        return "Medium", "#f39c12"
    else:
        return "High", "#e74c3c"

def generate_pdf(user_id, pred, messages):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Burnout Risk Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"User ID: {user_id[:8]}...")
    c.drawString(50, 750, f"Score: {pred:.2f} → {risk_category(pred)[0]} Risk")
    y = 720
    for msg in messages:
        c.drawString(70, y, "• " + msg)
        y -= 25
    c.save()
    buffer.seek(0)
    return buffer

# ========================
# LOAD MODEL
# ========================
model = joblib.load("burnout_xgb_model.pkl")
scaler = joblib.load("burnout_scaler.pkl")

# ========================
# SIDEBAR + USER ID
# ========================
st.sidebar.title("User")
username = st.sidebar.text_input("Username (optional, for history)")

if not username:
    if "guest_uuid" not in st.session_state:
        st.session_state.guest_uuid = str(uuid.uuid4())
    user_id = st.session_state.guest_uuid
    st.sidebar.caption(f"Guest mode: {user_id[:8]}...")
else:
    user_id = username

# Live counter (viral!)
all_scores = load_all_scores()
st.sidebar.metric("People tested so far", len(all_scores))
if len(all_scores) > 0:
    st.sidebar.caption(f"Global average: {all_scores['score'].mean():.2f}")

# ========================
# MAIN UI
# ========================
st.title("Burnout Risk Predictor")
st.markdown("Answer 10 quick questions → see your risk + live comparison with everyone else")

c1, c2 = st.columns(2)
with c1:
    work_hours = st.number_input("Work hours per week", 0, 100, 40)
    sleep_hours = st.number_input("Sleep hours per night", 0.0, 24.0, 7.5)
    stress_level = st.slider("Current stress level (1–10)", 1, 10, 5)
    job_satisfaction = st.slider("Job satisfaction (1–10)", 1, 10, 6)
    support_level = st.slider("Support from colleagues/manager (1–10)", 1, 10, 6)

with c2:
    exercise_days = st.number_input("Exercise days per week", 0, 7, 3)
    exercise_minutes = st.number_input("Avg exercise minutes per day", 0, 300, 30)
    remote_work = st.selectbox("Do you work remotely?", ("No", "Yes"))
    remote_work = 1 if remote_work == "Yes" else 0
    caffeine_mg = st.number_input("Daily caffeine (mg)", 0, 1000, 200)
    screen_time_hours = st.number_input("Daily screen time (hours)", 0.0, 24.0, 7.0)

# ========================
# PREDICT BUTTON
# ========================
if st.button("Predict My Burnout Risk", type="primary", use_container_width=True):

    # Prepare data
    new_data = pd.DataFrame([{
        "work_hours": work_hours, "sleep_hours": sleep_hours, "stress_level": stress_level,
        "job_satisfaction": job_satisfaction, "support_level": support_level,
        "exercise_days": exercise_days, "exercise_minutes": exercise_minutes,
        "remote_work": remote_work, "caffeine_mg": caffeine_mg,
        "screen_time_hours": screen_time_hours
    }])

    # Predict
    scaled = scaler.transform(new_data)
    pred = float(model.predict(scaled)[0])

    # SAVE everywhere
    save_user_data(user_id, new_data.iloc[0].to_dict(), pred)
    save_score(pred)   # This makes the live graph grow!

    # Results
    st.success(f"### Your Burnout Score: **{pred:.2f} / 10**")
    category, color = risk_category(pred)
    st.markdown(f"**Risk Level:** <span style='color:{color};font-size:1.5em'>{category}</span>", unsafe_allow_html=True)

    # LIVE POPULATION GRAPH
    all_scores = load_all_scores()
    total = len(all_scores)

    st.subheader(f"Your Score vs {total:,} Other People (Live & Updating)")

    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bins, _ = ax.hist(all_scores["score"], bins=20, color="#3498db", edgecolor="black", alpha=0.8)
    ax.axvline(pred, color="#e74c3c", linewidth=5, label=f"You: {pred:.2f}")
    ax.set_xlabel("Burnout Risk Score", fontsize=12)
    ax.set_ylabel("Number of People", fontsize=12)
    ax.set_title(f"Live Burnout Score Distribution — {total:,} people (updates instantly)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Highlight your bin
    bin_idx = np.digitize(pred, bins) - 1
    if 0 <= bin_idx < len(ax.patches):
        ax.patches[bin_idx].set_facecolor("#e74c3c")
        ax.patches[bin_idx].set_alpha(0.9)

    st.pyplot(fig)

    # Personal feedback
    df = load_user_data()
    trends = get_trends(user_id, df)
    latest = df[df["user_id"] == user_id].sort_values("date").iloc[-1].to_dict()
    messages = generate_feedback(latest, trends)

    st.subheader("Your Personalized Feedback")
    for msg in messages:
        st.write("• " + msg)

    # PDF download
    pdf = generate_pdf(user_id, pred, messages)
    st.download_button("Download PDF Report", pdf, "burnout_report.pdf", "application/pdf")

# Final touch
if len(all_scores) == 0:
    st.info("Be the first person to take the test — watch the graph appear live!")

st.caption("Built with love by a fellow burnt-out human. Share this if it helps.")