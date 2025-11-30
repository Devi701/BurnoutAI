# app.py — Burnout Test + Mixpanel Analytics + Time Tracking (Investor Ready)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import hashlib
import os
import matplotlib.pyplot as plt

# ========================
# 0. MIXPANEL JS SDK (via components.html)
# ========================
components.html("""
<script src="https://cdn.mxpnl.com/libs/mixpanel-2-latest.min.js"></script>
<script>
(function() {
    // Persistent anonymous user ID
    let anonId = localStorage.getItem('mp_anon');
    if (!anonId) {
        anonId = crypto.randomUUID();
        localStorage.setItem('mp_anon', anonId);
    }

    // Initialize Mixpanel
    mixpanel.init('2cc93e326a41d1b5791d57359f323114', {
        autocapture: true,
        record_sessions_percent: 100,
        api_host: 'https://api-eu.mixpanel.com',
        debug: false,
        default_tracking: { sessions: true }
    });

    // Identify anonymous user
    mixpanel.identify(anonId);

    // Track initial app open
    mixpanel.track('App Loaded');
})();
</script>
""", height=120)

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="Burnout Test", page_icon="🔥", layout="centered")

# ========================
# SESSION START
# ========================
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()

# ========================
# FILE PATHS
# ========================
ALL_SCORES = "scores.csv"
USER_DATA = "users.csv"
REACTIONS = "reactions.csv"
MODEL_PATH = "burnout_xgb_model.pkl"
SCALER_PATH = "burnout_scaler.pkl"

# ========================
# SEED DATA IF EMPTY
# ========================
if not os.path.exists(ALL_SCORES):
    np.random.seed(42)
    scores = np.round(np.random.beta(2,3,19)*8+1.5,2)
    jobs = [
        "Nurse","Software Engineer","Teacher","Founder","Nurse","Doctor","Teacher",
        "Designer","Marketing","Customer Support","Lawyer","Accountant","Product Manager",
        "Freelancer","Data Analyst","Nurse","Teacher","DevOps","Parent"
    ]
    pd.DataFrame({
        "fingerprint":[f"seed_{i}" for i in range(19)],
        "score":scores,
        "job_title":jobs,
        "timestamp": pd.date_range("2025-04-01", periods=19).astype(str)
    }).to_csv(ALL_SCORES,index=False)

df_global = pd.read_csv(ALL_SCORES)

# ========================
# FINGERPRINT
# ========================
def get_fingerprint():
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.sha256(os.urandom(32)).hexdigest()
    return hashlib.sha256(st.session_state.session_id.encode()).hexdigest()

fp = get_fingerprint()

# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

model, scaler = load_model()
if model is None:
    st.error("Model missing — upload burnout_xgb_model.pkl and burnout_scaler.pkl")
    st.stop()

# ========================
# DUPLICATE PROTECTION
# ========================
if os.path.exists(USER_DATA):
    history = pd.read_csv(USER_DATA)
else:
    history = pd.DataFrame(columns=["fingerprint"])

is_first_time = fp not in history["fingerprint"].values

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.header("Live Global Results")
    st.metric("People Tested", f"{len(df_global):,}")
    st.metric("Average Score", f"{df_global['score'].mean():.2f}/10")

# ========================
# MAIN APP
# ========================
st.title("How Burnt Out Are You Right Now?")
st.markdown("**10 seconds → your score joins the live global chart instantly**")

col1, col2 = st.columns(2)

with col1:
    work_hours = st.slider("Work hours/week",0,100,40)
    sleep = st.slider("Sleep/night (hours)",4.0,12.0,7.5,0.5)
    stress = st.slider("Current stress level",1,10,5)
    satisfaction = st.slider("Job satisfaction",1,10,7)
    support = st.slider("Support from team/manager",1,10,6)

with col2:
    exercise_days = st.slider("Exercise days/week",0,7,3)
    remote = st.selectbox("Work setup",["Office / Hybrid","Fully Remote"])
    remote_work = 1 if "Remote" in remote else 0
    caffeine = st.slider("Daily caffeine (mg)",0,1000,200)
    screen = st.slider("Screen time/day (hours)",2.0,20.0,8.0)
    job_title = st.text_input("Job title (optional)")

# ========================
# SHOW SCORE
# ========================
if st.button("Show My Score", type="primary", use_container_width=True):
    # Predict
    X = [[work_hours, sleep, stress, satisfaction, support, exercise_days, 30, remote_work, caffeine, screen]]
    pred = float(model.predict(scaler.transform(X))[0])

    # Save user fingerprint
    pd.DataFrame([{"fingerprint": fp}]).to_csv(USER_DATA, mode="a", header=not os.path.exists(USER_DATA), index=False)

    # Save to global scores
    if is_first_time:
        row = {"fingerprint":fp,"score":pred,"job_title":job_title or "Anonymous","timestamp":datetime.now().isoformat()}
        pd.DataFrame([row]).to_csv(ALL_SCORES,mode="a",header=False,index=False)
        df_global = pd.read_csv(ALL_SCORES)

    # Display score
    col = "#e74c3c" if pred >= 6.5 else "#f39c12" if pred >= 3.5 else "#2ecc71"
    risk = "HIGH RISK" if pred >=6.5 else "MEDIUM RISK" if pred>=3.5 else "LOW RISK"

    st.markdown(f"""
    <div style="text-align:center;padding:60px;border-radius:30px;background:{col}15;border:6px solid {col}">
        <h1 style="color:{col};margin:0">{pred:.2f}/10</h1>
        <h2 style="color:{col}">{risk}</h2>
    </div><br>
    """, unsafe_allow_html=True)

    # Histogram
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(df_global["score"], bins=20, edgecolor="white")
    ax.axvline(pred,color=col,linewidth=4)
    ax.set_title(f"You vs {len(df_global):,} people worldwide")
    st.pyplot(fig)

    # Mixpanel tracking for investor metrics
    components.html(f"""
    <script>
        const anonId = localStorage.getItem('mp_anon') || '';
        mixpanel.track("Burnout Score Calculated", {{
            score: {pred},
            job_title: "{job_title}",
            anon_id: anonId
        }});
    </script>
    """, height=60)

# ========================
# FOOTER — track duration
# ========================
session_duration = (datetime.now() - st.session_state.session_start).seconds
components.html(f"""
<script>
    const anonId = localStorage.getItem('mp_anon') || '';
    mixpanel.track("Session Ended", {{duration_sec: {session_duration}, anon_id: anonId}});
</script>
""", height=60)

st.markdown("---")
st.caption("Watching the bars grow is addictive!")
