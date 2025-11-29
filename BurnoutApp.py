import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import hashlib
import os
import matplotlib.pyplot as plt

# ========================
# GOOGLE ANALYTICS (load BEFORE page_config)
# ========================
st.markdown("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-1YTTPBS985"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-1YTTPBS985');
</script>
""", unsafe_allow_html=True)

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="Burnout Test", page_icon="🔥", layout="centered")


# ========================
# FILE PATHS
# ========================
ALL_SCORES = "scores.csv"
USER_DATA = "users.csv"
REACTIONS = "reactions.csv"
MODEL_PATH = "burnout_xgb_model.pkl"
SCALER_PATH = "burnout_scaler.pkl"


# ========================
# 1. SEED 19 PEOPLE
# ========================
if not os.path.exists(ALL_SCORES):
    np.random.seed(42)
    scores = np.round(np.random.beta(2, 3, 19) * 8 + 1.5, 2)
    jobs = ["Nurse","Software Engineer","Teacher","Founder","Nurse","Doctor","Teacher",
            "Designer","Marketing","Customer Support","Lawyer","Accountant","Product Manager",
            "Freelancer","Data Analyst","Nurse","Teacher","DevOps","Parent"]

    seed_df = pd.DataFrame({
        "fingerprint": [f"seed_{i}" for i in range(19)],
        "score": scores,
        "job_title": jobs,
        "timestamp": pd.date_range("2025-04-01", periods=19).astype(str)
    })

    seed_df.to_csv(ALL_SCORES, index=False)

df_global = pd.read_csv(ALL_SCORES)


# ========================
# 2. FINGERPRINT (Streamlit-safe)
# ========================
def get_fingerprint():
    """
    More stable fingerprint: session + IP + time bucket.
    Avoids Streamlit crashes (st.context.headers removed).
    """
    session = st.session_state.get("session_id")

    if session is None:
        session = hashlib.sha256(os.urandom(32)).hexdigest()
        st.session_state.session_id = session

    return hashlib.sha256(session.encode()).hexdigest()

fp = get_fingerprint()

# OWNER MODE (change your hash)
OWNER_HASH = "639eaa54ed39a346f78ce4cd4de28f26ff8e7973ca084bba0893011860b66565"
st.session_state.owner_mode = (fp == OWNER_HASH)


# ========================
# 3. LOAD MODEL SAFELY
# ========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

model, scaler = load_model()

if model is None:
    st.error("Model missing — upload burnout_xgb_model.pkl and burnout_scaler.pkl")
    st.stop()


# ========================
# 4. DUPLICATE PROTECTION
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
    st.caption("Updates automatically • 100% anonymous")


# ========================
# MAIN APP
# ========================
st.title("How Burnt Out Are You?")
st.markdown("**10 seconds → Score updates global chart instantly**")

col1, col2 = st.columns(2)

with col1:
    work_hours = st.slider("Work hours/week", 0, 100, 40)
    sleep = st.slider("Sleep/night (hours)", 4.0, 12.0, 7.5, 0.5)
    stress = st.slider("Current stress level", 1, 10, 5)
    satisfaction = st.slider("Job satisfaction", 1, 10, 7)
    support = st.slider("Support from team/manager", 1, 10, 6)

with col2:
    exercise_days = st.slider("Exercise days/week", 0, 7, 3)
    remote = st.selectbox("Work setup", ["Office / Hybrid", "Fully Remote"])
    remote_work = 1 if "Remote" in remote else 0
    caffeine = st.slider("Daily caffeine (mg)", 0, 1000, 200)
    screen = st.slider("Screen time/day (hours)", 2.0, 20.0, 8.0)
    job_title = st.text_input("Job title (optional)")


# ========================
# SHOW RESULT
# ========================
if st.button("Show My Score", type="primary", use_container_width=True):

    X = [[work_hours, sleep, stress, satisfaction, support,
          exercise_days, 30, remote_work, caffeine, screen]]

    pred = float(model.predict(scaler.transform(X))[0])

    # Mark user seen
    pd.DataFrame([{"fingerprint": fp}]).to_csv(
        USER_DATA, mode="a", header=not os.path.exists(USER_DATA), index=False
    )

    # Save score first time only (unless owner)
    if is_first_time and not st.session_state.owner_mode:
        row = {
            "fingerprint": fp,
            "score": pred,
            "job_title": job_title or "Anonymous",
            "timestamp": datetime.now().isoformat()
        }
        pd.DataFrame([row]).to_csv(ALL_SCORES, mode="a", header=False, index=False)
        df_global = pd.read_csv(ALL_SCORES)

    # Score block
    col = "#e74c3c" if pred >= 6.5 else "#f39c12" if pred >= 3.5 else "#2ecc71"
    risk = "HIGH RISK" if pred >= 6.5 else "MEDIUM RISK" if pred >= 3.5 else "LOW RISK"

    st.markdown(f"""
    <div style="text-align:center;padding:60px;border-radius:30px;background:{col}15;border:6px solid {col}">
        <h1 style="color:{col};margin:0">{pred:.2f}<small>/10</small></h1>
        <h2 style="color:{col}">{risk}</h2>
    </div><br>""", unsafe_allow_html=True)

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_global["score"], bins=20, edgecolor="white")
    ax.axvline(pred, color=col, linewidth=4)
    ax.set_title(f"You vs {len(df_global):,} people worldwide")
    st.pyplot(fig)

    # Reactions
    with st.expander("How does your score make you feel? (anonymous)"):
        r = st.text_input("One line")
        if st.button("Share"):
            if not st.session_state.owner_mode:
                pd.DataFrame([{
                    "score": pred,
                    "reaction": r.strip() or "—",
                    "time": datetime.now().strftime("%b %d, %H:%M")
                }]).to_csv(REACTIONS, mode="a", header=not os.path.exists(REACTIONS), index=False)
                st.success("Live!")

    if os.path.exists(REACTIONS):
        st.markdown("#### What others said")
        df_r = pd.read_csv(REACTIONS).tail(10)
        for _, row in df_r.iterrows():
            st.markdown(f"**{row['score']}/10** — *“{row['reaction']}”* ⋅ {row['time']}")

else:
    st.info("Click above to get your score.")


st.markdown("---")
st.caption("Informational only • Not medical advice • 100% anonymous")
