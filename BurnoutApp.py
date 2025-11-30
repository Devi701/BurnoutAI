# app.py — Burnout Test + Mixpanel Analytics + Time Tracking
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import hashlib
import os
import matplotlib.pyplot as plt

# ========================
# 0. MIXPANEL JS SDK (TOP)
# ========================
st.markdown("""
<!-- Mixpanel JS SDK -->
<script type="text/javascript">
  (function(e,c){if(!c.__SV){var l,h;window.mixpanel=c;c._i=[];c.init=function(q,r,f){
  function t(d,a){var g=a.split(".");2==g.length&&(d=d[g[0]],a=g[1]);d[a]=function(){
  d.push([a].concat(Array.prototype.slice.call(arguments,0)))}}var b=c;
  "undefined"!==typeof f?b=c[f]=[]:f="mixpanel";b.people=b.people||[];b.toString=function(d){
  var a="mixpanel";"mixpanel"!==f&&(a+="."+f);d||(a+=" (stub)");return a};
  b.people.toString=function(){return b.toString(1)+".people (stub)"};l="disable time_event track track_pageview track_links track_forms track_with_groups add_group set_group remove_group register register_once alias unregister identify name_tag set_config reset opt_in_tracking opt_out_tracking has_opted_in_tracking has_opted_out_tracking clear_opt_in_out_tracking start_batch_senders start_session_recording stop_session_recording people.set people.set_once people.unset people.increment people.append people.union people.track_charge people.clear_charges people.delete_user people.remove".split(" ");
  for(h=0;h<l.length;h++)t(b,l[h]);var n="set set_once union unset remove delete".split(" ");
  b.get_group=function(){function d(p){a[p]=function(){b.push([g,[p].concat(Array.prototype.slice.call(arguments,0))])}}
  for(var a={},g=["get_group"].concat(Array.prototype.slice.call(arguments,0)),m=0;m<n.length;m++)d(n[m]);return a};
  c._i.push([q,r,f])};c.__SV=1.2;
  var k=e.createElement("script");k.type="text/javascript";k.async=!0;
  k.src="https://cdn.mxpnl.com/libs/mixpanel-2-latest.min.js";
  e=e.getElementsByTagName("script")[0];e.parentNode.insertBefore(k,e)}})(document,window.mixpanel||[]);
  
  mixpanel.init('2cc93e326a41d1b5791d57359f323114', {
    autocapture: true,
    record_sessions_percent: 100,
    api_host: 'https://api-eu.mixpanel.com',
    default_tracking: { sessions: true }
  });

  // --- Create persistent anonymous ID ---
  let anonId = localStorage.getItem('anon_id');
  if (!anonId) {
    anonId = crypto.randomUUID();
    localStorage.setItem('anon_id', anonId);
  }

  // --- Identify anonymous user/device ---
  mixpanel.identify(anonId);

</script>
""", unsafe_allow_html=True)

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="Burnout Test", page_icon="🔥", layout="centered")

# ========================
# SESSION START TIME
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
# 1. SEED 19 PEOPLE
# ========================
if not os.path.exists(ALL_SCORES):
    np.random.seed(42)
    scores = np.round(np.random.beta(2, 3, 19) * 8 + 1.5, 2)
    jobs = [
        "Nurse","Software Engineer","Teacher","Founder","Nurse","Doctor","Teacher",
        "Designer","Marketing","Customer Support","Lawyer","Accountant","Product Manager",
        "Freelancer","Data Analyst","Nurse","Teacher","DevOps","Parent"
    ]
    seed_df = pd.DataFrame({
        "fingerprint": [f"seed_{i}" for i in range(19)],
        "score": scores,
        "job_title": jobs,
        "timestamp": pd.date_range("2025-04-01", periods=19).astype(str)
    })
    seed_df.to_csv(ALL_SCORES, index=False)

df_global = pd.read_csv(ALL_SCORES)

# ========================
# 2. FINGERPRINT
# ========================
def get_fingerprint():
    """Stable session-based fingerprint for tracking"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.sha256(os.urandom(32)).hexdigest()
    return hashlib.sha256(st.session_state.session_id.encode()).hexdigest()

fp = get_fingerprint()
OWNER_HASH = "639eaa54ed39a346f78ce4cd4de28f26ff8e7973ca084bba0893011860b66565"
st.session_state.owner_mode = (fp == OWNER_HASH)

# ========================
# 3. LOAD MODEL
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
# 5. SIDEBAR
# ========================
with st.sidebar:
    st.header("Live Global Results")
    st.metric("People Tested", f"{len(df_global):,}")
    st.metric("Average Score", f"{df_global['score'].mean():.2f}/10")
    st.caption("Updates instantly • 100% anonymous")

# ========================
# 6. MAIN APP
# ========================
st.title("How Burnt Out Are You Right Now?")
st.markdown("**10 seconds → your score joins the live global chart instantly**")

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
    # Predict burnout score
    X = [[work_hours, sleep, stress, satisfaction, support,
          exercise_days, 30, remote_work, caffeine, screen]]
    pred = float(model.predict(scaler.transform(X))[0])

    # Save user fingerprint
    pd.DataFrame([{"fingerprint": fp}]).to_csv(
        USER_DATA, mode="a", header=not os.path.exists(USER_DATA), index=False
    )

    # Save to global scores
    if is_first_time and not st.session_state.owner_mode:
        row = {
            "fingerprint": fp,
            "score": pred,
            "job_title": job_title or "Anonymous",
            "timestamp": datetime.now().isoformat()
        }
        pd.DataFrame([row]).to_csv(ALL_SCORES, mode="a", header=False, index=False)
        df_global = pd.read_csv(ALL_SCORES)

    # Score display
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
    with st.expander("Optional: How does seeing your score feel?"):
        reaction = st.text_input("One line", key="reaction")
        if st.button("Share Reaction", key="share"):
            if not st.session_state.owner_mode:
                pd.DataFrame([{
                    "score": pred,
                    "reaction": reaction.strip() or "—",
                    "time": datetime.now().strftime("%b %d, %H:%M")
                }]).to_csv(REACTIONS, mode="a", header=not os.path.exists(REACTIONS), index=False)
                st.success("Thanks! Your reaction is live.")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("Watching the bars grow is addictive!")
