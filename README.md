## ⚠️ Disclaimer

This tool is for informational and educational purposes only.  
It is **not** medical advice and should not be used to diagnose or treat any condition.
# Burnout Risk Predictor

A simple, interactive Streamlit app that predicts the likelihood of **employee burnout** based on work-related factors such as workload, stress levels, work–life balance, and job satisfaction.  
Built to help individuals and teams identify early warning signs and make better decisions about mental wellbeing.

---

## Features

- **Burnout Risk Prediction** using a trained machine-learning model  
- **Interactive sliders & inputs** for work habits, stress, sleep, and job satisfaction  
- **Clear risk output** (Low / Medium / High burnout likelihood)  
- **Actionable suggestions** based on the predicted category  
- **Fast, lightweight, and fully browser-based** — no data is stored

---
## How It Works

The app uses:
- A trained classification model (e.g., Random Forest / Logistic Regression)  
- Inputs such as:
  - Weekly working hours  
  - Stress level  
  - Sleep quality  
  - Job satisfaction  
  - Work–life balance  
  - Support from management / team  
- The model returns a burnout-risk score and category  
- The app then provides tailored recommendations

All processing happens in real time directly in the app.

---

## How to Use

1. Open the app: **https://your-app-name.streamlit.app**  
2. Adjust the sliders and input fields to match your current work habits  
3. Click **Predict**  
4. Read your burnout-risk score and suggested improvements  

That's it — no account, no signup, no data stored.

## Tech Stack

- **Python 3.x**  
- **Streamlit** (UI layer)  
- **scikit-learn / XGBoost / TensorFlow** (depending on your model)  
- **pandas & numpy**  
- Optional: **joblib** for loading model files

---

## Why This App Is Useful

Burnout affects productivity, health, and overall wellbeing.  
This app helps users:

- Identify early symptoms of burnout  
- Self-assess work–life balance and stress  
- Take action before burnout becomes serious  
- Support wellbeing initiatives in workplaces  
- Run quick “what-if” scenarios (e.g., lowering work hours or increasing sleep)

---

#Run the App Locally

```bash
git clone https://github.com/devi701/BurnoutAI.git
cd your-repo
pip install -r requirements.txt
streamlit run app.py
