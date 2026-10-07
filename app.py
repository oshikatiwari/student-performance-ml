from pathlib import Path
import sys
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root directory is on sys.path for Streamlit Cloud
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline
from pipeline import PreExamFeatureExtractor, SentinelAndBoundsSanitizer, build_pipeline

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = ROOT / "artifacts" / "model_pipeline.joblib"
DATA_PATH = ROOT / "data" / "student_performance.csv"

RMSE_ESTIMATE = 6.949
PASSING_THRESHOLD = 50.0

# Custom Styling to match exact UI / UX specifications
st.markdown("""
<style>
/* Exact deep navy button (#0d2a63) */
div.stButton > button:first-child {
    background-color: #0d2a63 !important;
    color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
}

div.stButton > button:first-child:hover {
    background-color: #123b8a !important;
    color: #ffffff !important;
    border: none !important;
}

div.stButton > button:first-child:active {
    background-color: #091f48 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    # 1. Try loading existing serialized pipeline
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    # 2. Resilient Cloud Fallback: If OS/Python pickle version differs on Streamlit Cloud,
    # immediately fit production champion in < 1 second on verified clean training data.
    from sklearn.ensemble import HistGradientBoostingRegressor
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        valid_mask = (df["FinalExamScore"] >= 0.0) & (df["FinalExamScore"] <= 100.0)
        df_clean = df[valid_mask].copy()
        X = df_clean.drop(columns=["FinalExamScore"])
        y = df_clean["FinalExamScore"].values
        champion = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=120,
            max_depth=4,
            min_samples_leaf=15,
            l2_regularization=0.5,
            random_state=42
        )
        pipe = build_pipeline(model=champion)
        pipe.fit(X, y)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(pipe, MODEL_PATH)
        except Exception:
            pass
        return pipe

    st.error("Model pipeline and training data not found.")
    return None


pipeline = load_pipeline()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🎓 Student Performance Predictor")
st.write(
    "Enter academic history and daily habits below, then click **Predict!** to forecast "
    "final examination performance and probability of passing."
)

st.divider()

# ---------------------------------------------------------
# Input Features
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📚 Academic Background")
    previous_score = st.slider("Previous Exam Score", min_value=0.0, max_value=100.0, value=72.0, step=0.5)
    attendance = st.slider("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
    assignments = st.slider("Assignments Completed (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
    backlogs = st.number_input("Previous Pending Backlogs", min_value=0, max_value=10, value=0, step=1)

with col2:
    st.markdown("#### 🕒 Study & Daily Habits")
    study_hours = st.slider("Daily Study Hours", min_value=0.0, max_value=16.0, value=5.0, step=0.5)
    sleep_hours = st.slider("Daily Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    participation = st.slider("Class Participation (0–10)", min_value=0.0, max_value=10.0, value=6.5, step=0.5)
    extracurricular = st.slider("Weekly Extracurricular Hours", min_value=0.0, max_value=20.0, value=4.0, step=0.5)

st.caption("Tip: Higher previous exam scores and fewer backlogs strongly increase the expected final score.")
st.divider()

# ---------------------------------------------------------
# Prediction Action & Results
# ---------------------------------------------------------
predict_button = st.button("Predict!")

if predict_button and pipeline is not None:
    features = pd.DataFrame([{
        "StudyHours": study_hours,
        "AttendancePercentage": attendance,
        "PreviousExamScore": previous_score,
        "AssignmentsCompleted": assignments,
        "SleepHours": sleep_hours,
        "ExtracurricularHours": extracurricular,
        "ClassParticipation": participation,
        "PreviousBacklogs": backlogs
    }])

    raw_prediction = pipeline.predict(features)[0]
    score = float(np.clip(raw_prediction, 0.0, 100.0))

    ci_lower = max(0.0, round(score - 1.96 * RMSE_ESTIMATE, 1))
    ci_upper = min(100.0, round(score + 1.96 * RMSE_ESTIMATE, 1))

    # Calculate Pass and At-Risk Probabilities using Normal CDF given predicted score and RMSE
    z_score = (score - PASSING_THRESHOLD) / RMSE_ESTIMATE
    pass_prob = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    pass_prob = float(np.clip(pass_prob, 0.01, 0.99))
    risk_prob = 1.0 - pass_prob

    # 1. Red indicator with small "Prediction" label right below the button
    st.markdown(
        "<div style='margin-top: 1rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;'>"
        "<span style='color: #e53935; font-size: 1.1rem;'>🔴</span>"
        "<span style='color: #555555; font-size: 1rem; font-weight: 500;'>Prediction</span>"
        "</div>",
        unsafe_allow_html=True
    )

    # 2. Prominent headline matching the maroon styling (#800020)
    if score >= PASSING_THRESHOLD:
        headline_text = "Student is likely to Pass"
    else:
        headline_text = "Student is at Risk of Failing"

    st.markdown(
        f"<h1 style='color: #800020; font-size: 2.2rem; font-weight: 700; margin-top: 0; margin-bottom: 0.5rem;'>"
        f"{headline_text}"
        f"</h1>",
        unsafe_allow_html=True
    )

    # 3. Prediction summary subtitle
    st.markdown(
        "<h3 style='color: #262730; font-size: 1.4rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1.2rem;'>"
        "Prediction summary"
        "</h3>",
        unsafe_allow_html=True
    )

    # 4. Two probability columns with large percentage numbers
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown(
            f"<div style='margin-bottom: 1.5rem;'>"
            f"<p style='color: #555555; font-size: 0.95rem; margin-bottom: 4px;'>Pass Probability</p>"
            f"<h1 style='font-size: 3rem; font-weight: 600; margin-top: 0; margin-bottom: 0;'>{pass_prob * 100:.1f}%</h1>"
            f"</div>",
            unsafe_allow_html=True
        )

    with p_col2:
        st.markdown(
            f"<div style='margin-bottom: 1.5rem;'>"
            f"<p style='color: #555555; font-size: 0.95rem; margin-bottom: 4px;'>At-Risk Probability</p>"
            f"<h1 style='font-size: 3rem; font-weight: 600; margin-top: 0; margin-bottom: 0;'>{risk_prob * 100:.1f}%</h1>"
            f"</div>",
            unsafe_allow_html=True
        )

    # 5. Clean Matplotlib bar chart with exact requested colors:
    # Olive: #848408 (Pass) | Maroon: #800020 (At-Risk)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)

    categories = ["Pass", "At Risk"]
    probabilities = [pass_prob * 100, risk_prob * 100]
    bar_colors = ["#848408", "#800020"]

    bars = ax.bar(categories, probabilities, color=bar_colors, width=0.28)

    ax.set_ylabel("Probability", fontsize=10, color="#555555")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], color="#777777", fontsize=9)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, color="#555555", fontsize=10)

    # Styling: clean minimalist frame (no top/right spines, faint horizontal guides)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#bbbbbb")
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 6. Additional Score & Academic Details
    st.divider()
    st.subheader("📊 Detailed Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Final Score", f"{score:.1f} / 100")
    m2.metric("95% Confidence Range", f"{ci_lower}% – {ci_upper}%")
    
    if score >= 75.0:
        tier_label = "🌟 Distinction"
    elif score >= 50.0:
        tier_label = "✅ Satisfactory Pass"
    else:
        tier_label = "⚠️ Academic Concern"
    m3.metric("Projected Standing", tier_label)

    st.progress(score / 100.0, text=f"Readiness Score: {score:.1f}%")

else:
    st.info("Adjust the academic background and daily habits above, then click **Predict!**")
