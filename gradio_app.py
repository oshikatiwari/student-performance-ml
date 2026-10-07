"""
Gradio Interface for Student Performance Prediction
Customized to match the exact Semantic Recommender prototype layout:
- Left-aligned header
- Horizontal controls with top-right action button (#e3e8f1 soft slate theme)
- Clean card-based 'Prediction Results' section mirroring the prototype card grid
- Dual-color distribution plot (Olive #848408 & Maroon #800020)
- Predicts ONLY when the user clicks 'Find predictions' (no premature initial predictions)
"""
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model_pipeline.joblib"

RMSE_ESTIMATE = 6.949
PASSING_THRESHOLD = 50.0

if MODEL_PATH.exists():
    pipeline = joblib.load(MODEL_PATH)
else:
    pipeline = None


def predict_student(
    previous_score: float,
    attendance: float,
    assignments: float,
    backlogs: float,
    study_hours: float,
    sleep_hours: float,
    participation: float,
    extracurricular: float
):
    if pipeline is None:
        return (
            "<p style='color: red;'>Error: Model pipeline artifact not found.</p>",
            gr.update(visible=False),
            gr.update(visible=False)
        )

    input_df = pd.DataFrame([{
        "StudyHours": study_hours,
        "AttendancePercentage": attendance,
        "PreviousExamScore": previous_score,
        "AssignmentsCompleted": assignments,
        "SleepHours": sleep_hours,
        "ExtracurricularHours": extracurricular,
        "ClassParticipation": participation,
        "PreviousBacklogs": backlogs
    }])

    raw_pred = pipeline.predict(input_df)[0]
    score = float(np.clip(raw_pred, 0.0, 100.0))

    ci_lower = max(0.0, round(score - 1.96 * RMSE_ESTIMATE, 1))
    ci_upper = min(100.0, round(score + 1.96 * RMSE_ESTIMATE, 1))

    # Calculate probabilities using normal distribution CDF
    z_score = (score - PASSING_THRESHOLD) / RMSE_ESTIMATE
    pass_prob = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    pass_prob = float(np.clip(pass_prob, 0.01, 0.99))
    risk_prob = 1.0 - pass_prob

    if score >= 75.0:
        standing_tier = "High Distinction"
        tier_color = "#1b5e20"
        badge_bg = "#e8f5e9"
        advice = "Excellent academic momentum. Maintain current study patterns and steady sleep schedule."
    elif score >= 50.0:
        standing_tier = "Satisfactory Pass"
        tier_color = "#0d47a1"
        badge_bg = "#e3f2fd"
        advice = "On track to pass successfully. Increasing weekly study hours can push performance to distinction."
    else:
        standing_tier = "At-Risk of Failing"
        tier_color = "#b71c1c"
        badge_bg = "#ffebee"
        advice = "Proactive academic intervention recommended. Prioritize clearing pending backlogs and attending tutorial office hours."

    # Top Cards HTML matching prototype card grid
    cards_html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-top: 8px; margin-bottom: 20px;">
        <!-- Card 1: Score -->
        <div style="background: #ffffff; border: 1px solid #e0e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Score</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: #1e293b; margin: 8px 0 4px 0;">{score:.1f} <span style="font-size: 1.1rem; color: #94a3b8; font-weight: 500;">/ 100</span></div>
            <div style="display: inline-block; background: {badge_bg}; color: {tier_color}; padding: 3px 10px; border-radius: 12px; font-size: 0.82rem; font-weight: 600;">{standing_tier}</div>
        </div>

        <!-- Card 2: Pass Probability -->
        <div style="background: #ffffff; border: 1px solid #e0e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Pass Probability</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: #848408; margin: 8px 0 4px 0;">{pass_prob * 100:.1f}%</div>
            <div style="color: #64748b; font-size: 0.82rem;">Score &ge; 50 threshold</div>
        </div>

        <!-- Card 3: At-Risk Probability -->
        <div style="background: #ffffff; border: 1px solid #e0e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">At-Risk Probability</div>
            <div style="font-size: 2.4rem; font-weight: 700; color: #800020; margin: 8px 0 4px 0;">{risk_prob * 100:.1f}%</div>
            <div style="color: #64748b; font-size: 0.82rem;">Deficiency risk index</div>
        </div>

        <!-- Card 4: 95% Confidence Interval -->
        <div style="background: #ffffff; border: 1px solid #e0e4e8; border-radius: 8px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">95% Confidence Range</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #1e293b; margin: 8px 0 4px 0;">{ci_lower}% &ndash; {ci_upper}%</div>
            <div style="color: #64748b; font-size: 0.82rem;">Based on &plusmn;1.96 RMSE (6.95)</div>
        </div>
    </div>
    """

    # Bar chart (Olive #848408 & Maroon #800020)
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=180)
    categories = ["Pass Probability", "At-Risk Probability"]
    probabilities = [pass_prob * 100, risk_prob * 100]
    bar_colors = ["#848408", "#800020"]

    bars = ax.bar(categories, probabilities, color=bar_colors, width=0.32)
    ax.set_ylabel("Probability (%)", fontsize=9, color="#555555")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], color="#777777", fontsize=8.5)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, color="#333333", fontsize=9.5, fontweight="600")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#d0d5dd")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#333333"
        )

    plt.tight_layout()

    # Advisory banner
    advisory_html = f"""
    <div style="background: #f8fafc; border-left: 4px solid {tier_color}; border-radius: 4px; padding: 12px 16px; margin-top: 10px;">
        <span style="font-weight: 600; color: #1e293b;">Advisor Guidance:</span>
        <span style="color: #475569; margin-left: 6px;">{advice}</span>
    </div>
    """

    return cards_html, gr.update(value=fig, visible=True), gr.update(value=advisory_html, visible=True)


# Custom styling for prototype look:
# Soft pastel slate-blue action button (#e3e8f1), clean inputs, elegant typography
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

#recommend_btn {
    background: #e3e8f1 !important;
    color: #1e293b !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border: 1px solid #ccd5e2 !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    height: 100% !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}

#recommend_btn:hover {
    background: #d4dde9 !important;
    border-color: #b8c4d4 !important;
}

#recommend_btn:active {
    background: #c5d1e0 !important;
}
"""

initial_placeholder = """
<div style="background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 32px 20px; text-align: center; color: #64748b; margin-top: 10px;">
    <div style="font-size: 1.8rem; margin-bottom: 8px;">📊</div>
    <div style="font-size: 1.05rem; font-weight: 600; color: #334155;">No predictions generated yet</div>
    <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 4px;">
        Adjust the student background and study metrics above, then click <strong>Find predictions</strong> to forecast final exam results.
    </div>
</div>
"""

with gr.Blocks(title="Student Performance Predictor") as demo:
    # 1. Clean left-aligned title matching prototype
    gr.Markdown("# Student Performance Predictor")

    # 2. Controls Row: Inputs on the left, action button on the right
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Row():
                prev_score = gr.Slider(0.0, 100.0, value=72.0, step=0.5, label="Previous Exam Score (0–100)")
                attendance = gr.Slider(0.0, 100.0, value=82.0, step=1.0, label="Attendance Percentage (%)")
                assignments = gr.Slider(0.0, 100.0, value=85.0, step=1.0, label="Assignments Completed (%)")
                backlogs = gr.Number(value=0, precision=0, label="Previous Backlogs")

            with gr.Row():
                study_hours = gr.Slider(0.0, 16.0, value=5.0, step=0.5, label="Daily Study Hours")
                sleep_hours = gr.Slider(3.0, 12.0, value=7.0, step=0.5, label="Daily Sleep Hours")
                participation = gr.Slider(0.0, 10.0, value=6.5, step=0.5, label="Class Participation (0–10)")
                extracurricular = gr.Slider(0.0, 20.0, value=4.0, step=0.5, label="Extracurricular Hours")

        with gr.Column(scale=1, min_width=180):
            predict_btn = gr.Button("Find predictions", elem_id="recommend_btn")

    # 3. Section Title matching prototype "Recommendations" heading
    gr.Markdown("## Recommendations")

    # 4. Result cards & visualizations (Initialized with clean placeholder, no auto-trigger)
    cards_out = gr.HTML(value=initial_placeholder)
    chart_out = gr.Plot(visible=False)
    advisory_out = gr.HTML(visible=False)

    # Triggered ONLY when user explicitly clicks the button
    predict_btn.click(
        fn=predict_student,
        inputs=[
            prev_score, attendance, assignments, backlogs,
            study_hours, sleep_hours, participation, extracurricular
        ],
        outputs=[cards_out, chart_out, advisory_out]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="slate", neutral_hue="slate")
    )
