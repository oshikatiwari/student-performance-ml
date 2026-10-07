---
title: Student Performance Predictor
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.16.0
app_file: gradio_app.py
pinned: false
---

# Student Performance Prediction

Production-grade Machine Learning pipeline to predict student final examination scores (`FinalExamScore`) while rigorously guarding against target leakage, out-of-bounds sentinels, and model instability.

## 🚀 Key Highlights
- **Leakage-Proof Pipeline:** Custom `PreExamFeatureExtractor` automatically strips non-semantic IDs and post-exam target leakage (`PostExamConfidence`).
- **Sentinel Sanitization:** Capping out-of-bounds outliers and converting sentinel `-1.0` markers to `NaN` for median imputation.
- **5-Fold Cross-Validation:** Benchmarked across 5 distinct model families reporting Mean ± Std.
- **Error Diagnostics:** Residual plots and segment breakdowns identifying higher variance among students with $\ge 3$ backlogs.
- **Packaged CLI:** `predict.py` for reproducible batch inference on clean machines.
- **Modern Web Demos:** Clean, single-screen UI in both **Streamlit** and **Gradio**.
- **Model Card:** Documented in [REPORT.md](REPORT.md) detailing intended use, prohibited use, and limitations.

---

## 📊 Benchmark Summary (5-Fold CV)

| Model | CV RMSE | CV MAE | CV $R^2$ |
| :--- | :---: | :---: | :---: |
| **Mean Baseline** | $13.989 \pm 1.194$ | $11.126 \pm 0.937$ | $-0.010 \pm 0.010$ |
| **Ridge Regression** | $7.923 \pm 0.633$ | $5.957 \pm 0.357$ | $0.673 \pm 0.043$ |
| **Random Forest** | $7.865 \pm 0.826$ | $6.115 \pm 0.562$ | $0.678 \pm 0.054$ |
| **HistGradientBoosting** | $7.167 \pm 0.739$ | $5.556 \pm 0.507$ | $0.735 \pm 0.025$ |
| **Tuned Booster (Champion)** | **$6.949 \pm 0.655$** | **$5.370 \pm 0.457$** | **$0.749 \pm 0.030$** |

---

## 🛠️ Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline Steps
```bash
# 1. Audit dataset and check for leakage
python audit.py

# 2. Train and benchmark models via 5-fold CV
python train.py

# 3. Generate residual diagnostic and feature importance plots
python evaluate.py

# 4. Run test set batch inference
python predict.py --input data/student_performance_test.csv --output submission.csv
```

### 3. Launch Web Application
```bash
# Streamlit Interface
streamlit run app.py

# Or Gradio Interface
python gradio_app.py
```

---

## 📁 Repository Structure
```
├── artifacts/
│   ├── feature_importance.png    # Permutation feature ranking
│   ├── metrics_summary.json      # 5-fold CV numerical results
│   ├── model_pipeline.joblib     # Production trained pipeline
│   └── residual_plot.png         # Diagnostic error plots
├── data/
│   ├── student_performance.csv
│   └── student_performance_test.csv
├── app.py                        # Streamlit web app
├── audit.py                      # Data audit & sentinel detection
├── evaluate.py                   # Residual analysis & backlog segment study
├── gradio_app.py                 # Gradio interactive web app
├── pipeline.py                   # Atomic Scikit-Learn Pipeline
├── predict.py                    # Packaged batch CLI
├── README.md                     # Project overview
├── REPORT.md                     # Full Technical Report & Model Card
├── requirements.txt              # Dependency specifications
└── submission.csv                # Generated test set predictions
```
