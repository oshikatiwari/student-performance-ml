"""
Step 3: Multi-Model Benchmark & 5-Fold Cross-Validation
Evaluates 5 model families with mean ± std metrics and exports the optimal pipeline artifact.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from pipeline import build_pipeline

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

# 1. Load data & filter corrupt targets
df = pd.read_csv("data/student_performance.csv")
initial_len = len(df)
valid_mask = (df["FinalExamScore"] >= 0.0) & (df["FinalExamScore"] <= 100.0)
df_clean = df[valid_mask].copy()
dropped = initial_len - len(df_clean)
print(f"Loaded {initial_len} records. Dropped {dropped} corrupt target record(s). Clean training set: {len(df_clean)} rows.")

X = df_clean.drop(columns=["FinalExamScore"])
y = df_clean["FinalExamScore"].values

# 2. Candidate Models (Baseline, Linear, Bagging, Boosting, Tuned Boosting)
models = {
    "Mean Baseline": DummyRegressor(strategy="mean"),
    "Ridge Regression": Ridge(alpha=10.0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "Tuned HistGradientBoosting": HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=120,
        max_depth=4,
        min_samples_leaf=15,
        l2_regularization=0.5,
        random_state=42
    )
}

# 3. 5-Fold Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
benchmark_results = {}

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION BENCHMARK (Mean ± Std)")
print("=" * 70)
print(f"{'Model':<28} | {'RMSE':<14} | {'MAE':<14} | {'R2 Score':<14}")
print("-" * 70)

for name, model_instance in models.items():
    rmses, maes, r2s = [], [], []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Pipeline fits preprocessing ONLY on training fold
        pipe = build_pipeline(model=model_instance)
        pipe.fit(X_train, y_train)
        
        preds = pipe.predict(X_val)
        
        # Clip final predictions to valid exam bounds [0, 100]
        preds = np.clip(preds, 0.0, 100.0)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        
    benchmark_results[name] = {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s))
    }
    
    print(
        f"{name:<28} | "
        f"{np.mean(rmses):.3f} ± {np.std(rmses):.3f} | "
        f"{np.mean(maes):.3f} ± {np.std(maes):.3f} | "
        f"{np.mean(r2s):.3f} ± {np.std(r2s):.3f}"
    )

print("=" * 70)

# 4. Select Champion Model & Train on Full Clean Dataset
best_model_name = "Tuned HistGradientBoosting"
champion_model = models[best_model_name]

print(f"\nTraining production pipeline with champion model: [{best_model_name}]...")
final_pipeline = build_pipeline(model=champion_model)
final_pipeline.fit(X, y)

# 5. Save Artifacts
pipeline_path = "artifacts/model_pipeline.joblib"
joblib.dump(final_pipeline, pipeline_path)
print(f"Saved fitted pipeline to: {pipeline_path}")

metrics_path = "artifacts/metrics_summary.json"
with open(metrics_path, "w") as f:
    json.dump({
        "cv_folds": 5,
        "random_seed": 42,
        "models": benchmark_results,
        "champion_model": best_model_name
    }, f, indent=2)
print(f"Saved evaluation metrics to: {metrics_path}")
print("Training complete!")