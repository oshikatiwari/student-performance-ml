"""
Step 2: Clean, Leakage-Proof Scikit-Learn Preprocessing Pipeline
"""
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# The 8 justified pre-exam features
PRE_EXAM_FEATURES: List[str] = [
    "StudyHours",
    "AttendancePercentage",
    "PreviousExamScore",
    "AssignmentsCompleted",
    "SleepHours",
    "ExtracurricularHours",
    "ClassParticipation",
    "PreviousBacklogs"
]

# Physical bounds for validation
FEATURE_BOUNDS: Dict[str, tuple] = {
    "StudyHours": (0.0, 24.0),
    "AttendancePercentage": (0.0, 100.0),
    "PreviousExamScore": (0.0, 100.0),
    "AssignmentsCompleted": (0.0, 100.0),
    "SleepHours": (0.0, 24.0),
    "ExtracurricularHours": (0.0, 24.0),
    "ClassParticipation": (0.0, 10.0),
    "PreviousBacklogs": (0.0, 15.0)
}


class PreExamFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts only justified pre-exam features.
    Silently drops non-semantic IDs and leaked post-exam signals (PostExamConfidence).
    """
    def __init__(self, feature_names: List[str] = PRE_EXAM_FEATURES):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X

        extracted = pd.DataFrame(index=X_df.index)
        for col in self.feature_names:
            if col in X_df.columns:
                extracted[col] = pd.to_numeric(X_df[col], errors="coerce")
            else:
                extracted[col] = np.nan
        return extracted


class SentinelAndBoundsSanitizer(BaseEstimator, TransformerMixin):
    """
    Maps negative sentinels (-1.0) to NaN for clean median imputation,
    and clips out-of-domain outliers to realistic physical limits.
    """
    def __init__(self, bounds: dict = FEATURE_BOUNDS):
        self.bounds = bounds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=PRE_EXAM_FEATURES)

        for col, (low, high) in self.bounds.items():
            if col in X_df.columns:
                # Convert negative sentinels to NaN
                mask_neg = X_df[col] < low
                X_df.loc[mask_neg, col] = np.nan
                # Clip upper physical bounds
                X_df[col] = X_df[col].clip(upper=high)

        return X_df.values


def build_pipeline(model=None) -> Pipeline:
    """
    Encapsulates all preprocessing into an atomic Scikit-Learn Pipeline.
    Guarantees zero data leakage because transformations are fitted exclusively on training data.
    """
    steps = [
        ("feature_selection", PreExamFeatureExtractor(PRE_EXAM_FEATURES)),
        ("sanitizer", SentinelAndBoundsSanitizer(FEATURE_BOUNDS)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps)


if __name__ == "__main__":
    test_pipe = build_pipeline()
    print("Scikit-Learn Pipeline successfully created:")
    print(test_pipe)