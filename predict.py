"""
Step 5: Packaged Inference CLI
Usage: python predict.py --input data/student_performance_test.csv --output submission.csv
"""
import argparse
import sys
import os
import joblib
import numpy as np
import pandas as pd

from pipeline import PreExamFeatureExtractor, SentinelAndBoundsSanitizer


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Final Exam Score from student data CSV.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV containing student records."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Path to save the resulting submission CSV."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/model_pipeline.joblib",
        help="Path to fitted model pipeline joblib."
    )
    return parser.parse_args()


def run_inference(input_path: str, output_path: str, model_path: str):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading input data from: {input_path}")
    df = pd.read_csv(input_path)

    if "ID" not in df.columns:
        print("Error: Input CSV must contain an 'ID' column.", file=sys.stderr)
        sys.exit(1)

    student_ids = df["ID"].values

    print(f"Loading serialized pipeline from: {model_path}")
    pipeline = joblib.load(model_path)

    print(f"Running inference on {len(df)} records...")
    raw_preds = pipeline.predict(df)

    # Post-processing: Bound to physical grade domain [0, 100] and round to 2 decimals
    bounded_preds = np.clip(raw_preds, 0.0, 100.0)
    final_preds = np.round(bounded_preds, 2)

    submission_df = pd.DataFrame({
        "ID": student_ids,
        "FinalExamScore": final_preds
    })

    # Save cleanly formatted CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Successfully generated predictions: {output_path}")
    print(f"Row count: {len(submission_df)}")
    print(f"Sample output preview:\n{submission_df.head()}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.input, args.output, args.model)
