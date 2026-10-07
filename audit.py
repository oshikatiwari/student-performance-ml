"""
Step 1: Data Audit, Sentinel Cleaning, and Pre-Exam Leakage Check

"""
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/student_performance.csv")

print("=" * 70)
print("1. DATASET SHAPE & MISSING VALUES")
print("=" * 70)
print(f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}\n")

# Check missing values
missing = df.isnull().sum()
for col in df.columns:
    pct = (missing[col] / len(df)) * 100
    print(f"  {col:<24s}: {missing[col]:3d} missing ({pct:4.1f}%)")

print("\n" + "=" * 70)
print("2. TARGET AUDIT (FinalExamScore)")
print("=" * 70)
target = df["FinalExamScore"]
print(f"Target Mean:   {target.mean():.2f}")
print(f"Target Median: {target.median():.2f}")
print(f"Target Min:    {target.min():.2f}  <-- Corrupt negative score!")
print(f"Target Max:    {target.max():.2f} <-- Corrupt score > 100!")

# Find bad records with corrupt target values
bad_records = df[(target < 0) | (target > 100)]
print(f"\nCorrupt Target Records to Drop ({len(bad_records)} rows):")
for _, row in bad_records.iterrows():
    print(f"  - Student ID {int(row['ID'])}: FinalExamScore = {row['FinalExamScore']}")

print("\n" + "=" * 70)
print("3. SENTINEL & BOUNDARY CHECKS")
print("=" * 70)
print(f"StudyHours < 0 (Sentinel -1):        {(df['StudyHours'] < 0).sum()} record(s)")
print(f"StudyHours > 24 (Out of bounds):     {(df['StudyHours'] > 24).sum()} record(s)")
print(f"Attendance > 100% (Out of bounds):   {(df['AttendancePercentage'] > 100).sum()} record(s)")
print(f"SleepHours > 24 (Out of bounds):     {(df['SleepHours'] > 24).sum()} record(s)")

print("\n" + "=" * 70)
print("4. PRE-EXAM LEAKAGE CHECK (WHAT TO KEEP VS DROP)")
print("=" * 70)
justifications = {
    "ID": ("Drop", "Arbitrary student identifier; no predictive signal."),
    "StudyHours": ("Keep", "Known pre-exam: student study commitment."),
    "AttendancePercentage": ("Keep", "Known pre-exam: semester classroom presence."),
    "PreviousExamScore": ("Keep", "Known pre-exam: historical exam benchmark."),
    "AssignmentsCompleted": ("Keep", "Known pre-exam: continuous term assessment."),
    "SleepHours": ("Keep", "Known pre-exam: physical readiness & daily rest."),
    "ExtracurricularHours": ("Keep", "Known pre-exam: co-curricular activity time."),
    "ClassParticipation": ("Keep", "Known pre-exam: instructor engagement score."),
    "PreviousBacklogs": ("Keep", "Known pre-exam: historical transcript backlogs."),
    "PostExamConfidence": ("DROP (LEAKAGE)", "TARGET LEAKAGE: Recorded AFTER the exam. Cannot exist before the exam!"),
    "FinalExamScore": ("Target", "Continuous target variable (0 - 100).")
}

for col, (decision, reason) in justifications.items():
    print(f"{col:<22s} | {decision:<14s} | {reason}")
print("=" * 70)