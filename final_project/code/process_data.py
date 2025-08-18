import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import CATEGORICAL, NUMERIC, TARGET

RAW_PATH = "data/raw/students_performance.csv"
PROC_DIR = "data/processed"

os.makedirs(PROC_DIR, exist_ok=True)

def maybe_create_sample():
    if not os.path.exists(RAW_PATH):
        # Create a tiny synthetic sample if dataset missing
        df = pd.DataFrame([
            {"gender":"female","ethnicity":"group B","parent_education":"bachelor","lunch":"standard","test_prep":"completed","math_score":72,"reading_score":90,"writing_score":88,"final_grade_class":"A"},
            {"gender":"male","ethnicity":"group C","parent_education":"some college","lunch":"free/reduced","test_prep":"none","math_score":50,"reading_score":52,"writing_score":47,"final_grade_class":"D"},
            {"gender":"female","ethnicity":"group A","parent_education":"master","lunch":"standard","test_prep":"completed","math_score":85,"reading_score":86,"writing_score":84,"final_grade_class":"A"},
            {"gender":"male","ethnicity":"group D","parent_education":"high school","lunch":"standard","test_prep":"none","math_score":60,"reading_score":65,"writing_score":62,"final_grade_class":"C"},
        ])
        df.to_csv(RAW_PATH, index=False)

def main():
    maybe_create_sample()
    df = pd.read_csv(RAW_PATH)

    # Basic cleaning: drop NA, clip scores
    df = df.dropna()
    for col in NUMERIC:
        df[col] = df[col].clip(0, 100)

    # Train/val/test split
    train_df, rest = train_test_split(df, test_size=0.3, random_state=42, stratify=df[TARGET])
    val_df, test_df = train_test_split(rest, test_size=0.5, random_state=42, stratify=rest[TARGET])

    train_df.to_csv(os.path.join(PROC_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROC_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROC_DIR, "test.csv"), index=False)
    print("Saved processed splits to data/processed/")

if __name__ == "__main__":
    main()
