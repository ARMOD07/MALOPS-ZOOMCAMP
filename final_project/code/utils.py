import pandas as pd

CATEGORICAL = ["gender","ethnicity","parent_education","lunch","test_prep"]
NUMERIC = ["math_score","reading_score","writing_score"]

TARGET = "final_grade_class"

GRADE_MAPPING = {"A":4,"B":3,"C":2,"D":1,"F":0}

def encode_target(y: pd.Series) -> pd.Series:
    return y.map(GRADE_MAPPING).astype(int)

def decode_target(y_int: int) -> str:
    rev = {v:k for k,v in GRADE_MAPPING.items()}
    return rev[int(y_int)]
