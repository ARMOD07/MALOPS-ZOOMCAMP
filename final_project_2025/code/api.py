from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from utils import CATEGORICAL, NUMERIC, decode_target

class StudentFeatures(BaseModel):
    gender: str
    ethnicity: str
    parent_education: str
    lunch: str
    test_prep: str
    math_score: float
    reading_score: float
    writing_score: float

app = FastAPI(title="Student Grade Classifier")

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

model = None
try:
    model = load_model()
except Exception:
    model = None

@app.get("/health")
def health():
    return {"status":"ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(features: StudentFeatures):
    global model
    if model is None:
        model = load_model()

    df = pd.DataFrame([features.dict()])
    y = model.predict(df)[0]
    grade = decode_target(y)
    return {"grade_class": grade, "class_id": int(y)}
