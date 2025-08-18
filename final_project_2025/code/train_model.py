import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from utils import CATEGORICAL, NUMERIC, TARGET, encode_target

PROC_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("student-performance")

def load_splits():
    train_df = pd.read_csv(f"{PROC_DIR}/train.csv")
    val_df = pd.read_csv(f"{PROC_DIR}/val.csv")
    return train_df, val_df

def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", model)])

def main():
    train_df, val_df = load_splits()

    X_train = train_df[CATEGORICAL + NUMERIC]
    y_train = encode_target(train_df[TARGET])

    X_val = val_df[CATEGORICAL + NUMERIC]
    y_val = encode_target(val_df[TARGET])

    with mlflow.start_run() as run:
        pipe = build_pipeline()
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_weighted", f1)

        mlflow.sklearn.log_model(pipe, artifact_path="model")
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipe, f)

        print(f"Trained. Accuracy={acc:.3f} F1={f1:.3f}. Saved to {model_path}")

if __name__ == "__main__":
    main()
