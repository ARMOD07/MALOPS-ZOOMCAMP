# 📚 Student Performance MLOps Pipeline (Local, No Cloud)

End-to-end pipeline predicting a student's final grade class (A–F) using demographic and academic features.

**Stack:** XGBoost, scikit-learn · MLflow (tracking + registry) · Prefect (orchestration) · FastAPI (serving) · Evidently (drift) · Docker · GitHub Actions (CI).

## Project Structure
```
mlops-student-pipeline/
├── code/
│   ├── process_data.py
│   ├── train_model.py
│   ├── prefect_pipeline.py
│   ├── api.py
│   ├── monitor.py
│   └── utils.py
├── data/
│   ├── raw/students_performance.csv
│   └── processed/...
├── models/
├── tests/
│   └── test_basic.py
├── requirements.txt
├── Makefile
├── Dockerfile
├── .gitignore
├── mlflow.db
└── .github/workflows/ci.yml
```

> **Note**: `mlflow.db` will be created at runtime; artifacts saved in `./mlartifacts`.

## Quickstart (Local)

1) Create venv and install:
```bash
python -m venv venv && source venv/bin/activate
make install
```

2) Launch MLflow UI (new terminal):
```bash
make run-mlflow
# open http://127.0.0.1:5000
```

3) Run end-to-end pipeline with Prefect:
```bash
make run-prefect
```

4) Serve the latest model with FastAPI:
```bash
make run-api
# POST http://127.0.0.1:8000/predict
```

## Docker (Optional)

Build:
```bash
docker build -t student-mlops:latest .
```

Run:
```bash
docker run --rm -p 8000:8000 -p 5000:5000 -v "$(pwd):/app" student-mlops:latest
```

## API Example
```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"gender":"female","ethnicity":"group B","parent_education":"bachelor","lunch":"standard","test_prep":"completed","math_score":72,"reading_score":90,"writing_score":88}'
```

## Evaluation Criteria Mapping
- **Cloud**: local only (dockerized) ✔
- **Experiment tracking + registry**: MLflow ✔
- **Workflow orchestration**: Prefect ✔
- **Deployment**: FastAPI + Docker ✔
- **Monitoring**: Evidently drift report ✔
- **Reproducibility**: Makefile, requirements, tests ✔
- **Best practices**: lint, tests, CI ✔
