install:
\tpip install -r requirements.txt

lint:
\tblack . && flake8 .

test:
\tpytest -q

run-mlflow:
\tmlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

run-api:
\tuvicorn code.api:app --host 0.0.0.0 --port 8000 --reload

run-prefect:
\tpython code/prefect_pipeline.py --process True --train True --register True
