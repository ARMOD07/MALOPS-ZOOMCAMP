import argparse
from prefect import flow, task
from subprocess import run, CalledProcessError

@task
def process_data():
    print("Processing data...")
    run(["python", "code/process_data.py"], check=True)

@task
def train_model():
    print("Training model with MLflow...")
    run(["python", "code/train_model.py"], check=True)

@flow(name="student-mlops-flow")
def main_flow(process: bool=True, train: bool=True, register: bool=True):
    if process:
        process_data()
    if train:
        train_model()
    # For simplicity, registration is part of MLflow log in train step
    print("Flow complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--register", type=bool, default=True)
    args = parser.parse_args()
    main_flow(process=args.process, train=args.train, register=args.register)
