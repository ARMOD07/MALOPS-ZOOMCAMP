import os
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset

PROC_DIR = "data/processed"

def main():
    ref = pd.read_csv(os.path.join(PROC_DIR, "train.csv"))
    cur = pd.read_csv(os.path.join(PROC_DIR, "val.csv"))

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/data_drift_report.html")
    print("Saved drift report to reports/data_drift_report.html")

if __name__ == "__main__":
    main()
