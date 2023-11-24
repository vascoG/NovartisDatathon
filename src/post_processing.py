import pandas as pd

from pathlib import Path


if __name__ == "__main__":
    # Load data
    PATH = Path("../data")
    submission_data = pd.read_csv(PATH / "submissionv2.csv")
    
    # transform prediction into rate within month

    submission_data["date"] = pd.to_datetime(submission_data["date"])

    submission_data["YearMonth"] = submission_data["date"].dt.to_period("M")

    submission_data["prediction"] = submission_data.groupby(["YearMonth","country","brand"])["prediction"].transform(lambda x: x / x.sum())

    submission_data["prediction"] = submission_data["prediction"].apply(lambda x: '{:.10f}'.format(x))

    submission_data = submission_data.drop(columns=["YearMonth"])


    # Save data

    submission_data.to_csv(PATH / "submissionv2normalized.csv", index=False)