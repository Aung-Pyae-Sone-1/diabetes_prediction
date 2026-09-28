"""Step 2: remove repeated rows and turn answers into numbers."""

import pandas as pd


def load_data():
    # Read the original survey answers.
    data = pd.read_csv("doc/diabetes_data_upload.csv")

    # Treat identical survey rows as repeated records. Keep one copy so the
    # same answers cannot appear in both training and test data.
    before = len(data)
    data = data.drop_duplicates().copy()
    print(f"Removed {before - len(data)} exact duplicate rows; {len(data)} rows remain.")

    # X means input columns; y means the answer we want to predict.
    # The target is 1 for Positive and 0 for Negative.
    y = data["class"].map({"Negative": 0, "Positive": 1})
    X = data.drop(columns="class")

    # Each answer has two possible values, so one number is enough.
    X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})
    for column in X.columns.drop(["Age", "Gender"]):
        X[column] = X[column].map({"No": 0, "Yes": 1})

    # Stop if new or blank answers were not converted.
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Found a missing or unexpected answer in the CSV file.")
    return X, y


if __name__ == "__main__":
    features, target = load_data()
    print(f"Features: {features.shape[1]}; positive cases: {target.sum()}")
