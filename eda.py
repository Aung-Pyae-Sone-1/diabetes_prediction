"""Step 1: look at the data before building models."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts as files, even without a display.
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data = pd.read_csv("doc/diabetes_data_upload.csv")
    Path("doc/eda_results").mkdir(exist_ok=True)

    # Check size, empty cells, repeated rows, and target balance.
    print(f"Rows: {len(data)}; columns: {len(data.columns)}")
    print(f"Missing cells: {data.isna().sum().sum()}")
    print(f"Exact duplicate rows: {data.duplicated().sum()}")
    print("Class counts:\n", data["class"].value_counts())

    # Plot how many people belong to each target class.
    data["class"].value_counts().reindex(["Negative", "Positive"]).plot.bar(
        color=["steelblue", "coral"]
    )
    plt.title("Survey class counts")
    plt.ylabel("Number of rows")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("doc/eda_results/class_counts.png")
    plt.close()

    # Compare the share of Yes answers for each symptom by class.
    symptoms = data.columns.drop(["Age", "Gender", "class"])
    shares = data[symptoms].eq("Yes").groupby(data["class"]).mean().T
    shares[["Negative", "Positive"]].plot.barh(figsize=(8, 7))
    plt.title("Share answering Yes to each symptom")
    plt.xlabel("Share of rows")
    plt.tight_layout()
    plt.savefig("doc/eda_results/symptom_shares.png")
    plt.close()
    print("Saved charts in doc/eda_results")
    print()
    print(data.head())


if __name__ == "__main__":
    main()
