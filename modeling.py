"""Step 3: compare three models with five training folds, then test the winner."""

from pathlib import Path
Path("doc/model_results").mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from cleaning import load_data
from evaluation import evaluate_test, scores


def main():
    X, y = load_data()

    # Keep 20% aside before comparing models, so it cannot influence the choice.
    # stratify=y keeps about the same Positive share in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # These models learn in different ways: a line, many trees, and boosted trees.
    models = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                random_state=42),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=100,
                                                             max_leaf_nodes=7,
                                                             learning_rate=0.05,
                                                             random_state=42),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    # Every model gets the same five train/validation splits.
    # iloc selects rows by their position in each split.
    for name, model in models.items():
        fold_scores = []
        for train_index, valid_index in folds.split(X_train, y_train):
            model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
            # Column 1 is the model's predicted chance of Positive.
            chance = model.predict_proba(X_train.iloc[valid_index])[:, 1]
            fold_scores.append(scores(y_train.iloc[valid_index], chance))
        # Average the five scores to compare models fairly.
        rows.append({"Model": name, **{
            metric: np.mean([fold[metric] for fold in fold_scores])
            for metric in fold_scores[0]
        }})

    # Lower RMSE means predicted chances were closer to the true 0/1 labels.
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    print("\nFive-fold training comparison (mean scores):")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    winner = table.loc[0, "Model"]
    print(f"\nChosen by lowest cross-validation RMSE: {winner}")

    # Train the winner on all training rows and check the held-out test rows once.
    best_model = models[winner]
    best_model.fit(X_train, y_train)
    test = evaluate_test(best_model, X_test, y_test)
    print("Test scores:", ", ".join(f"{key}={value:.3f}" for key, value in test.items()))

    table.plot.bar(x="Model", y=["MAE", "RMSE"], rot=0, figsize=(8, 4))
    plt.title("Five-fold mean errors (lower is better)")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig("doc/model_results/model_comparison.png")
    plt.close()

    # Save exact numbers so the result can be read without running Python.
    table.to_csv("doc/model_results/cross_validation_results.csv", index=False, float_format="%.4f")
    with open("doc/model_results/model_winner.txt", "w") as file:
        file.write("Five-fold training means (lower MAE/RMSE, higher R2/Accuracy):\n")
        file.write(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        file.write(f"\n\nWinner by lowest CV RMSE: {winner}\n")
        file.write("Held-out test scores: " + ", ".join(
            f"{key}={value:.3f}" for key, value in test.items()) + "\n")
        file.write("The target is Positive=1, Negative=0; MAE/RMSE/R2 score probabilities.\n")


if __name__ == "__main__":
    main()
