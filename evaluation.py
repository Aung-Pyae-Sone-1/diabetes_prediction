"""Step 4: measure predictions on data the chosen model never saw."""

from pathlib import Path
Path("doc/evaluation_result").mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

def scores(y_true, probability):
    # Compare the true 0/1 label with the predicted chance of Positive.
    return {
        "MAE": mean_absolute_error(y_true, probability),
        "RMSE": np.sqrt(mean_squared_error(y_true, probability)),
        "R2": r2_score(y_true, probability),
        "Accuracy": accuracy_score(y_true, probability >= 0.5),
    }


def evaluate_test(model, X_test, y_test):
    probability = model.predict_proba(X_test)[:, 1]
    result = scores(y_test, probability)

    # Count correct and incorrect decisions at a 0.5 cutoff.
    actual = np.asarray(y_test)
    predicted = probability >= 0.5
    counts = np.array([
        [np.sum((actual == 0) & ~predicted), np.sum((actual == 0) & predicted)],
        [np.sum((actual == 1) & ~predicted), np.sum((actual == 1) & predicted)],
    ])
    
    fig, ax = plt.subplots()
    ax.imshow(counts, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(counts[row, col]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
           yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Actual",
           title="Test predictions (cutoff: 0.5)")
    fig.tight_layout()
    fig.savefig("doc/evaluation_result/test_predictions.png")
    plt.close(fig)
    return result
