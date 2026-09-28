"""Step 4: measure predictions on data the chosen model never saw."""

from pathlib import Path
Path("doc/evaluation_result").mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss, recall_score

def scores(y_true, probability):
    # Recall and specificity use the same 0.5 cutoff as the confusion matrix.
    probability = np.asarray(probability)
    predicted = probability >= 0.5
    return {
        "Recall": recall_score(y_true, predicted, pos_label=1, zero_division=0),
        "Specificity": recall_score(y_true, predicted, pos_label=0, zero_division=0),
        "Brier": brier_score_loss(y_true, probability),
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
