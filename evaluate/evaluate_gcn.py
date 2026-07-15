import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def dice_score(
    y_true,
    y_pred,
    num_classes=4
):

    scores = []

    for c in range(num_classes):

        yt = (y_true == c)
        yp = (y_pred == c)

        inter = (yt & yp).sum()

        dice = (
            2 * inter
        ) / (
            yt.sum() + yp.sum() + 1e-8
        )

        scores.append(dice)

    return scores


def evaluate_gcn(model, graph):

    device = next(model.parameters()).device

    graph = graph.to(device)

    model.eval()

    with torch.no_grad():

        logits = model(
            graph.x,
            graph.edge_index
        )

        pred = logits.argmax(dim=1)

    y_true = graph.y.cpu().numpy()
    y_pred = pred.cpu().numpy()

    print("\n===== Evaluation Results =====")

    print(
        "Accuracy  :",
        accuracy_score(y_true, y_pred)
    )

    print(
        "Precision :",
        precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    print(
        "Recall    :",
        recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    print(
        "F1 Score  :",
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )
    scores = dice_score(
    y_true,
    y_pred
    )

    print("\n===== Dice Scores =====")

    for i, score in enumerate(scores):

        print(
            f"Class {i}: {score:.4f}"
        )