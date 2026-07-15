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


def compute_dice_scores(
    y_true,
    y_pred,
    num_classes=4
):

    dice_scores = []

    for cls in range(num_classes):

        true_mask = (y_true == cls)
        pred_mask = (y_pred == cls)

        intersection = np.logical_and(
            true_mask,
            pred_mask
        ).sum()

        denominator = (
            true_mask.sum()
            +
            pred_mask.sum()
        )

        if denominator == 0:

            dice = 1.0

        else:

            dice = (
                2.0 * intersection
            ) / denominator

        dice_scores.append(dice)

    return dice_scores


def evaluate_isgnn(
    model,
    test_loader
):

    device = next(
        model.parameters()
    ).device

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for graph in test_loader:

            graph = graph.to(device)

            logits = model(
                graph.x,
                graph.edge_index
            )

            preds = logits.argmax(
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                graph.y.cpu().numpy()
            )

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_preds
    )

    print("\n===== Evaluation Results =====")

    print(
        f"Accuracy  : "
        f"{accuracy_score(y_true, y_pred):.4f}"
    )

    print(
        f"Precision : "
        f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )

    print(
        f"Recall    : "
        f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )

    print(
        f"F1 Score  : "
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )

    print("\nConfusion Matrix:\n")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    dice_scores = compute_dice_scores(
        y_true,
        y_pred
    )

    print("\n===== Dice Scores =====")

    for i, score in enumerate(
        dice_scores
    ):

        print(
            f"Class {i}: "
            f"{score:.4f}"
        )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "dice_scores": dice_scores
    }