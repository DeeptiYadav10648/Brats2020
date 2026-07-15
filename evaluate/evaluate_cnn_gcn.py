import numpy as np
import torch

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
    y_pred
):

    intersection = np.sum(
        y_true * y_pred
    )

    denominator = (
        np.sum(y_true)
        + np.sum(y_pred)
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection
    ) / denominator


def evaluate_cnn_gcn(
    model,
    loader
):

    device = next(
        model.parameters()
    ).device

    model.eval()

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():

        for graph in loader:

            graph = graph.to(device)

            logits = model(
                graph.x,
                graph.edge_index
            )

            pred = logits.argmax(dim=1)

            y_true_all.extend(
                graph.y.cpu().numpy()
            )

            y_pred_all.extend(
                pred.cpu().numpy()
            )

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

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

    print("\n===== Dice Scores =====")

    for cls in range(4):

        score = dice_score(
            (y_true == cls).astype(int),
            (y_pred == cls).astype(int)
        )

        print(
            f"Class {cls}: {score:.4f}"
        )

    WT_true = (y_true > 0).astype(int)
    WT_pred = (y_pred > 0).astype(int)

    TC_true = np.isin(
        y_true,
        [1, 3]
    ).astype(int)

    TC_pred = np.isin(
        y_pred,
        [1, 3]
    ).astype(int)

    ET_true = (y_true == 3).astype(int)
    ET_pred = (y_pred == 3).astype(int)

    print("\n===== BraTS Region Dice Scores =====")

    print(
        f"WT (Whole Tumor)      : "
        f"{dice_score(WT_true, WT_pred):.4f}"
    )

    print(
        f"TC (Tumor Core)       : "
        f"{dice_score(TC_true, TC_pred):.4f}"
    )

    print(
        f"ET (Enhancing Tumor)  : "
        f"{dice_score(ET_true, ET_pred):.4f}"
    )