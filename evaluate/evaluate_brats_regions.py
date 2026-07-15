import torch
import numpy as np


def dice_score(y_true, y_pred):

    intersection = np.sum(y_true * y_pred)

    denominator = (
        np.sum(y_true)
        + np.sum(y_pred)
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection
    ) / denominator


def evaluate_brats_regions(
    model,
    loader
):

    device = next(
        model.parameters()
    ).device

    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():

        for graph in loader:

            graph = graph.to(device)

            logits = model(
                graph.x,
                graph.edge_index
            )

            pred = logits.argmax(
                dim=1
            )

            all_true.append(
                graph.y.cpu().numpy()
            )

            all_pred.append(
                pred.cpu().numpy()
            )

    y_true = np.concatenate(
        all_true
    )

    y_pred = np.concatenate(
        all_pred
    )

    # -----------------------
    # Whole Tumor (WT)
    # classes 1,2,3
    # -----------------------
    wt_true = (y_true > 0).astype(
        np.uint8
    )

    wt_pred = (y_pred > 0).astype(
        np.uint8
    )

    wt_dice = dice_score(
        wt_true,
        wt_pred
    )

    # -----------------------
    # Tumor Core (TC)
    # classes 1,3
    # -----------------------
    tc_true = np.isin(
        y_true,
        [1, 3]
    ).astype(np.uint8)

    tc_pred = np.isin(
        y_pred,
        [1, 3]
    ).astype(np.uint8)

    tc_dice = dice_score(
        tc_true,
        tc_pred
    )

    # -----------------------
    # Enhancing Tumor (ET)
    # class 3
    # -----------------------
    et_true = (y_true == 3).astype(
        np.uint8
    )

    et_pred = (y_pred == 3).astype(
        np.uint8
    )

    et_dice = dice_score(
        et_true,
        et_pred
    )

    print("\n===== BraTS Region Dice Scores =====")

    print(
        f"WT (Whole Tumor)      : {wt_dice:.4f}"
    )

    print(
        f"TC (Tumor Core)       : {tc_dice:.4f}"
    )

    print(
        f"ET (Enhancing Tumor)  : {et_dice:.4f}"
    )

    return {
        "WT": wt_dice,
        "TC": tc_dice,
        "ET": et_dice
    }