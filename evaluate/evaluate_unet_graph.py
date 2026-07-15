import time
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


def dice_score(pred, target, cls):
    pred = (pred == cls)
    target = (target == cls)

    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()

    if union == 0:
        return 1.0

    return (2.0 * intersection) / (union + 1e-8)


def region_dice(pred, target, region):
    pred = region(pred)
    target = region(target)

    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()

    if union == 0:
        return 1.0

    return (2.0 * intersection) / (union + 1e-8)


def evaluate_unet_graph(
    model,
    loader,
    device="cuda",
    max_batches=10
):

    if isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    print("=" * 60)
    print("Starting Evaluation")
    print("=" * 60)
    print(f"Device : {device}")
    print(f"Evaluating first {max_batches} batches only")
    print()

    preds = []
    labels = []

    total_start = time.time()

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            if batch_idx >= max_batches:
                break

            batch_start = time.time()

            print(f"Batch {batch_idx + 1}/{max_batches}", end=" ... ")

            image = batch["image"].to(device)
            label = batch["label"].to(device)

            if label.dim() == 5:
                label = label.squeeze(1)

            output = model(image)

            prediction = output.argmax(dim=1)

            preds.extend(
                prediction.cpu().numpy().reshape(-1)
            )

            labels.extend(
                label.cpu().numpy().reshape(-1)
            )

            print(f"{time.time() - batch_start:.2f} sec")

            # Free memory
            del image
            del label
            del output
            del prediction

    total_time = time.time() - total_start

    print()
    print("=" * 60)
    print(f"Evaluation completed in {total_time:.2f} seconds")
    print("=" * 60)

    preds = np.array(preds)
    labels = np.array(labels)

    acc = accuracy_score(labels, preds)

    prec = precision_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    rec = recall_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(labels, preds)

    report = classification_report(
        labels,
        preds,
        zero_division=0
    )

    print("\n===== Evaluation Results =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:\n")
    print(cm)

    print("\nClassification Report:\n")
    print(report)

    print("\n===== Dice Scores =====")

    for cls in range(4):
        d = dice_score(preds, labels, cls)
        print(f"Class {cls}: {d:.4f}")

    wt = region_dice(preds, labels, lambda x: x > 0)

    tc = region_dice(
        preds,
        labels,
        lambda x: np.logical_or(x == 1, x == 3)
    )

    et = region_dice(
        preds,
        labels,
        lambda x: x == 3
    )

    print("\n===== BraTS Region Dice Scores =====")
    print(f"WT (Whole Tumor)      : {wt:.4f}")
    print(f"TC (Tumor Core)       : {tc:.4f}")
    print(f"ET (Enhancing Tumor)  : {et:.4f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "wt": wt,
        "tc": tc,
        "et": et
    }