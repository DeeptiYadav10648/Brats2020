import copy
from collections import Counter

import torch

from torch.optim import Adam

from losses.focal_loss import FocalLoss
from losses.dice_loss import DiceLoss

from models.graphsage import GraphSAGE


def compute_class_weights(
    loader,
    device
):

    all_labels = []

    for graph in loader:

        all_labels.extend(
            graph.y.cpu().numpy()
        )

    counts = Counter(all_labels)

    print("\nTraining Class Distribution:")
    print(counts)

    total = sum(counts.values())

    weights = []

    for cls in range(4):

        weights.append(
            total / (4 * counts[cls])
        )

    weights = torch.tensor(
        weights,
        dtype=torch.float
    ).to(device)

    print("\nClass Weights:")
    print(weights)

    return weights


def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_nodes = 0

    with torch.no_grad():

        for graph in loader:

            graph = graph.to(device)

            out = model(
                graph.x,
                graph.edge_index
            )

            loss = criterion(
                out,
                graph.y
            )

            pred = out.argmax(dim=1)

            total_correct += (
                pred == graph.y
            ).sum().item()

            total_nodes += graph.y.size(0)

            total_loss += loss.item()

    acc = total_correct / total_nodes

    return (
        acc,
        total_loss / len(loader)
    )


def train_graphsage(
    train_loader,
    val_loader,
    epochs=300,
    lr=1e-3,
    hidden=128,
    patience=30
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    in_channels = train_loader.dataset[0].x.shape[1]

    model = GraphSAGE(
        in_channels=in_channels,
        hidden=hidden,
        num_classes=4
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    class_weights = compute_class_weights(
        train_loader,
        device
    )

    focal_loss = FocalLoss(
        alpha=class_weights,
        gamma=2
    )

    dice_loss = DiceLoss()

    def criterion(
        pred,
        target
    ):

        return (
            0.5 * focal_loss(pred, target)
            +
            0.5 * dice_loss(pred, target)
        )

    best_val_acc = 0
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for graph in train_loader:

            graph = graph.to(device)

            optimizer.zero_grad()

            out = model(
                graph.x,
                graph.edge_index
            )

            loss = criterion(
                out,
                graph.y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            pred = out.argmax(dim=1)

            train_correct += (
                pred == graph.y
            ).sum().item()

            train_total += graph.y.size(0)

            train_loss += loss.item()

        train_acc = train_correct / train_total

        val_acc, val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_model = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss/len(train_loader):.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        if patience_counter >= patience:

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break

    if best_model is not None:

        model.load_state_dict(
            best_model
        )

    torch.save(
        model.state_dict(),
        "best_graphsage_model.pth"
    )

    print("\nTraining Complete")
    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.4f}"
    )

    return model