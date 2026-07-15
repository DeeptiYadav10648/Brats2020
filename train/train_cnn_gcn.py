import copy
from collections import Counter

from losses.focal_loss import FocalLoss
from losses.dice_loss import DiceLoss

import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.cnn_gcn import HybridGCN
from models.gcn import GCN


def compute_class_weights(loader, device):

    all_labels = []

    for graph in loader:

        all_labels.extend(
            graph.y.cpu().numpy()
        )

    counts = Counter(all_labels)

    print("\nTraining Class Distribution:")
    print(counts)

    total = sum(counts.values())
    num_classes = 4

    weights = []

    for cls in range(num_classes):

        if cls in counts:

            weights.append(
                total / (num_classes * counts[cls])
            )

        else:

            weights.append(1.0)

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
    device,
    criterion
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

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

            correct += (
                pred == graph.y
            ).sum().item()

            total += graph.y.size(0)

            total_loss += loss.item()

    acc = correct / total

    return acc, total_loss / len(loader)


def train_cnn_gcn(
    train_loader,
    val_loader,
    epochs=200,
    lr=5e-4,
    hidden=128,
    patience=20
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    in_channels = train_loader.dataset[0].x.shape[1]

    model = HybridGCN(
        in_channels=in_channels,
        hidden=128,
        num_classes=4
    )

    optimizer = Adam(
        model.parameters(),
        lr=lr
    )

    class_weights = torch.tensor([
        1.0,   # background
        8.0,   # class 1
        3.0,   # class 2
        6.0    # class 3
    ]).to(device)

    focal_loss = FocalLoss(
    alpha=class_weights,
    gamma=2
)

    dice_loss = DiceLoss()

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

            loss = (
                0.5 * focal_loss(out, graph.y)
                +
                0.5 * dice_loss(out, graph.y)
            )

            loss.backward()

            optimizer.step()

            pred = out.argmax(dim=1)

            train_correct += (
                pred == graph.y
            ).sum().item()

            train_total += graph.y.size(0)

            train_loss += loss.item()

        train_acc = train_correct / train_total

        def combined_loss(out, y):

            return (
                0.5 * focal_loss(out, y)
                +
                0.5 * dice_loss(out, y)
            )


        val_acc, val_loss = evaluate(
            model,
            val_loader,
            device,
            combined_loss
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
        "cnn+gcn_model_test.pth"
    )

    print("\nTraining Complete")
    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.4f}"
    )

    return model