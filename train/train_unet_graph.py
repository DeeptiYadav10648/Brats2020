import copy
import torch

from torch import amp
from torch.optim import AdamW

from losses.combined_loss import CombinedLoss
from models.unet_graph import UNetGraph


def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for batch in loader:

            image = batch["image"].to(device)

            label = batch["label"].to(device)

            if label.dim() == 5:
                label = label.squeeze(1)

            label = label.long()

            output = model(image)

            loss = criterion(
                output,
                label
            )

            pred = output.argmax(dim=1)

            correct += (
                pred == label
            ).sum().item()

            total += label.numel()

            total_loss += loss.item()

    return (

        correct / total,

        total_loss / len(loader)

    )


def train_unet_graph(

    train_loader,

    val_loader,

    epochs=200,

    lr=1e-4,

    patience=20

):

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print()

    print("Device :", device)

    print()

    model = UNetGraph().to(device)

    criterion = CombinedLoss()

    optimizer = AdamW(

        model.parameters(),

        lr=lr,

        weight_decay=1e-5

    )

    scaler = amp.GradScaler(

        "cuda",

        enabled=torch.cuda.is_available()

    )

    best_acc = 0

    best_model = None

    patience_counter = 0

    history = {

        "train_loss": [],

        "train_acc": [],

        "val_loss": [],

        "val_acc": []

    }

    for epoch in range(epochs):

        print()

        print("=" * 60)

        print(f"Epoch {epoch+1}/{epochs}")

        print("=" * 60)

        model.train()

        train_loss = 0

        train_correct = 0

        train_total = 0

        for batch_idx, batch in enumerate(train_loader):

            image = batch["image"].to(device)

            label = batch["label"].to(device)

            if label.dim() == 5:
                label = label.squeeze(1)

            label = label.long()

            optimizer.zero_grad()

            with amp.autocast(

                "cuda",

                enabled=torch.cuda.is_available()

            ):

                output = model(image)

                loss = criterion(

                    output,

                    label

                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            pred = output.argmax(dim=1)

            train_correct += (

                pred == label

            ).sum().item()

            train_total += label.numel()

            train_loss += loss.item()

            if batch_idx % 5 == 0:

                print(

                    f"Batch "

                    f"{batch_idx+1}/{len(train_loader)} "

                    f"Loss {loss.item():.4f}"

                )

        train_loss /= len(train_loader)

        train_acc = train_correct / train_total

        val_acc, val_loss = evaluate(

            model,

            val_loader,

            criterion,

            device

        )

        history["train_loss"].append(train_loss)

        history["train_acc"].append(train_acc)

        history["val_loss"].append(val_loss)

        history["val_acc"].append(val_acc)

        print()

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Train Acc  : {train_acc:.4f}")

        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Val Acc    : {val_acc:.4f}")

        if val_acc > best_acc:

            best_acc = val_acc

            best_model = copy.deepcopy(

                model.state_dict()

            )

            torch.save(

                best_model,

                "best_unet_graph.pth"

            )

            patience_counter = 0

            print()

            print("✓ Best model updated")

        else:

            patience_counter += 1

            print(

                f"EarlyStopping "

                f"{patience_counter}/{patience}"

            )

        if patience_counter >= patience:

            print()

            print("Early stopping")

            break

    print()

    print("=" * 60)

    print("Training Finished")

    print("=" * 60)

    print()

    print(f"Best Validation Accuracy : {best_acc:.4f}")

    if best_model is not None:

        model.load_state_dict(best_model)

    return model, history