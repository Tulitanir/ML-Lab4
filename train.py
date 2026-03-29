import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    CKPT_DIR,
    DATASET_DIR,
    DEVICE,
    EPOCHS,
    IMG_SIZE,
    LR,
    NUM_CLASSES,
    VERSION,
)
from model import EfficientNet


def get_transforms(train: bool) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(IMG_SIZE, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_data():
    train_ds = datasets.ImageFolder(
        DATASET_DIR / "train", transform=get_transforms(True)
    )
    val_ds = datasets.ImageFolder(DATASET_DIR / "validation", transform=get_transforms(False))
    test_ds = datasets.ImageFolder(
        DATASET_DIR / "test", transform=get_transforms(False)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(
            device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")
        ):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def plot_results(history: dict, save_path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


def train() -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = load_data()

    print(f"Device: {DEVICE}")
    print(
        f"EfficientNet-{VERSION.upper()}, {NUM_CLASSES} classes, {IMG_SIZE}×{IMG_SIZE} images"
    )

    model = EfficientNet(version=VERSION, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        v_loss, v_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        flag = ""
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "version": VERSION,
                    "num_classes": NUM_CLASSES,
                },
                CKPT_DIR / "best.pt",
            )
            flag = "  ← best"

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train loss {t_loss:.4f} acc {t_acc:.4f} | "
            f"val loss {v_loss:.4f} acc {v_acc:.4f}{flag}"
        )

    torch.save(
        {
            "epoch": EPOCHS,
            "model": model.state_dict(),
            "version": VERSION,
            "num_classes": NUM_CLASSES,
        },
        CKPT_DIR / "last.pt",
    )

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest loss {test_loss:.4f} acc {test_acc:.4f}")
    print(f"Best val accuracy: {best_acc:.4f}")

    plot_results(history, CKPT_DIR / "training_curves.png")


@torch.no_grad()
def infer(image_path: str, checkpoint: str) -> None:
    from PIL import Image

    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model = EfficientNet(version=ckpt["version"], num_classes=ckpt["num_classes"]).to(
        DEVICE
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = get_transforms(train=False)(img).unsqueeze(0).to(DEVICE)

    classes = sorted(p.name for p in (DATASET_DIR / "train").iterdir() if p.is_dir())

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze()
    top5 = probs.topk(5)

    def class_name(idx: int) -> str:
        return classes[idx] if idx < len(classes) else str(idx)

    print(f"\nImage: {image_path}")
    print(f"Prediction: {class_name(top5.indices[0].item())}  ({top5.values[0]*100:.1f}%)")
    print("Top-5:")
    for prob, idx in zip(top5.values, top5.indices):
        print(f"  {class_name(idx.item()):<30s}  {prob*100:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_DIR / "best.pt"))
    args = parser.parse_args()

    if args.infer:
        infer(args.infer, args.checkpoint)
    else:
        train()
