import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.hiragana_dataset import HiraganaDataset
from models.cnn.hiragana_cnn import HiraganaCNN


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_dataset(dataset: HiraganaDataset, val_rate: float, seed: int):
    labels = [label for _, label in dataset.data]
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_rate,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(range(len(labels)), labels))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_epoch(model, loader, criterion, optimizer, device, epoch: int, epochs: int):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader)

    for mel, label in progress:
        mel = mel.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

        progress.set_description(f"Epoch {epoch + 1}/{epochs}")
        progress.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for mel, label in loader:
            mel = mel.to(device)
            label = label.to(device)
            output = model(mel)
            loss = criterion(output, label)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

    return total_loss / len(loader), correct / total


def save_plots(history: dict[str, list[float]], loss_path: Path, accuracy_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(accuracy_path)
    plt.close()


def train(args: argparse.Namespace) -> None:
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HiraganaDataset(
        root_dir=args.root_dir,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
    )
    train_dataset, val_dataset = split_dataset(dataset, args.val_rate, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("Device:", device)
    print("Train      :", len(train_dataset))
    print("Validation :", len(val_dataset))

    model = HiraganaCNN(num_classes=len(dataset.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    args.best_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.last_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.epochs,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print()
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.best_model_path)
            print("Best model saved.")

        if val_acc >= args.target_acc:
            print(f"Target validation accuracy reached: {args.target_acc:.2%}")
            break

    torch.save(model.state_dict(), args.last_model_path)
    save_plots(history, args.loss_plot_path, args.accuracy_plot_path)
    print("Training finished.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Hiragana CNN recognizer.")
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_AUDIO_CONFIG.sample_rate)
    parser.add_argument("--n-mels", type=int, default=DEFAULT_PREPROCESS_CONFIG.n_mels)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--val-rate", type=float, default=0.2)
    parser.add_argument("--target-acc", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--best-model-path", type=Path, default=DEFAULT_RECOGNITION_CONFIG.cnn_weight_path)
    parser.add_argument("--last-model-path", type=Path, default=DEFAULT_RECOGNITION_CONFIG.last_model_path)
    parser.add_argument("--loss-plot-path", type=Path, default=DEFAULT_RECOGNITION_CONFIG.loss_plot_path)
    parser.add_argument("--accuracy-plot-path", type=Path, default=DEFAULT_RECOGNITION_CONFIG.accuracy_plot_path)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
