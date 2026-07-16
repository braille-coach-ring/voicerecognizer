import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import HiraganaDataset
from model import HiraganaCNN

# ==========================
# 設定
# ==========================

BATCH_SIZE = 8
EPOCHS = 150
LEARNING_RATE = 0.001
VAL_RATE = 0.2
TARGET_ACC = 0.97
SEED = 42

# ==========================
# 乱数固定
# ==========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================
# Device
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# ==========================
# Dataset
# ==========================

dataset = HiraganaDataset()

num_classes = len(dataset.labels)

labels = [label for _, label in dataset.data]

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATE, random_state=SEED)

train_idx, val_idx = next(sss.split(range(len(labels)), labels))

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train      :", len(train_dataset))
print("Validation :", len(val_dataset))

# ==========================
# Model
# ==========================

model = HiraganaCNN(num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================
# 学習履歴
# ==========================

train_losses = []
val_losses = []

train_accs = []
val_accs = []

best_acc = 0.0

# ==========================
# Training
# ==========================

for epoch in range(EPOCHS):
    # ----------------------
    # Train
    # ----------------------

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    progress = tqdm(train_loader)

    for mel, label in progress:
        mel = mel.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(mel)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        pred = output.argmax(dim=1)

        train_correct += (pred == label).sum().item()
        train_total += label.size(0)

        progress.set_description(f"Epoch {epoch + 1}/{EPOCHS}")

        progress.set_postfix(loss=f"{loss.item():.3f}")

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # ----------------------
    # Validation
    # ----------------------

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for mel, label in val_loader:
            mel = mel.to(device)
            label = label.to(device)

            output = model(mel)

            loss = criterion(output, label)

            val_loss += loss.item()

            pred = output.argmax(dim=1)

            val_correct += (pred == label).sum().item()
            val_total += label.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    # ----------------------
    # 保存
    # ----------------------

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print()

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")

    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.4f}")

    # ----------------------
    # ベストモデル保存
    # ----------------------

    if val_acc > best_acc:
        best_acc = val_acc

        torch.save(model.state_dict(), "best_model.pth")

        print("Best model saved!")

    # ----------------------
    # 終了条件
    # ----------------------

    if val_acc >= TARGET_ACC:
        print()
        print(f"Validation Accuracy {TARGET_ACC * 100:.0f}% 到達")
        break

# ==========================
# 最終モデル保存
# ==========================

torch.save(model.state_dict(), "last_model.pth")

print("\nModel Saved!")

# ==========================
# Loss
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig("loss.png")
plt.close()

# ==========================
# Accuracy
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(train_accs, label="Train")
plt.plot(val_accs, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig("accuracy.png")
plt.close()

print("\nTraining Finished!")
