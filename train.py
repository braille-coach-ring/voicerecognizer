import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HiraganaDataset
from model import HiraganaCNN

# ==========================
# 設定
# ==========================

BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.001
SEED = 42

# ==========================
# 乱数固定
# ==========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================
# デバイス
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ==========================
# Dataset
# ==========================

dataset = HiraganaDataset()

num_classes = len(dataset.labels)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Train:", len(dataset))

# ==========================
# モデル
# ==========================

model = HiraganaCNN(num_classes=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================
# 学習履歴
# ==========================

train_losses = []
train_accs = []

# ==========================
# 学習開始
# ==========================

for epoch in range(EPOCHS):

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

        progress.set_description(
            f"Epoch {epoch+1}/{EPOCHS}"
        )

        progress.set_postfix(
            loss=f"{loss.item():.3f}"
        )

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    print()

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")

# ==========================
# モデル保存
# ==========================

torch.save(
    model.state_dict(),
    "best_model.pth"
)

torch.save(
    model.state_dict(),
    "last_model.pth"
)

print("\nModel Saved!")

# ==========================
# Lossグラフ
# ==========================

plt.figure(figsize=(8,5))

plt.plot(train_losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(True)

plt.savefig("loss.png")

plt.close()

# ==========================
# Accuracyグラフ
# ==========================

plt.figure(figsize=(8,5))

plt.plot(train_accs)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig("accuracy.png")

plt.close()

print("\nTraining Finished!")