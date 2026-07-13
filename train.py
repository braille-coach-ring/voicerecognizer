import random

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import HiraganaDataset
from model import HiraganaCNN

# ==========================
# 設定
# ==========================

BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.001
VAL_RATE = 0.2
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

labels = []

for _, label in dataset.data:
    labels.append(label)

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=VAL_RATE,
    random_state=SEED
)

train_idx, val_idx = next(
    sss.split(
        range(len(labels)),
        labels
    )
)

train_dataset = dataset

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)


print("Train:", len(train_dataset))


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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

best_loss = float("inf")
patience = 8
counter = 0

train_losses = []
val_losses = []

train_accs = []
val_accs = []
# ==========================
# 学習開始
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

        progress.set_description(
            f"Epoch {epoch+1}/{EPOCHS}"
        )

        progress.set_postfix(
            loss=f"{loss.item():.3f}"
        )

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # ----------------------
    # Validation
    # ----------------------

    
    train_losses.append(train_loss)
    

    train_accs.append(train_acc)
    

    print()

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")


    # ----------------------
    # ベストモデル保存
    # ----------------------

    
        
    torch.save(
                model.state_dict(),
                "best_model.pth"
            )

    counter = 0

    print("Best model saved!")
    
    # ----------------------
    # EarlyStopping
    # ----------------------

    

torch.save(
    model.state_dict(),
    "last_model.pth"
)

print()

# ==========================
# Lossグラフ
# ==========================

plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("loss.png")

plt.close()

# ==========================
# Accuracyグラフ
# ==========================

plt.figure(figsize=(8,5))

plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("accuracy.png")

plt.close()

print("Training Finished!")