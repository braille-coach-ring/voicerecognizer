from pathlib import Path

import librosa
import numpy as np
import torch

from model import HiraganaCNN
from dataset import HiraganaDataset

# ==========================
# モデル読み込み
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = HiraganaDataset()

labels = dataset.labels

model = HiraganaCNN(num_classes=len(labels))

model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.to(device)
model.eval()

# ==========================
# 評価
# ==========================

root = Path("processed_dataset")

total = 0
correct = 0

class_total = {}
class_correct = {}

for label in labels:
    class_total[label] = 0
    class_correct[label] = 0

print("=" * 50)

with torch.no_grad():
    for label in labels:
        folder = root / label

        for wav in sorted(folder.glob("*.wav")):
            y, sr = librosa.load(wav, sr=16000)

            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=400, hop_length=160, n_mels=64
            )

            mel = librosa.power_to_db(mel, ref=np.max)

            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

            mel = torch.tensor(mel, dtype=torch.float32)

            mel = mel.unsqueeze(0).unsqueeze(0)

            mel = mel.to(device)

            output = model(mel)

            prob = torch.softmax(output, dim=1)[0]

            pred = prob.argmax().item()

            pred_label = labels[pred]

            ok = pred_label == label

            total += 1
            class_total[label] += 1

            if ok:
                correct += 1
                class_correct[label] += 1

            mark = "〇" if ok else "×"

            print("=" * 50)
            print(f"{label}/{wav.name}")
            print()
            print(f"予測 : {pred_label}")
            print(f"正解 : {label}")
            print()

            for i, c in enumerate(labels):
                print(f"{c} : {prob[i].item() * 100:.2f}%")

            print()
            print(f"{mark} {'正解' if ok else '不正解'}")
            print()

            print()

            print("=" * 50)
            print("文字ごとの正答率")
            print("=" * 50)

for label in labels:
    acc = class_correct[label] / class_total[label] * 100

    print(f"{label} : {class_correct[label]}/{class_total[label]} ({acc:.2f}%)")

print()

print("=" * 50)
print("全体")
print("=" * 50)

print(f"{correct}/{total}")

print(f"Accuracy : {correct / total * 100:.2f}%")
