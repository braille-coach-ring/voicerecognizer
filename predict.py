from pathlib import Path
import sys

import librosa
import numpy as np
import torch

from model import HiraganaCNN
from dataset import HiraganaDataset

# ==========================
# モデル読み込み
# ==========================

dataset = HiraganaDataset()

labels = dataset.labels

model = HiraganaCNN(num_classes=len(labels))

model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))

model.eval()

# ==========================
# 引数確認
# ==========================

if len(sys.argv) != 2:
    print("使い方")
    print("python predict.py ファイル名.wav")
    exit()

wav_path = Path(sys.argv[1])

if not wav_path.exists():
    print("ファイルがありません")
    exit()

# ==========================
# 前処理
# ==========================

y, sr = librosa.load(wav_path, sr=16000)

mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=64)

mel = librosa.power_to_db(mel, ref=np.max)

# 学習時と同じ正規化
mel = (mel - mel.mean()) / (mel.std() + 1e-8)

mel = torch.tensor(mel, dtype=torch.float32)

mel = mel.unsqueeze(0)
mel = mel.unsqueeze(0)

# ==========================
# 推論
# ==========================

with torch.no_grad():
    output = model(mel)

    prob = torch.softmax(output, dim=1)

    pred = torch.argmax(prob, dim=1).item()

print("=" * 30)
print()

print("予測:", labels[pred])

print()

print("確率")

for i, p in enumerate(prob[0]):
    print(f"{labels[i]} : {p.item() * 100:.2f}%")

print()
print("=" * 30)
