import numpy as np
import librosa
import sounddevice as sd
import torch
from time import sleep

from model import HiraganaCNN

# ==========================
# 設定
# ==========================

MODEL_PATH = "best_model.pth"

SR = 16000
RECORD_SECONDS = 1.0
TARGET_LENGTH = 1.0
TOP_DB = 20
N_MELS = 64

LABELS = sorted(["a", "e", "i", "o", "u"])

# ==========================
# Device
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# モデル読み込み
# ==========================

model = HiraganaCNN(num_classes=len(LABELS))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

# ==========================
# 録音
# ==========================

print("3秒後に1秒間録音します...")
sleep(1)
print("3")
sleep(1)
print("2")
sleep(1)
print("1")
sleep(1)

print("発声してください。")

audio = sd.rec(int(RECORD_SECONDS * SR), samplerate=SR, channels=1, dtype="float32")

sd.wait()

print("録音終了")

# (16000,1) → (16000,)
y = audio.flatten()

# ==========================
# 前処理
# ==========================

y, _ = librosa.effects.trim(y, top_db=TOP_DB)

if np.max(np.abs(y)) > 0:
    y = y / np.max(np.abs(y))

target_samples = int(TARGET_LENGTH * SR)

if len(y) > target_samples:
    y = y[:target_samples]
else:
    y = np.pad(y, (0, target_samples - len(y)))

# ==========================
# メルスペクトログラム
# ==========================

mel = librosa.feature.melspectrogram(
    y=y, sr=SR, n_fft=400, hop_length=160, n_mels=N_MELS
)

mel = librosa.power_to_db(mel, ref=np.max)

mel = (mel - mel.mean()) / (mel.std() + 1e-8)

mel = torch.tensor(mel, dtype=torch.float32)

# (64,101) → (1,64,101)
mel = mel.unsqueeze(0)

# (1,64,101) → (1,1,64,101)
mel = mel.unsqueeze(0)

mel = mel.to(device)

# ==========================
# 推論
# ==========================

with torch.no_grad():
    output = model(mel)

    probs = torch.softmax(output, dim=1)[0]

    pred = torch.argmax(probs).item()

# ==========================
# 結果表示
# ==========================

print("\n==============================")
print("認識結果")
print("==============================")

print(f"\n予測: {LABELS[pred]}\n")

for label, prob in zip(LABELS, probs):
    print(f"{label} : {prob.item() * 100:.2f}%")
