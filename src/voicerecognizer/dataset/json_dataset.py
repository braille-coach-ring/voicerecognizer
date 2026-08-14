import json
from pathlib import Path

from torch.utils.data import Dataset


class JsonSpeechDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.items = self._load_manifest()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]

    def _load_manifest(self) -> list[dict]:
        with self.manifest_path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
