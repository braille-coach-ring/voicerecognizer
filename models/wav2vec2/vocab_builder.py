import json
from pathlib import Path


def build_vocab(texts: list[str], output_path: str | Path) -> dict[str, int]:
    characters = sorted(set("".join(texts)))
    vocab = {character: index for index, character in enumerate(characters)}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab
