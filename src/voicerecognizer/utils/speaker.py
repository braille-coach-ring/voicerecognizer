from pathlib import Path

UNKNOWN_SPEAKER_ID = "unknown"


def infer_speaker_id_from_path(
    filepath: str | Path,
    *,
    default: str = UNKNOWN_SPEAKER_ID,
) -> str:
    path_text = str(filepath).replace("\\", "/")
    parts = [part for part in path_text.split("/") if part]

    for index, part in enumerate(parts):
        if part != "dataset":
            continue

        if index + 1 >= len(parts):
            break

        source = parts[index + 1]
        if source == "collected":
            if index + 2 < len(parts):
                return parts[index + 2]
            break

        return source

    return default


def normalize_speaker_id(
    speaker_id: str | None,
    filepath: str | Path,
    *,
    default: str = UNKNOWN_SPEAKER_ID,
) -> str:
    cleaned = (speaker_id or "").strip()
    if cleaned:
        return cleaned
    return infer_speaker_id_from_path(filepath, default=default)
