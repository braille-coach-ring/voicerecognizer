from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG, PROJECT_ROOT
from voicerecognizer.evaluation.review import VALID_REVIEW_DECISIONS, load_review_decisions


def _top_counter(counter: Counter[str], limit: int) -> list[dict[str, int | str]]:
    return [
        {"name": name, "count": int(count)}
        for name, count in counter.most_common(limit)
        if count > 0
    ]


def _path_for_json(path: Path) -> str:
    return str(path).replace("\\", "/")


def _resolve_audio_path(path_value: str, *, index_base: Path, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    index_relative = index_base / path
    if index_relative.exists():
        return index_relative

    return project_root / path


def _infer_speaker(path: Path, *, label: str, raw_dataset_dir: Path) -> str:
    try:
        rel = path.resolve().relative_to(raw_dataset_dir.resolve())
        parts = rel.parts
        if not parts:
            return ""
        if parts[0] == "collected":
            return parts[1] if len(parts) >= 2 else "collected"
        return parts[0]
    except ValueError:
        pass

    if path.parent.name.startswith("pc_"):
        return path.parent.name
    if label and path.parent.name == label and path.parent.parent.name:
        return path.parent.parent.name
    return path.parent.name


def _read_index_rows(
    index_path: Path,
    *,
    project_root: Path,
    raw_dataset_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not index_path.exists():
        return rows

    with open(index_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = str(row.get("filepath") or row.get("\ufefffilepath") or "").strip()
            label = str(row.get("label") or "").strip()
            if not filepath or not label:
                continue

            resolved = _resolve_audio_path(
                filepath,
                index_base=index_path.parent,
                project_root=project_root,
            )
            speaker = str(row.get("speaker") or "").strip()
            if not speaker:
                speaker = _infer_speaker(resolved, label=label, raw_dataset_dir=raw_dataset_dir)

            rows.append(
                {
                    "filepath": filepath,
                    "resolved_filepath": str(resolved),
                    "label": label,
                    "speaker": speaker,
                    "predicted_text": str(row.get("predicted_text") or "").strip(),
                }
            )

    return rows


def _hash_file(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _find_audio_hash_duplicates(
    paths: list[Path],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_digest: dict[str, list[str]] = defaultdict(list)
    for path in sorted(set(paths), key=lambda item: str(item)):
        try:
            by_digest[_hash_file(path)].append(_path_for_json(path))
        except OSError:
            continue

    groups = [
        {"hash": digest, "count": len(items), "filepaths": items[:limit]}
        for digest, items in by_digest.items()
        if len(items) > 1
    ]
    groups.sort(key=lambda item: int(item["count"]), reverse=True)
    return groups[:limit]


def _audit_processed_dataset(
    processed_dataset_dir: Path,
    *,
    expected_existing_files: int,
) -> dict[str, Any]:
    wav_files = (
        sorted(processed_dataset_dir.rglob("*.wav")) if processed_dataset_dir.exists() else []
    )
    index_path = processed_dataset_dir / "index.csv"
    index_rows = 0
    if index_path.exists():
        with open(index_path, encoding="utf-8", newline="") as f:
            index_rows = sum(1 for _ in csv.DictReader(f))

    reasons: list[str] = []
    if not processed_dataset_dir.exists():
        reasons.append("processed_dataset_missing")
    if processed_dataset_dir.exists() and not index_path.exists():
        reasons.append("processed_index_missing")
    if expected_existing_files and len(wav_files) != expected_existing_files:
        reasons.append("processed_wav_count_mismatch")
    if index_path.exists() and index_rows != len(wav_files):
        reasons.append("processed_index_count_mismatch")

    return {
        "exists": processed_dataset_dir.exists(),
        "index_exists": index_path.exists(),
        "index_rows": index_rows,
        "wav_files": len(wav_files),
        "expected_wav_files": expected_existing_files,
        "is_stale_or_incomplete": bool(reasons),
        "reasons": reasons,
    }


def _audit_review_candidates(
    review_candidates_path: Path,
    *,
    review_decisions_path: Path | None,
) -> dict[str, Any]:
    if not review_candidates_path.exists():
        return {
            "exists": False,
            "total": 0,
            "unreviewed": 0,
            "decisions": {},
            "quality_flags": {},
            "top_unreviewed": [],
        }

    with open(review_candidates_path, encoding="utf-8") as f:
        payload = json.load(f)

    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    saved_decisions = load_review_decisions(review_decisions_path)
    decision_counter: Counter[str] = Counter()
    quality_counter: Counter[str] = Counter()
    top_unreviewed: list[dict[str, Any]] = []

    for item in candidates:
        if not isinstance(item, dict):
            continue
        filepath = str(item.get("filepath") or "").replace("\\", "/")
        decision = str(item.get("decision") or "").strip()
        if filepath in saved_decisions:
            decision = saved_decisions[filepath].decision

        if decision in VALID_REVIEW_DECISIONS:
            decision_counter[decision] += 1
        else:
            decision_counter["unreviewed"] += 1
            if len(top_unreviewed) < 20:
                top_unreviewed.append(
                    {
                        "filepath": filepath,
                        "true_label": item.get("true_label", ""),
                        "predicted_label": item.get("predicted_label", ""),
                        "confidence": item.get("confidence"),
                        "review_priority": item.get("review_priority", 0),
                        "quality_flags": item.get("quality_flags", []),
                    }
                )

        for flag in item.get("quality_flags", []):
            quality_counter[str(flag)] += 1

    total = sum(decision_counter.values())
    return {
        "exists": True,
        "total": total,
        "unreviewed": int(decision_counter["unreviewed"]),
        "decisions": dict(sorted(decision_counter.items())),
        "quality_flags": dict(quality_counter.most_common()),
        "top_unreviewed": top_unreviewed,
    }


def _audit_confusions(evaluation_result_path: Path, *, limit: int) -> dict[str, Any]:
    if not evaluation_result_path.exists():
        return {"exists": False, "top_pairs": []}

    with open(evaluation_result_path, encoding="utf-8") as f:
        payload = json.load(f)

    matrix = payload.get("confusion_matrix", {}) if isinstance(payload, dict) else {}
    pairs: list[dict[str, Any]] = []
    if isinstance(matrix, dict):
        for true_label, row in matrix.items():
            if not isinstance(row, dict):
                continue
            total = sum(int(value) for value in row.values())
            if total <= 0:
                continue
            for predicted_label, count_value in row.items():
                count = int(count_value)
                if predicted_label == true_label or count <= 0:
                    continue
                pairs.append(
                    {
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                        "rate": round(count / total, 4),
                    }
                )

    pairs.sort(key=lambda item: int(item["count"]), reverse=True)
    return {"exists": True, "top_pairs": pairs[:limit]}


def audit_dataset(
    *,
    merged_dataset_dir: Path | str = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
    processed_dataset_dir: Path | str = DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir,
    raw_dataset_dir: Path | str = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir,
    review_candidates_path: Path | str = PROJECT_ROOT
    / "evaluation_results"
    / "review_candidates.json",
    review_decisions_path: Path | str | None = PROJECT_ROOT
    / "evaluation_results"
    / "review_decisions.json",
    evaluation_result_path: Path | str = PROJECT_ROOT
    / "evaluation_results"
    / "evaluation_result.json",
    labels: tuple[str, ...] = DEFAULT_RECOGNITION_CONFIG.labels,
    min_label_count: int = 50,
    duplicate_hash_limit: int = 20,
    include_hash_duplicates: bool = True,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, Any]:
    merged_dataset_dir = Path(merged_dataset_dir)
    processed_dataset_dir = Path(processed_dataset_dir)
    raw_dataset_dir = Path(raw_dataset_dir)
    review_candidates_path = Path(review_candidates_path)
    review_decisions = Path(review_decisions_path) if review_decisions_path is not None else None
    evaluation_result_path = Path(evaluation_result_path)
    project_root = Path(project_root)

    index_path = merged_dataset_dir / "index.csv"
    rows = _read_index_rows(
        index_path,
        project_root=project_root,
        raw_dataset_dir=raw_dataset_dir,
    )

    existing_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    filepath_counter: Counter[str] = Counter()
    existing_path_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    missing_label_counter: Counter[str] = Counter()
    missing_speaker_counter: Counter[str] = Counter()
    invalid_label_counter: Counter[str] = Counter()

    for row in rows:
        filepath_counter[row["filepath"]] += 1
        label = row["label"]
        if label not in labels:
            invalid_label_counter[label] += 1

        resolved = Path(row["resolved_filepath"])
        if resolved.exists():
            existing_rows.append(row)
            existing_path_counter[str(resolved)] += 1
            label_counter[label] += 1
        else:
            missing_rows.append(row)
            missing_label_counter[label] += 1
            missing_speaker_counter[row["speaker"] or "unknown"] += 1

    duplicate_filepaths = [
        {"filepath": filepath, "count": int(count)}
        for filepath, count in filepath_counter.most_common()
        if count > 1
    ][:20]
    duplicate_sources = [
        {"filepath": filepath, "count": int(count)}
        for filepath, count in existing_path_counter.most_common()
        if count > 1
    ][:20]

    below_min = [
        {"label": label, "count": int(label_counter[label]), "needed": min_label_count}
        for label in labels
        if label_counter[label] < min_label_count
    ]
    max_count = max(label_counter.values()) if label_counter else 0
    min_count = min((label_counter[label] for label in labels), default=0)
    imbalance_ratio = round(max_count / min_count, 4) if min_count > 0 else None

    existing_paths = [Path(row["resolved_filepath"]) for row in existing_rows]
    audio_hash_duplicates = (
        _find_audio_hash_duplicates(existing_paths, limit=duplicate_hash_limit)
        if include_hash_duplicates
        else []
    )

    return {
        "paths": {
            "merged_index": _path_for_json(index_path),
            "processed_dataset": _path_for_json(processed_dataset_dir),
            "raw_dataset": _path_for_json(raw_dataset_dir),
            "review_candidates": _path_for_json(review_candidates_path),
            "review_decisions": _path_for_json(review_decisions)
            if review_decisions is not None
            else "",
            "evaluation_result": _path_for_json(evaluation_result_path),
        },
        "merged_index": {
            "exists": index_path.exists(),
            "rows": len(rows),
            "existing_files": len(existing_rows),
            "missing_files": len(missing_rows),
            "missing_by_label": _top_counter(missing_label_counter, 20),
            "missing_by_speaker": _top_counter(missing_speaker_counter, 20),
            "duplicate_filepaths": duplicate_filepaths,
            "duplicate_sources": duplicate_sources,
            "duplicate_audio_hashes": audio_hash_duplicates,
            "invalid_labels": _top_counter(invalid_label_counter, 20),
        },
        "label_distribution": {
            "min_required_per_label": min_label_count,
            "label_count": len(labels),
            "counts": {label: int(label_counter[label]) for label in labels},
            "below_min_count": below_min,
            "min_count": int(min_count),
            "max_count": int(max_count),
            "imbalance_ratio": imbalance_ratio,
        },
        "processed_dataset": _audit_processed_dataset(
            processed_dataset_dir,
            expected_existing_files=len(existing_rows),
        ),
        "review_candidates": _audit_review_candidates(
            review_candidates_path,
            review_decisions_path=review_decisions,
        ),
        "confusions": _audit_confusions(evaluation_result_path, limit=20),
    }


def write_audit_json(report: dict[str, Any], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def format_audit_summary(report: dict[str, Any]) -> str:
    merged = report["merged_index"]
    labels = report["label_distribution"]
    processed = report["processed_dataset"]
    review = report["review_candidates"]
    confusions = report["confusions"]

    lines = [
        "=== Dataset Audit ===",
        f"Index rows: {merged['rows']} "
        f"(existing={merged['existing_files']}, missing={merged['missing_files']})",
        f"Processed wavs: {processed['wav_files']} "
        f"(index rows={processed['index_rows']}, stale={processed['is_stale_or_incomplete']})",
        f"Labels below {labels['min_required_per_label']}: "
        f"{len(labels['below_min_count'])}/{labels['label_count']}",
        f"Review candidates: total={review['total']}, unreviewed={review['unreviewed']}",
    ]

    if merged["missing_by_label"]:
        top = ", ".join(
            f"{item['name']}={item['count']}" for item in merged["missing_by_label"][:8]
        )
        lines.append(f"Top missing labels: {top}")

    if labels["below_min_count"]:
        top = ", ".join(
            f"{item['label']}={item['count']}" for item in labels["below_min_count"][:12]
        )
        lines.append(f"Lowest labels: {top}")

    if merged["duplicate_filepaths"]:
        lines.append(f"Duplicate index filepaths: {len(merged['duplicate_filepaths'])} groups")
    if merged["duplicate_audio_hashes"]:
        lines.append(f"Duplicate audio hashes: {len(merged['duplicate_audio_hashes'])} groups")

    if processed["reasons"]:
        lines.append("Processed dataset flags: " + ", ".join(processed["reasons"]))

    if review["quality_flags"]:
        flags = ", ".join(f"{name}={count}" for name, count in review["quality_flags"].items())
        lines.append(f"Review quality flags: {flags}")

    if confusions["top_pairs"]:
        top = ", ".join(
            f"{item['true_label']}->{item['predicted_label']}:{item['count']}"
            for item in confusions["top_pairs"][:8]
        )
        lines.append(f"Top confusion pairs: {top}")

    return "\n".join(lines)
