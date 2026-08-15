from __future__ import annotations

import html
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

ReviewDecisionValue = Literal["keep", "delete_candidate", "maybe"]

VALID_REVIEW_DECISIONS: tuple[ReviewDecisionValue, ...] = (
    "keep",
    "delete_candidate",
    "maybe",
)


@dataclass(frozen=True)
class ReviewPriorityConfig:
    low_confidence_threshold: float = 0.60
    very_low_confidence_threshold: float = 0.40
    high_confidence_mismatch_threshold: float = 0.85
    min_speech_duration_ms: float = 120.0
    late_onset_ms: float = 350.0


DEFAULT_REVIEW_PRIORITY_CONFIG = ReviewPriorityConfig()


@dataclass(frozen=True)
class PredictionCandidate:
    label: str
    confidence: float


@dataclass(frozen=True)
class ReviewDecision:
    filepath: str
    label: str
    prediction: str
    confidence: float | None
    decision: ReviewDecisionValue
    decided_at: str = ""


@dataclass(frozen=True)
class ReviewCandidate:
    filepath: str
    true_label: str
    predicted_label: str
    confidence: float | None
    review_priority: float
    decision: str = ""
    top_candidates: list[PredictionCandidate] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    onset_ms: float | None = None
    offset_ms: float | None = None
    speech_duration_ms: float | None = None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_filepath(filepath: str) -> str:
    return filepath.replace("\\", "/")


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_quality_flags(
    *,
    onset_ms: float | None,
    speech_duration_ms: float | None,
    config: ReviewPriorityConfig = DEFAULT_REVIEW_PRIORITY_CONFIG,
) -> list[str]:
    flags: list[str] = []
    if speech_duration_ms is not None:
        if speech_duration_ms <= 0:
            flags.append("no_speech_detected")
        elif speech_duration_ms < config.min_speech_duration_ms:
            flags.append("short_speech")

    if onset_ms is not None and onset_ms > config.late_onset_ms:
        flags.append("late_onset")

    return flags


def compute_review_priority(
    *,
    true_label: str,
    predicted_label: str,
    confidence: float | None,
    quality_flags: list[str],
    config: ReviewPriorityConfig = DEFAULT_REVIEW_PRIORITY_CONFIG,
) -> float:
    priority = 0.0
    mismatch = true_label != predicted_label

    if mismatch:
        priority += 300.0
        if confidence is None:
            priority += 20.0
        elif confidence >= config.high_confidence_mismatch_threshold:
            priority += 100.0 + confidence * 20.0
        elif confidence < config.low_confidence_threshold:
            priority += 40.0 + (config.low_confidence_threshold - confidence) * 50.0
        else:
            priority += confidence * 20.0
    elif confidence is None:
        priority += 5.0
    elif confidence < config.very_low_confidence_threshold:
        priority += 140.0 + (config.very_low_confidence_threshold - confidence) * 50.0
    elif confidence < config.low_confidence_threshold:
        priority += 100.0 + (config.low_confidence_threshold - confidence) * 40.0

    quality_weights = {
        "no_speech_detected": 45.0,
        "short_speech": 25.0,
        "late_onset": 10.0,
    }
    priority += sum(quality_weights.get(flag, 0.0) for flag in quality_flags)
    return round(priority, 4)


def build_review_candidate(
    *,
    filepath: str,
    true_label: str,
    predicted_label: str,
    confidence: float | None,
    top_candidates: list[PredictionCandidate] | None = None,
    quality_stats: Mapping[str, Any] | None = None,
    existing_decision: ReviewDecision | None = None,
    config: ReviewPriorityConfig = DEFAULT_REVIEW_PRIORITY_CONFIG,
) -> ReviewCandidate:
    stats = quality_stats or {}
    onset_ms = _optional_float(stats.get("onset_ms"))
    offset_ms = _optional_float(stats.get("offset_ms"))
    speech_duration_ms = _optional_float(stats.get("speech_duration_ms"))
    flags = build_quality_flags(
        onset_ms=onset_ms,
        speech_duration_ms=speech_duration_ms,
        config=config,
    )
    normalized_path = normalize_filepath(filepath)
    priority = compute_review_priority(
        true_label=true_label,
        predicted_label=predicted_label,
        confidence=confidence,
        quality_flags=flags,
        config=config,
    )
    return ReviewCandidate(
        filepath=normalized_path,
        true_label=true_label,
        predicted_label=predicted_label,
        confidence=confidence,
        review_priority=priority,
        decision=existing_decision.decision if existing_decision else "",
        top_candidates=top_candidates or [],
        quality_flags=flags,
        onset_ms=onset_ms,
        offset_ms=offset_ms,
        speech_duration_ms=speech_duration_ms,
    )


def _coerce_decision(value: Any) -> ReviewDecisionValue | None:
    if value in VALID_REVIEW_DECISIONS:
        return cast(ReviewDecisionValue, value)
    return None


def load_review_decisions(path: Path | str | None) -> dict[str, ReviewDecision]:
    if path is None:
        return {}

    decision_path = Path(path)
    if not decision_path.exists():
        return {}

    with open(decision_path, encoding="utf-8") as f:
        payload = json.load(f)

    raw_items: Any = payload.get("decisions", []) if isinstance(payload, dict) else payload

    decisions: dict[str, ReviewDecision] = {}
    if isinstance(raw_items, dict):
        raw_items = [
            {"filepath": filepath, "decision": decision} for filepath, decision in raw_items.items()
        ]

    if not isinstance(raw_items, list):
        return decisions

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        filepath = normalize_filepath(str(item.get("filepath", "")))
        decision = _coerce_decision(item.get("decision"))
        if not filepath or decision is None:
            continue

        decisions[filepath] = ReviewDecision(
            filepath=filepath,
            label=str(item.get("label", item.get("true_label", ""))),
            prediction=str(item.get("prediction", item.get("predicted_label", ""))),
            confidence=_optional_float(item.get("confidence")),
            decision=decision,
            decided_at=str(item.get("decided_at", "")),
        )

    return decisions


def write_review_decisions(
    output_path: Path | str,
    decisions: Mapping[str, ReviewDecision],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": utc_now_iso(),
        "decisions": [asdict(decision) for decision in decisions.values()],
    }
    temp_path = path.with_name(f"{path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def ensure_review_decisions_file(path: Path | str | None) -> None:
    if path is None:
        return
    decision_path = Path(path)
    if decision_path.exists():
        return
    write_review_decisions(decision_path, {})


def _review_priority(candidate: ReviewCandidate) -> float:
    return candidate.review_priority


def review_candidates_payload(candidates: list[ReviewCandidate]) -> dict[str, Any]:
    sorted_candidates = sorted(
        candidates,
        key=_review_priority,
        reverse=True,
    )
    return {
        "version": 1,
        "generated_at": utc_now_iso(),
        "total": len(sorted_candidates),
        "candidates": [asdict(candidate) for candidate in sorted_candidates],
    }


def write_review_candidates_json(
    output_path: Path | str,
    candidates: list[ReviewCandidate],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(review_candidates_payload(candidates), f, ensure_ascii=False, indent=2)


def _json_for_script(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def generate_review_html_report(
    candidates: list[ReviewCandidate],
    *,
    title: str = "Voice Data Quality Review",
    review_results_path: Path | str | None = None,
    storage_key: str = "voice-data-review",
) -> str:
    sorted_candidates = sorted(
        candidates,
        key=_review_priority,
        reverse=True,
    )
    payload = [asdict(candidate) for candidate in sorted_candidates]
    candidates_json = _json_for_script(payload)
    escaped_title = html.escape(title)
    escaped_results_path = html.escape(str(review_results_path or "review_decisions.json"))
    storage_key_json = _json_for_script(storage_key)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escaped_title}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #d6dae1;
      --keep: #166534;
      --keep-bg: #dcfce7;
      --delete: #991b1b;
      --delete-bg: #fee2e2;
      --maybe: #92400e;
      --maybe-bg: #fef3c7;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      line-height: 1.5;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0 0 10px; font-size: 1.35rem; }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 8px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-top: 12px;
    }}
    input, select, button {{
      font: inherit;
    }}
    input, select {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 7px 9px;
    }}
    button {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 7px 10px;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); }}
    main {{ padding: 18px 24px 32px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 10px;
      vertical-align: top;
      text-align: left;
      font-size: 0.9rem;
    }}
    th {{
      background: #eef1f5;
      color: #374151;
      position: sticky;
      top: 95px;
      z-index: 5;
    }}
    tr.is-reviewed {{ background: #fbfcfd; }}
    tr.is-hidden {{ display: none; }}
    .num {{ width: 56px; }}
    .priority {{ width: 88px; }}
    .labels {{ width: 148px; }}
    .confidence {{ width: 108px; }}
    .audio {{ width: 270px; }}
    .decision {{ width: 230px; }}
    .path {{ word-break: break-all; color: var(--muted); font-family: Consolas, monospace; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 7px;
      margin: 1px 3px 3px 0;
      background: #eef2ff;
      color: #3730a3;
      font-size: 0.82rem;
      white-space: nowrap;
    }}
    .flag {{ background: #fff7ed; color: #9a3412; }}
    .mismatch {{ color: var(--delete); font-weight: 700; }}
    .match {{ color: var(--keep); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .top-list {{ margin: 0; padding-left: 18px; }}
    .top-list li {{ margin-bottom: 2px; }}
    .decision-buttons {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
      margin-bottom: 7px;
    }}
    .decision-buttons button {{
      padding: 6px 4px;
      overflow-wrap: anywhere;
    }}
    button[data-decision="keep"].selected {{ background: var(--keep-bg); color: var(--keep); border-color: #86efac; font-weight: 700; }}
    button[data-decision="delete_candidate"].selected {{ background: var(--delete-bg); color: var(--delete); border-color: #fca5a5; font-weight: 700; }}
    button[data-decision="maybe"].selected {{ background: var(--maybe-bg); color: var(--maybe); border-color: #fcd34d; font-weight: 700; }}
    .status-line {{ font-size: 0.82rem; color: var(--muted); min-height: 1.2em; }}
    audio {{ width: 245px; max-width: 100%; height: 32px; }}
    @media (max-width: 900px) {{
      header {{ position: static; }}
      main {{ padding: 12px; }}
      table, thead, tbody, th, td, tr {{ display: block; }}
      thead {{ display: none; }}
      tr {{ border: 1px solid var(--border); border-radius: 8px; margin-bottom: 12px; background: var(--panel); }}
      td {{ border-bottom: 0; }}
      td::before {{
        content: attr(data-label);
        display: block;
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 3px;
      }}
      .num, .priority, .labels, .confidence, .audio, .decision {{ width: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <div class="summary">
      <span class="metric">Total: <strong id="countTotal">0</strong></span>
      <span class="metric">Pending: <strong id="countPending">0</strong></span>
      <span class="metric">Keep: <strong id="countKeep">0</strong></span>
      <span class="metric">Delete candidate: <strong id="countDelete">0</strong></span>
      <span class="metric">Maybe: <strong id="countMaybe">0</strong></span>
      <span id="saveState" class="muted">Decisions: {escaped_results_path}</span>
    </div>
    <div class="toolbar">
      <input id="searchBox" type="search" placeholder="Search label or path">
      <select id="filterMode">
        <option value="all">All candidates</option>
        <option value="pending">Pending only</option>
        <option value="mismatch">Mismatch</option>
        <option value="low_confidence">Low confidence</option>
        <option value="quality">Quality flags</option>
      </select>
      <button id="exportJson" type="button">Export JSON</button>
      <span class="muted">Shortcuts: K=keep, D=delete_candidate, M=maybe</span>
    </div>
  </header>
  <main>
    <table>
      <thead>
        <tr>
          <th class="num">#</th>
          <th class="priority">Priority</th>
          <th class="labels">Labels</th>
          <th class="confidence">Confidence</th>
          <th>Top candidates / Quality</th>
          <th class="audio">Audio</th>
          <th>Path</th>
          <th class="decision">Decision</th>
        </tr>
      </thead>
      <tbody id="reviewRows"></tbody>
    </table>
  </main>
  <script>
    const candidates = {candidates_json};
    const storageKey = {storage_key_json};
    const validDecisions = new Set(["keep", "delete_candidate", "maybe"]);
    const decisionLabels = {{
      keep: "正しい",
      delete_candidate: "間違い",
      maybe: "保留"
    }};
    const qualityLabels = {{
      no_speech_detected: "無音候補",
      short_speech: "短い発話",
      late_onset: "開始が遅い"
    }};
    const decisions = new Map();
    const rowByPath = new Map();

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function confidenceText(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) {{
        return "-";
      }}
      return Number(value).toFixed(3);
    }}

    function decisionRecord(candidate, decision) {{
      return {{
        filepath: candidate.filepath,
        label: candidate.true_label,
        prediction: candidate.predicted_label,
        confidence: candidate.confidence,
        decision,
        decided_at: new Date().toISOString()
      }};
    }}

    function loadDecisions() {{
      for (const candidate of candidates) {{
        if (validDecisions.has(candidate.decision)) {{
          decisions.set(candidate.filepath, decisionRecord(candidate, candidate.decision));
        }}
      }}
      try {{
        const raw = localStorage.getItem(storageKey);
        if (!raw) return;
        const payload = JSON.parse(raw);
        const items = Array.isArray(payload.decisions) ? payload.decisions : [];
        for (const item of items) {{
          if (item && item.filepath && validDecisions.has(item.decision)) {{
            decisions.set(item.filepath, item);
          }}
        }}
      }} catch (error) {{
        console.warn("Could not load local review decisions", error);
      }}
    }}

    function persistLocal() {{
      const payload = {{
        version: 1,
        updated_at: new Date().toISOString(),
        decisions: Array.from(decisions.values())
      }};
      localStorage.setItem(storageKey, JSON.stringify(payload));
    }}

    async function persistServer(record) {{
      try {{
        const response = await fetch("/api/review-decisions", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(record)
        }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        document.getElementById("saveState").textContent = "Saved to review_decisions.json";
      }} catch (error) {{
        document.getElementById("saveState").textContent =
          "Saved in this browser. Run script/review_server.py for direct JSON saving, or export JSON.";
      }}
    }}

    function topCandidatesHtml(candidate) {{
      if (!candidate.top_candidates || candidate.top_candidates.length === 0) {{
        return '<span class="muted">No Top-K data</span>';
      }}
      const items = candidate.top_candidates
        .map((item) => `<li>${{escapeHtml(item.label)}} <span class="muted">${{confidenceText(item.confidence)}}</span></li>`)
        .join("");
      return `<ol class="top-list">${{items}}</ol>`;
    }}

    function qualityHtml(candidate) {{
      if (!candidate.quality_flags || candidate.quality_flags.length === 0) {{
        return '<span class="muted">No quality flags</span>';
      }}
      return candidate.quality_flags
        .map((flag) => `<span class="badge flag">${{escapeHtml(qualityLabels[flag] || flag)}}</span>`)
        .join("");
    }}

    function rowHtml(candidate, index) {{
      const relAudio = candidate.filepath.replaceAll("\\\\", "/");
      const encodedAudio = encodeURI(relAudio);
      const matchClass = candidate.true_label === candidate.predicted_label ? "match" : "mismatch";
      const current = decisions.get(candidate.filepath);
      const selected = current ? current.decision : "";
      const decisionText = selected ? decisionLabels[selected] : "未レビュー";
      return `
        <tr data-filepath="${{escapeHtml(candidate.filepath)}}" data-index="${{index}}">
          <td class="num" data-label="#">${{index + 1}}</td>
          <td class="priority" data-label="Priority">${{candidate.review_priority.toFixed(2)}}</td>
          <td class="labels" data-label="Labels">
            <div>正解: <strong>${{escapeHtml(candidate.true_label)}}</strong></div>
            <div>予測: <strong class="${{matchClass}}">${{escapeHtml(candidate.predicted_label)}}</strong></div>
          </td>
          <td class="confidence" data-label="Confidence">${{confidenceText(candidate.confidence)}}</td>
          <td data-label="Top candidates / Quality">
            ${{topCandidatesHtml(candidate)}}
            <div style="margin-top:6px;">${{qualityHtml(candidate)}}</div>
            <div class="muted" style="margin-top:5px;">
              speech=${{confidenceText(candidate.speech_duration_ms)}}ms,
              onset=${{confidenceText(candidate.onset_ms)}}ms,
              offset=${{confidenceText(candidate.offset_ms)}}ms
            </div>
          </td>
          <td class="audio" data-label="Audio">
            <audio controls preload="none">
              <source src="../${{encodedAudio}}" type="audio/wav">
              <source src="/${{encodedAudio}}" type="audio/wav">
              <source src="${{encodedAudio}}" type="audio/wav">
            </audio>
          </td>
          <td class="path" data-label="Path">${{escapeHtml(candidate.filepath)}}</td>
          <td class="decision" data-label="Decision">
            <div class="decision-buttons">
              <button type="button" data-decision="keep" class="${{selected === "keep" ? "selected" : ""}}">正しい</button>
              <button type="button" data-decision="delete_candidate" class="${{selected === "delete_candidate" ? "selected" : ""}}">間違い</button>
              <button type="button" data-decision="maybe" class="${{selected === "maybe" ? "selected" : ""}}">保留</button>
            </div>
            <div class="status-line">${{escapeHtml(decisionText)}}</div>
          </td>
        </tr>`;
    }}

    function candidateMatches(candidate, mode, search) {{
      const record = decisions.get(candidate.filepath);
      if (mode === "pending" && record) return false;
      if (mode === "mismatch" && candidate.true_label === candidate.predicted_label) return false;
      if (mode === "low_confidence" && !(candidate.confidence !== null && candidate.confidence < 0.60)) return false;
      if (mode === "quality" && (!candidate.quality_flags || candidate.quality_flags.length === 0)) return false;
      if (!search) return true;
      const haystack = `${{candidate.filepath}} ${{candidate.true_label}} ${{candidate.predicted_label}}`.toLowerCase();
      return haystack.includes(search);
    }}

    function render() {{
      const mode = document.getElementById("filterMode").value;
      const search = document.getElementById("searchBox").value.trim().toLowerCase();
      const rows = candidates
        .map((candidate, index) => ({{
          candidate,
          index,
          visible: candidateMatches(candidate, mode, search)
        }}))
        .filter((item) => item.visible)
        .map((item) => rowHtml(item.candidate, item.index))
        .join("");
      document.getElementById("reviewRows").innerHTML = rows;
      rowByPath.clear();
      document.querySelectorAll("tr[data-filepath]").forEach((row) => {{
        rowByPath.set(row.dataset.filepath, row);
        applyRowDecision(row.dataset.filepath);
      }});
      updateCounts();
    }}

    function applyRowDecision(filepath) {{
      const row = rowByPath.get(filepath);
      if (!row) return;
      const record = decisions.get(filepath);
      row.classList.toggle("is-reviewed", Boolean(record));
      row.querySelectorAll("button[data-decision]").forEach((button) => {{
        button.classList.toggle("selected", Boolean(record && button.dataset.decision === record.decision));
      }});
      const status = row.querySelector(".status-line");
      status.textContent = record ? decisionLabels[record.decision] : "未レビュー";
    }}

    function updateCounts() {{
      const counts = {{ keep: 0, delete_candidate: 0, maybe: 0 }};
      for (const record of decisions.values()) {{
        if (record.decision in counts) counts[record.decision] += 1;
      }}
      document.getElementById("countTotal").textContent = String(candidates.length);
      document.getElementById("countKeep").textContent = String(counts.keep);
      document.getElementById("countDelete").textContent = String(counts.delete_candidate);
      document.getElementById("countMaybe").textContent = String(counts.maybe);
      document.getElementById("countPending").textContent = String(
        Math.max(0, candidates.length - decisions.size)
      );
    }}

    function focusNextPending(fromRow) {{
      const rows = Array.from(document.querySelectorAll("tr[data-filepath]"));
      const start = fromRow ? rows.indexOf(fromRow) + 1 : 0;
      const rotated = rows.slice(start).concat(rows.slice(0, start));
      const next = rotated.find((row) => !decisions.has(row.dataset.filepath));
      if (next) {{
        next.scrollIntoView({{ behavior: "smooth", block: "center" }});
        const audio = next.querySelector("audio");
        if (audio) audio.focus({{ preventScroll: true }});
      }}
    }}

    function setDecision(filepath, decision) {{
      if (!validDecisions.has(decision)) return;
      const candidate = candidates.find((item) => item.filepath === filepath);
      if (!candidate) return;
      const record = decisionRecord(candidate, decision);
      decisions.set(filepath, record);
      persistLocal();
      persistServer(record);
      const row = rowByPath.get(filepath);
      applyRowDecision(filepath);
      updateCounts();
      focusNextPending(row);
    }}

    document.getElementById("reviewRows").addEventListener("click", (event) => {{
      const button = event.target.closest("button[data-decision]");
      if (!button) return;
      const row = button.closest("tr[data-filepath]");
      setDecision(row.dataset.filepath, button.dataset.decision);
    }});

    document.addEventListener("keydown", (event) => {{
      if (event.target.matches("input, textarea, select, button")) return;
      const key = event.key.toLowerCase();
      const decision = key === "k" ? "keep" : key === "d" ? "delete_candidate" : key === "m" ? "maybe" : "";
      if (!decision) return;
      const row = Array.from(document.querySelectorAll("tr[data-filepath]"))
        .find((item) => !decisions.has(item.dataset.filepath));
      if (row) setDecision(row.dataset.filepath, decision);
    }});

    document.getElementById("filterMode").addEventListener("change", render);
    document.getElementById("searchBox").addEventListener("input", render);
    document.getElementById("exportJson").addEventListener("click", () => {{
      const payload = {{
        version: 1,
        updated_at: new Date().toISOString(),
        decisions: Array.from(decisions.values())
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "review_decisions.json";
      link.click();
      URL.revokeObjectURL(url);
    }});

    loadDecisions();
    render();
  </script>
</body>
</html>
"""
