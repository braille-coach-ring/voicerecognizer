from __future__ import annotations

import html
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

ReviewDecisionValue = Literal["keep", "delete_candidate", "maybe", "relabel", "other"]

VALID_REVIEW_DECISIONS: tuple[ReviewDecisionValue, ...] = (
    "keep",
    "delete_candidate",
    "maybe",
    "relabel",
    "other",
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
    new_label: str = ""
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
            new_label=str(item.get("new_label", "")),
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
    html, body {{
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      line-height: 1.5;
      display: flex;
      flex-direction: column;
    }}
    header {{
      flex-shrink: 0;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
      z-index: 20;
    }}
    .header-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    h1 {{ margin: 0; font-size: 1.25rem; white-space: nowrap; }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 0.88rem;
    }}
    .metric {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 3px 8px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }}
    input, select, button {{
      font: inherit;
    }}
    input, select {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 6px 9px;
      font-size: 0.88rem;
    }}
    button {{
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 6px 10px;
      cursor: pointer;
      font-size: 0.88rem;
    }}
    button:hover {{ border-color: var(--accent); }}
    main {{
      flex: 1;
      overflow-y: auto;
      padding: 16px 20px 32px;
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      vertical-align: top;
      text-align: left;
      font-size: 0.88rem;
    }}
    th {{
      background: #eef1f5;
      color: #374151;
      position: sticky;
      top: 0;
      z-index: 10;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }}
    tr.is-reviewed {{ background: #fafafa; }}
    tr.is-hidden {{ display: none; }}
    tr.is-active td {{
      background-color: #e0f2fe !important;
      border-top: 2px solid #0284c7;
      border-bottom: 2px solid #0284c7;
    }}
    tr.is-active td:first-child {{
      border-left: 4px solid #0284c7;
    }}
    .num {{ width: 50px; text-align: center; }}
    .priority {{ width: 75px; }}
    .labels {{ width: 140px; }}
    .confidence {{ width: 95px; }}
    .audio {{ width: 260px; }}
    .decision {{ width: 330px; }}
    .path {{ word-break: break-all; color: var(--muted); font-family: Consolas, monospace; font-size: 0.82rem; }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 7px;
      margin: 1px 3px 3px 0;
      background: #eef2ff;
      color: #3730a3;
      font-size: 0.8rem;
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
      grid-template-columns: 1fr 1fr;
      gap: 5px;
      margin-bottom: 5px;
    }}
    .decision-buttons button {{
      padding: 5px 6px;
      font-size: 0.82rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      text-align: left;
    }}
    .kbd-hint {{
      display: inline-block;
      background: rgba(0, 0, 0, 0.07);
      border: 1px solid rgba(0, 0, 0, 0.15);
      border-radius: 3px;
      padding: 1px 4px;
      font-size: 0.72rem;
      font-family: monospace;
      font-weight: bold;
      margin-right: 4px;
    }}
    button[data-decision="keep"].selected {{ background: var(--keep-bg); color: var(--keep); border-color: #86efac; font-weight: 700; }}
    .other-btn {{ background: #fffbeb; color: #b45309; border: 1px solid #fde68a; font-weight: 600; }}
    .other-btn:hover {{ background: #fef3c7; }}
    .other-btn.selected {{ background: #f59e0b; color: #fff; border-color: #d97706; font-weight: 700; }}
    button[data-decision="delete_candidate"].selected {{ background: var(--delete-bg); color: var(--delete); }}
    button[data-decision="maybe"].selected {{ background: var(--maybe-bg); color: var(--maybe); border-color: #fcd34d; font-weight: 700; }}
    .quick-relabel-btn {{
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      color: #15803d;
      font-weight: 600;
    }}
    .quick-relabel-btn:hover {{ background: #dcfce7; }}
    .quick-relabel-btn.selected {{ background: #16a34a; color: #fff; border-color: #15803d; font-weight: 700; }}
    .relabel-box {{
      display: flex;
      gap: 4px;
      align-items: center;
      margin-bottom: 4px;
    }}
    .relabel-input {{
      flex: 1;
      min-width: 0;
      padding: 4px 6px;
      font-size: 0.82rem;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: #fff;
    }}
    .relabel-btn {{
      padding: 4px 8px;
      font-size: 0.8rem;
      background: #f0f9ff;
      color: #0369a1;
      border: 1px solid #bae6fd;
      border-radius: 4px;
      font-weight: 600;
      cursor: pointer;
    }}
    .relabel-btn:hover {{ background: #e0f2fe; }}
    .relabel-btn.selected {{ background: #0284c7; color: #fff; border-color: #0369a1; font-weight: 700; }}
    .status-line {{ font-size: 0.8rem; color: var(--muted); min-height: 1.2em; font-weight: 500; }}
    audio {{ width: 245px; max-width: 100%; height: 30px; }}
    .guide-banner {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px 14px;
      background: #1e293b;
      color: #f8fafc;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      align-items: center;
    }}
    .guide-banner .key {{
      display: inline-block;
      background: #334155;
      border: 1px solid #64748b;
      color: #38bdf8;
      font-family: monospace;
      font-weight: 700;
      padding: 1px 4px;
      border-radius: 3px;
    }}
    @media (max-width: 900px) {{
      html, body {{ height: auto; overflow: visible; }}
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
    <div class="header-top">
      <h1>{escaped_title}</h1>
      <div class="summary">
        <span class="metric">全件: <strong id="countTotal">0</strong></span>
        <span class="metric">未確認: <strong id="countPending">0</strong></span>
        <span class="metric" style="color:var(--keep)">現行OK: <strong id="countKeep">0</strong></span>
        <span class="metric" style="color:#b45309">雑音: <strong id="countOther">0</strong></span>
        <span class="metric" style="color:#0284c7">修正: <strong id="countRelabel">0</strong></span>
        <span class="metric">保留: <strong id="countMaybe">0</strong></span>
        <span id="saveState" class="muted" style="margin-left: 4px;"></span>
      </div>
    </div>
    <div class="toolbar">
      <input id="searchBox" type="search" placeholder="音素名・パスで検索">
      <select id="filterMode">
        <option value="all" selected>全サンプル表示</option>
        <option value="pending">未確認のみ</option>
        <option value="mismatch">モデル不一致のみ</option>
        <option value="other">雑音 (other) のみ</option>
        <option value="relabel">音素修正のみ</option>
        <option value="low_confidence">低確信度 (&lt; 0.6)</option>
        <option value="quality">品質フラグあり</option>
      </select>
      <label style="display:flex; align-items:center; gap:5px; cursor:pointer; font-weight:600; font-size:0.86rem; user-select:none;">
        <input type="checkbox" id="autoPlayToggle" checked> 音声自動再生 (Auto Play)
      </label>
      <button id="exportJson" type="button" style="margin-left:auto;">Export JSON</button>
    </div>
    <div class="guide-banner">
      <span>⚡ <strong>キー操作</strong>:</span>
      <span><span class="key">1</span> or <span class="key">A</span> 現行OK</span>
      <span><span class="key">2</span> or <span class="key">S</span> 予測適用</span>
      <span><span class="key">3</span> or <span class="key">D</span> 雑音(other)</span>
      <span><span class="key">4</span> or <span class="key">F</span> 保留</span>
      <span><span class="key">Space</span> 音声再生</span>
      <span><span class="key">E</span> 音素手動入力</span>
      <span><span class="key">↓</span> / <span class="key">J</span> 次 / <span class="key">↑</span> / <span class="key">K</span> 前</span>
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
          <th class="decision">判定アクション</th>
        </tr>
      </thead>
      <tbody id="reviewRows"></tbody>
    </table>
  </main>
  <script>
    const candidates = {candidates_json};
    const storageKey = {storage_key_json};
    const validDecisions = new Set(["keep", "delete_candidate", "maybe", "relabel", "other"]);
    const decisionLabels = {{
      keep: "現行ラベルでOK",
      other: "雑音・その他 (other)",
      relabel: "音素ラベル修正",
      maybe: "保留",
      delete_candidate: "削除"
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

    function decisionRecord(candidate, decision, newLabel = "") {{
      return {{
        filepath: candidate.filepath,
        label: candidate.true_label,
        prediction: candidate.predicted_label,
        confidence: candidate.confidence,
        decision,
        new_label: newLabel,
        decided_at: new Date().toISOString()
      }};
    }}

    function loadDecisions() {{
      for (const candidate of candidates) {{
        if (validDecisions.has(candidate.decision)) {{
          decisions.set(candidate.filepath, decisionRecord(candidate, candidate.decision, candidate.new_label || ""));
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
      const isOther = selected === "other" || (selected === "relabel" && current && current.new_label === "other");

      let decisionText = "未確認";
      if (selected === "keep") {{
        decisionText = "✅ 現行ラベルでOK";
      }} else if (isOther) {{
        decisionText = "🔇 雑音 ➔ other フォルダへ移動";
      }} else if (selected === "relabel" && current && current.new_label) {{
        decisionText = `✏️ 修正 ➔ ${{escapeHtml(current.new_label)}} フォルダへ移動`;
      }} else if (selected === "maybe") {{
        decisionText = "⏳ 保留";
      }}

      const inputVal = (current && current.new_label && current.new_label !== "other") ? current.new_label : "";

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
              <button type="button" data-decision="keep" class="${{selected === "keep" ? "selected" : ""}}" title="音声は正しく発音されている [1 / A]"><span class="kbd-hint">1/A</span>現行OK</button>
              <button type="button" data-action="quick-relabel" data-target="${{escapeHtml(candidate.predicted_label)}}" class="quick-relabel-btn ${{selected === "relabel" && current && current.new_label === candidate.predicted_label ? "selected" : ""}}" title="予測 '${{escapeHtml(candidate.predicted_label)}}' を適用 [2 / S]"><span class="kbd-hint">2/S</span>予測 [${{escapeHtml(candidate.predicted_label)}}]</button>
              <button type="button" data-action="set-other" class="other-btn ${{isOther ? "selected" : ""}}" title="雑音・咳・無音など [3 / D]"><span class="kbd-hint">3/D</span>雑音 (other)</button>
              <button type="button" data-decision="delete_candidate" class="${{selected === "delete_candidate" ? "selected" : ""}}" title="学習データからの削除候補として記録">削除候補</button>
              <button type="button" data-decision="maybe" class="${{selected === "maybe" ? "selected" : ""}}" title="判断保留 [4 / F]"><span class="kbd-hint">4/F</span>保留</button>
            </div>
            <div class="relabel-box">
              <input type="text" class="relabel-input" placeholder="手動修正 (Eで入力, Enter確定)" value="${{escapeHtml(inputVal)}}">
              <button type="button" data-action="relabel" class="relabel-btn ${{selected === "relabel" && !isOther && (!current || current.new_label !== candidate.predicted_label) ? "selected" : ""}}" title="手動入力で変更">変更</button>
            </div>
            <div class="status-line">${{escapeHtml(decisionText)}}</div>
          </td>
        </tr>`;
    }}

    function candidateMatches(candidate, mode, search) {{
      const record = decisions.get(candidate.filepath);
      if (mode === "pending" && record) return false;
      if (mode === "mismatch" && candidate.true_label === candidate.predicted_label) return false;
      if (mode === "other" && (!record || (record.decision !== "other" && record.new_label !== "other"))) return false;
      if (mode === "relabel" && (!record || record.decision !== "relabel" || record.new_label === "other")) return false;
      if (mode === "low_confidence" && !(candidate.confidence !== null && candidate.confidence < 0.60)) return false;
      if (mode === "quality" && (!candidate.quality_flags || candidate.quality_flags.length === 0)) return false;
      if (!search) return true;
      const haystack = `${{candidate.filepath}} ${{candidate.true_label}} ${{candidate.predicted_label}}`.toLowerCase();
      return haystack.includes(search);
    }}

    let activeFilepath = null;

    function getVisibleRows() {{
      return Array.from(document.querySelectorAll("#reviewRows tr[data-filepath]"));
    }}

    function getActiveRow() {{
      if (!activeFilepath) return null;
      return rowByPath.get(activeFilepath) || null;
    }}

    function playRowAudio(row) {{
      if (!row) return;
      document.querySelectorAll("audio").forEach((a) => {{
        if (!a.paused) {{
          a.pause();
          a.currentTime = 0;
        }}
      }});
      const audio = row.querySelector("audio");
      if (audio) {{
        audio.currentTime = 0;
        audio.play().catch((err) => {{
          console.warn("Autoplay blocked or audio load error:", err);
        }});
      }}
    }}

    function setActiveRow(row, autoPlay = true) {{
      document.querySelectorAll("tr.is-active").forEach((r) => r.classList.remove("is-active"));
      if (!row) {{
        activeFilepath = null;
        return;
      }}
      row.classList.add("is-active");
      activeFilepath = row.dataset.filepath;
      row.scrollIntoView({{ behavior: "smooth", block: "nearest" }});
      if (autoPlay && document.getElementById("autoPlayToggle").checked) {{
        playRowAudio(row);
      }}
    }}

    function moveActiveRow(delta) {{
      const rows = getVisibleRows();
      if (rows.length === 0) return;
      const current = getActiveRow();
      let index = current ? rows.indexOf(current) : -1;
      let nextIndex = index + delta;
      if (nextIndex < 0) nextIndex = 0;
      if (nextIndex >= rows.length) nextIndex = rows.length - 1;
      setActiveRow(rows[nextIndex], true);
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

      // Maintain or find active row
      const visibleRows = getVisibleRows();
      if (visibleRows.length > 0) {{
        const existing = activeFilepath ? rowByPath.get(activeFilepath) : null;
        if (existing && visibleRows.includes(existing)) {{
          setActiveRow(existing, false);
        }} else {{
          const firstPending = visibleRows.find((r) => !decisions.has(r.dataset.filepath)) || visibleRows[0];
          setActiveRow(firstPending, false);
        }}
      }} else {{
        activeFilepath = null;
      }}
    }}

    function applyRowDecision(filepath) {{
      const row = rowByPath.get(filepath);
      if (!row) return;
      const record = decisions.get(filepath);
      row.classList.toggle("is-reviewed", Boolean(record));

      const selected = record ? record.decision : "";
      const isOther = selected === "other" || (selected === "relabel" && record && record.new_label === "other");

      row.querySelectorAll("button[data-decision]").forEach((button) => {{
        button.classList.toggle("selected", Boolean(record && button.dataset.decision === record.decision));
      }});
      const otherBtn = row.querySelector("button[data-action='set-other']");
      if (otherBtn) {{
        otherBtn.classList.toggle("selected", Boolean(isOther));
      }}
      const quickBtn = row.querySelector("button[data-action='quick-relabel']");
      if (quickBtn) {{
        quickBtn.classList.toggle("selected", Boolean(record && record.decision === "relabel" && record.new_label === quickBtn.dataset.target));
      }}
      const relabelBtn = row.querySelector("button[data-action='relabel']");
      if (relabelBtn) {{
        const isQuick = quickBtn && record && record.new_label === quickBtn.dataset.target;
        relabelBtn.classList.toggle("selected", Boolean(record && record.decision === "relabel" && !isOther && !isQuick));
      }}
      const input = row.querySelector(".relabel-input");
      if (input && record && record.new_label && record.new_label !== "other") {{
        input.value = record.new_label;
      }}
      const status = row.querySelector(".status-line");
      if (selected === "keep") {{
        status.textContent = "✅ 現行ラベルでOK";
      }} else if (isOther) {{
        status.textContent = "🔇 雑音 ➔ other フォルダへ移動";
      }} else if (selected === "relabel" && record && record.new_label) {{
        status.textContent = `✏️ 修正 ➔ ${{escapeHtml(record.new_label)}} フォルダへ移動`;
      }} else if (selected === "maybe") {{
        status.textContent = "⏳ 保留";
      }} else {{
        status.textContent = "未確認";
      }}
    }}

    function updateCounts() {{
      const counts = {{ keep: 0, delete_candidate: 0, maybe: 0, relabel: 0, other: 0 }};
      for (const record of decisions.values()) {{
        if (record.decision === "other" || record.new_label === "other") {{
          counts.other += 1;
        }} else if (record.decision in counts) {{
          counts[record.decision] += 1;
        }}
      }}
      document.getElementById("countTotal").textContent = String(candidates.length);
      document.getElementById("countKeep").textContent = String(counts.keep);
      document.getElementById("countOther").textContent = String(counts.other);
      document.getElementById("countRelabel").textContent = String(counts.relabel);
      document.getElementById("countMaybe").textContent = String(counts.maybe);
      document.getElementById("countPending").textContent = String(
        Math.max(0, candidates.length - decisions.size)
      );
    }}

    function focusNextPending(fromRow) {{
      const rows = getVisibleRows();
      if (rows.length === 0) return;
      const start = fromRow ? rows.indexOf(fromRow) + 1 : 0;
      const rotated = rows.slice(start).concat(rows.slice(0, start));
      const next = rotated.find((row) => !decisions.has(row.dataset.filepath));
      if (next) {{
        setActiveRow(next, true);
      }} else if (fromRow && start < rows.length) {{
        setActiveRow(rows[start], true);
      }}
    }}

    function setDecision(filepath, decision, newLabel = "") {{
      if (!validDecisions.has(decision)) return;
      const candidate = candidates.find((item) => item.filepath === filepath);
      if (!candidate) return;
      const record = decisionRecord(candidate, decision, newLabel);
      decisions.set(filepath, record);
      persistLocal();
      persistServer(record);
      const row = rowByPath.get(filepath);
      applyRowDecision(filepath);
      updateCounts();
      focusNextPending(row);
    }}

    document.getElementById("reviewRows").addEventListener("click", (event) => {{
      const row = event.target.closest("tr[data-filepath]");
      if (row && row !== getActiveRow()) {{
        setActiveRow(row, false);
      }}

      const button = event.target.closest("button");
      if (!button || !row) return;
      const filepath = row.dataset.filepath;

      if (button.dataset.decision) {{
        setDecision(filepath, button.dataset.decision);
        return;
      }}

      if (button.dataset.action === "set-other") {{
        setDecision(filepath, "other", "other");
        return;
      }}

      if (button.dataset.action === "quick-relabel") {{
        const targetLabel = button.dataset.target;
        const input = row.querySelector(".relabel-input");
        if (input) input.value = targetLabel;
        setDecision(filepath, "relabel", targetLabel);
        return;
      }}

      if (button.dataset.action === "relabel") {{
        const input = row.querySelector(".relabel-input");
        const val = input ? input.value.trim() : "";
        if (!val) {{
          alert("変更先の音素名（例: a, ka, shi等）を入力してください");
          if (input) input.focus();
          return;
        }}
        setDecision(filepath, "relabel", val);
        return;
      }}
    }});

    document.getElementById("reviewRows").addEventListener("keydown", (event) => {{
      if (event.key === "Enter" && event.target.classList.contains("relabel-input")) {{
        event.preventDefault();
        const row = event.target.closest("tr[data-filepath]");
        if (!row) return;
        const val = event.target.value.trim();
        if (!val) {{
          alert("変更先の音素名を入力してください");
          return;
        }}
        event.target.blur();
        setDecision(row.dataset.filepath, "relabel", val);
      }} else if (event.key === "Escape" && event.target.classList.contains("relabel-input")) {{
        event.target.blur();
      }}
    }});

    document.addEventListener("keydown", (event) => {{
      const isInput = event.target.matches("input, textarea, select");
      if (isInput) return;

      const key = event.key.toLowerCase();
      const activeRow = getActiveRow() || getVisibleRows()[0];
      if (!activeRow) return;
      const filepath = activeRow.dataset.filepath;

      // Navigation: J/K or ArrowDown/ArrowUp
      if (event.key === "ArrowDown" || key === "j") {{
        event.preventDefault();
        moveActiveRow(1);
        return;
      }}
      if (event.key === "ArrowUp" || key === "k") {{
        event.preventDefault();
        moveActiveRow(-1);
        return;
      }}

      // Audio replay: Space
      if (event.code === "Space" || event.key === " ") {{
        event.preventDefault();
        playRowAudio(activeRow);
        return;
      }}

      // Manual input focus: E
      if (key === "e") {{
        event.preventDefault();
        const input = activeRow.querySelector(".relabel-input");
        if (input) {{
          input.focus();
          input.select();
        }}
        return;
      }}

      // Jump to next pending: N
      if (key === "n") {{
        event.preventDefault();
        focusNextPending(activeRow);
        return;
      }}

      // Fast Decision keys:
      // 1 or A: Keep
      // 2 or S: Quick Relabel (predict)
      // 3 or D: Other (noise)
      // 4 or F: Maybe (pending)
      if (key === "1" || key === "a") {{
        event.preventDefault();
        setDecision(filepath, "keep");
      }} else if (key === "2" || key === "s") {{
        event.preventDefault();
        const quickBtn = activeRow.querySelector("button[data-action='quick-relabel']");
        if (quickBtn && quickBtn.dataset.target) {{
          setDecision(filepath, "relabel", quickBtn.dataset.target);
        }}
      }} else if (key === "3" || key === "d" || key === "o") {{
        event.preventDefault();
        setDecision(filepath, "other", "other");
      }} else if (key === "4" || key === "f" || key === "m") {{
        event.preventDefault();
        setDecision(filepath, "maybe");
      }}
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
