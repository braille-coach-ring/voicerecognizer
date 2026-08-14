from __future__ import annotations

import argparse
import json
import logging
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
REVIEW_ROOT = SRC_ROOT / "voicerecognizer" / "evaluation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(REVIEW_ROOT))

from review import (  # noqa: E402
    ReviewDecision,
    ReviewDecisionValue,
    load_review_decisions,
    normalize_filepath,
    utc_now_iso,
    write_review_decisions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    decisions_path: Path

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: HTTPStatus) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/review-decisions":
            decisions = load_review_decisions(self.decisions_path)
            self._send_json(
                {
                    "version": 1,
                    "decisions": [decision.__dict__ for decision in decisions.values()],
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/review-decisions":
            self._send_error_json("Unknown endpoint", HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_error_json("Invalid JSON", HTTPStatus.BAD_REQUEST)
            return

        incoming = payload.get("decisions") if isinstance(payload, dict) else None
        if incoming is None:
            incoming = [payload]
        if not isinstance(incoming, list):
            self._send_error_json("Expected a decision object or decisions list", HTTPStatus.BAD_REQUEST)
            return

        decisions = load_review_decisions(self.decisions_path)
        for item in incoming:
            if not isinstance(item, dict):
                continue

            filepath = normalize_filepath(str(item.get("filepath", "")))
            decision = item.get("decision")
            if not filepath or decision not in {"keep", "delete_candidate", "maybe"}:
                continue
            decision_value = cast(ReviewDecisionValue, decision)

            decisions[filepath] = ReviewDecision(
                filepath=filepath,
                label=str(item.get("label", item.get("true_label", ""))),
                prediction=str(item.get("prediction", item.get("predicted_label", ""))),
                confidence=_optional_float(item.get("confidence")),
                decision=decision_value,
                decided_at=str(item.get("decided_at") or utc_now_iso()),
            )

        write_review_decisions(self.decisions_path, decisions)
        self._send_json({"ok": True, "saved": len(decisions)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve review_report.html and save decisions.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Static file root. Defaults to the project root.",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "review_decisions.json",
        help="JSON file where review decisions are saved.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    resolved_decisions_path = args.decisions.resolve()

    class Handler(ReviewRequestHandler):
        decisions_path = resolved_decisions_path

        def __init__(self, *handler_args: Any, **handler_kwargs: Any) -> None:
            super().__init__(*handler_args, directory=str(root), **handler_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    report_url = f"http://{args.host}:{args.port}/evaluation_results/review_report.html"
    logger.info("Serving %s", root)
    logger.info("Saving decisions to %s", resolved_decisions_path)
    logger.info("Open %s", report_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping review server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
