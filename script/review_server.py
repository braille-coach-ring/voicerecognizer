from __future__ import annotations

import argparse
import json
import logging
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast, override
from urllib.parse import urlparse

from voicerecognizer.evaluation.review import (
    ReviewDecision,
    ReviewDecisionValue,
    load_review_decisions,
    normalize_filepath,
    utc_now_iso,
    write_review_decisions,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
REVIEW_ROOT = SRC_ROOT / "voicerecognizer" / "evaluation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(REVIEW_ROOT))


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 保存する review_decisions.json は小さい JSON なので、上限を超える POST は読まずに弾く
MAX_POST_BYTES = 4 * 1024 * 1024

API_PATH = "/api/review-decisions"


def _is_hidden_path(url_path: str) -> bool:
    """ドット始まりの要素を含むパスかどうか。

    静的配信のルートはプロジェクト全体 (音声ファイルを相対パスで再生するため必要) なので、
    .env / .git / .venv などがそのまま読めてしまう。ドット始まりは一律で拒否する。
    """
    return any(part.startswith(".") for part in url_path.split("/") if part not in ("", "."))


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

    @override
    def send_head(self) -> Any:
        # GET / HEAD の両方がここを通る
        if _is_hidden_path(urlparse(self.path).path):
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return None
        return super().send_head()

    def _is_same_origin_request(self) -> bool:
        """ブラウザが付ける Origin が自分自身かどうか。

        Origin 検査がないと、ユーザーが開いた任意のサイトから 127.0.0.1 の
        この API に POST して review_decisions.json を書き換えられる。
        """
        origin = self.headers.get("Origin")
        if origin is None:
            # curl など、ブラウザ経由でないリクエストは Origin を付けない
            return True
        host = self.headers.get("Host")
        return bool(host) and urlparse(origin).netloc == host

    @override
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == API_PATH:
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
        if path != API_PATH:
            self._send_error_json("Unknown endpoint", HTTPStatus.NOT_FOUND)
            return

        if not self._is_same_origin_request():
            logger.warning("Rejected cross-origin POST from %s", self.headers.get("Origin"))
            self._send_error_json("Cross-origin request rejected", HTTPStatus.FORBIDDEN)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error_json("Invalid Content-Length", HTTPStatus.BAD_REQUEST)
            return
        if length < 0:
            self._send_error_json("Invalid Content-Length", HTTPStatus.BAD_REQUEST)
            return
        if length > MAX_POST_BYTES:
            self._send_error_json(
                f"Payload too large (limit {MAX_POST_BYTES} bytes)",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error_json("Invalid JSON", HTTPStatus.BAD_REQUEST)
            return

        incoming = payload.get("decisions") if isinstance(payload, dict) else None
        if incoming is None:
            incoming = [payload]
        if not isinstance(incoming, list):
            self._send_error_json(
                "Expected a decision object or decisions list", HTTPStatus.BAD_REQUEST
            )
            return

        decisions = load_review_decisions(self.decisions_path)
        for item in incoming:
            if not isinstance(item, dict):
                continue

            filepath = normalize_filepath(str(item.get("filepath", "")))
            decision = item.get("decision")
            if not filepath or decision not in {"keep", "delete_candidate", "maybe", "relabel", "other"}:
                continue
            decision_value = cast(ReviewDecisionValue, decision)

            decisions[filepath] = ReviewDecision(
                filepath=filepath,
                label=str(item.get("label", item.get("true_label", ""))),
                prediction=str(item.get("prediction", item.get("predicted_label", ""))),
                confidence=_optional_float(item.get("confidence")),
                decision=decision_value,
                new_label=str(item.get("new_label", "")),
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
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "%s 配下を %s で公開します。ローカル以外に bind するとプロジェクトのファイルが"
            "ネットワークから読めるため、必要な場合のみにしてください。",
            root,
            args.host,
        )
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
