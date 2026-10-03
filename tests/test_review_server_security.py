"""review_server の露出範囲と POST 保護のテスト。

静的配信のルートはプロジェクト全体 (音声を相対パスで再生するため必要) なので、
.env などが読めないこと、および任意サイトから決定ファイルを書き換えられないことを保証する。
"""

import http.client
import json
import tempfile
import threading
import unittest
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, override

from script.review_server import MAX_POST_BYTES, ReviewRequestHandler, _is_hidden_path


class TestHiddenPathDetection(unittest.TestCase):
    def test_dotfiles_are_hidden(self) -> None:
        for path in ("/.env", "/.git/config", "/sub/.env", "/.venv/pyvenv.cfg"):
            self.assertTrue(_is_hidden_path(path), path)

    def test_normal_report_paths_are_allowed(self) -> None:
        for path in (
            "/",
            "/evaluation_results/review_report.html",
            "/dataset/collected/pc_1/1.wav",
            "/./evaluation_results/review_report.html",
        ):
            self.assertFalse(_is_hidden_path(path), path)


class TestReviewServerRequests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / ".env").write_text("HF_TOKEN=hf_secret_value\n", encoding="utf-8")
        (self.root / "evaluation_results").mkdir()
        (self.root / "evaluation_results" / "review_report.html").write_text(
            "<html>report</html>", encoding="utf-8"
        )
        self.decisions_file = self.root / "evaluation_results" / "review_decisions.json"

        root = self.root
        decisions_file = self.decisions_file

        class Handler(ReviewRequestHandler):
            decisions_path = decisions_file

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=str(root), **kwargs)

            @override
            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @override
    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self._tmp.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        content_length: str | None = None,
    ) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.putrequest(method, path)
            for name, value in (headers or {}).items():
                conn.putheader(name, value)
            if body is not None:
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", content_length or str(len(body)))
            conn.endheaders()
            if body is not None:
                conn.send(body)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_dotfile_is_not_served(self) -> None:
        status, payload = self._request("GET", "/.env")
        self.assertEqual(status, HTTPStatus.FORBIDDEN)
        self.assertNotIn(b"hf_secret_value", payload)

    def test_dotfile_head_is_not_served(self) -> None:
        status, _ = self._request("HEAD", "/.env")
        self.assertEqual(status, HTTPStatus.FORBIDDEN)

    def test_report_is_served(self) -> None:
        status, payload = self._request("GET", "/evaluation_results/review_report.html")
        self.assertEqual(status, HTTPStatus.OK)
        self.assertIn(b"report", payload)

    def test_post_without_origin_is_accepted(self) -> None:
        body = json.dumps(
            {"decisions": [{"filepath": "dataset/a/1.wav", "decision": "keep"}]}
        ).encode()
        status, _ = self._request("POST", "/api/review-decisions", body=body)
        self.assertEqual(status, HTTPStatus.OK)

        saved = json.loads(self.decisions_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["decisions"][0]["filepath"], "dataset/a/1.wav")

    def test_post_from_other_origin_is_rejected(self) -> None:
        body = json.dumps(
            {"decisions": [{"filepath": "dataset/a/1.wav", "decision": "delete_candidate"}]}
        ).encode()
        status, _ = self._request(
            "POST",
            "/api/review-decisions",
            body=body,
            headers={"Origin": "https://evil.example.com"},
        )
        self.assertEqual(status, HTTPStatus.FORBIDDEN)
        self.assertFalse(self.decisions_file.exists(), "拒否したのに決定ファイルが書かれている")

    def test_post_from_same_origin_is_accepted(self) -> None:
        body = json.dumps(
            {"decisions": [{"filepath": "dataset/a/1.wav", "decision": "maybe"}]}
        ).encode()
        status, _ = self._request(
            "POST",
            "/api/review-decisions",
            body=body,
            headers={"Origin": f"http://127.0.0.1:{self.port}"},
        )
        self.assertEqual(status, HTTPStatus.OK)

    def test_oversized_content_length_is_rejected_without_reading(self) -> None:
        status, _ = self._request(
            "POST",
            "/api/review-decisions",
            body=b"{}",
            content_length=str(MAX_POST_BYTES + 1),
        )
        self.assertEqual(status, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

    def test_malformed_content_length_is_rejected(self) -> None:
        status, _ = self._request(
            "POST", "/api/review-decisions", body=b"{}", content_length="not-a-number"
        )
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)


if __name__ == "__main__":
    unittest.main()
