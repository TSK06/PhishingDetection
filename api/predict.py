import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit

from prediction import predictor
from validation import validate_url


MAX_BODY_BYTES = 16_384
DEFAULT_ALLOWED_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}


def _allowed_origins() -> set[str]:
    configured = os.getenv("ALLOWED_ORIGINS")
    if configured is None:
        return DEFAULT_ALLOWED_ORIGINS
    return {origin.strip() for origin in configured.split(",") if origin.strip()}


class handler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict[str, str]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin")
        if origin in _allowed_origins():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin")
        self.send_response(204)
        if origin in _allowed_origins():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        if urlsplit(self.path).path not in {"/api/predict", "/predict"}:
            self._write_json(404, {"detail": "Not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(400, {"detail": "Invalid Content-Length header."})
            return
        if content_length <= 0:
            self._write_json(400, {"detail": "Request body is required."})
            return
        if content_length > MAX_BODY_BYTES:
            self._write_json(413, {"detail": "Request body is too large."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._write_json(400, {"detail": "Request body must be valid JSON."})
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("url"), str):
            self._write_json(422, {"detail": "Provide a URL as a string."})
            return
        try:
            url = validate_url(payload["url"])
        except ValueError as exc:
            self._write_json(422, {"detail": str(exc)})
            return

        label = "good" if predictor(url) == 0 else "bad"
        self._write_json(200, {"label": label})
