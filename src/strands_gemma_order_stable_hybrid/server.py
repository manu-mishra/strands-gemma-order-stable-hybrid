"""Single-threaded loopback HTTP server.

ORIGIN/CHANGE NOTICE: Adapted from the completed experiment server. Loading now
uses portable manifests, and unexpected 500 responses are deliberately redacted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from strands_decider import schema

from .artifacts import DEFAULT_MODE, MODES
from .hybrid import HybridBundle, load_hybrid, resolve_mode

MAX_BODY_BYTES = 16 * 1024 * 1024


class HybridHTTPServer(HTTPServer):
    bundle: HybridBundle
    default_mode: str


def _artifact_identity(bundle: HybridBundle) -> dict[str, Any]:
    manifest = bundle.artifact.manifest
    return {
        "portable_manifest_sha256": bundle.artifact.portable_manifest_sha256,
        "selected_candidate_ids": dict(manifest["selected_candidate_ids"]),
        "historical_lock_sha256": dict(manifest["historical_lock_sha256"]),
        "upstream": dict(manifest["upstream"]),
    }


def create_server(
    artifact: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    attention_implementation: str | None = "eager",
    default_mode: str = DEFAULT_MODE,
    loader: Callable[..., HybridBundle] = load_hybrid,
) -> HybridHTTPServer:
    """Create a serialized loopback server and load the hybrid once."""
    if host != "127.0.0.1":
        raise ValueError("server must bind exactly to loopback 127.0.0.1")
    selected_default = resolve_mode(default_mode)
    bundle = loader(
        artifact,
        device=device,
        attention_implementation=attention_implementation,
    )
    if bundle.engine_kind != "HybridOptionBranchEngine":
        raise RuntimeError("loader returned an unexpected engine kind")

    class Handler(BaseHTTPRequestHandler):
        server: HybridHTTPServer

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/health":
                self._json(404, {"detail": "not found"})
                return
            identity = _artifact_identity(self.server.bundle)
            self._json(
                200,
                {
                    "status": "ok",
                    "engine": self.server.bundle.engine_kind,
                    "modes": list(MODES),
                    "default_mode": self.server.default_mode,
                    **identity,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            if parsed.path != "/v1/systemone":
                self._json(404, {"detail": "not found"})
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ValueError("request body size is invalid")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise TypeError("request body must be a JSON object")
                request_payload = dict(payload)
                body_mode = request_payload.pop("mode", None)
                query_modes = parse_qs(parsed.query, keep_blank_values=True).get("mode", [])
                if len(query_modes) > 1:
                    raise ValueError("mode query parameter must appear at most once")
                query_mode = query_modes[0] if query_modes else None
                if body_mode is not None and query_mode is not None:
                    if resolve_mode(str(body_mode)) != resolve_mode(query_mode):
                        raise ValueError("body and query modes disagree")
                mode = resolve_mode(
                    str(body_mode)
                    if body_mode is not None
                    else query_mode or self.server.default_mode
                )
                request = schema.SystemOneRequest.model_validate(request_payload)
                started = time.perf_counter()
                response = self.server.bundle.engine.evaluate(request, mode=mode)
                validated = schema.SystemOneResponse.model_validate(response.model_dump())
                body = validated.model_dump()
                identity = _artifact_identity(self.server.bundle)
                body.update(
                    {
                        "mode": mode,
                        "portable_manifest_sha256": identity["portable_manifest_sha256"],
                        "selected_candidate_id": identity["selected_candidate_ids"][mode],
                        "historical_lock_sha256": identity["historical_lock_sha256"],
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                )
                self._json(200, body)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(422, {"detail": str(exc)})
            except Exception as exc:
                if type(exc).__module__.startswith("pydantic"):
                    self._json(422, {"detail": str(exc)})
                else:
                    self._json(500, {"detail": "internal server error"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = HybridHTTPServer((host, port), Handler)
    server.bundle = bundle
    server.default_mode = selected_default
    return server


def serve(
    artifact: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    attention_implementation: str | None = "eager",
    default_mode: str = DEFAULT_MODE,
) -> None:
    server = create_server(
        artifact,
        host=host,
        port=port,
        device=device,
        attention_implementation=attention_implementation,
        default_mode=default_mode,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
