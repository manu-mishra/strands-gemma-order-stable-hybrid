from __future__ import annotations

import http.client
import json
import threading
from types import SimpleNamespace

import pytest
from strands_decider import schema

from strands_gemma_order_stable_hybrid.artifacts import (
    HISTORICAL_LOCK_HASHES,
    PORTABLE_MANIFEST_SHA256,
    SELECTED_CANDIDATE_IDS,
    UPSTREAM_IDENTITIES,
)
from strands_gemma_order_stable_hybrid.server import HybridHTTPServer, create_server


class FakeEngine:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.modes = []

    def evaluate(self, request, *, mode=None):
        self.modes.append(mode)
        if self.fail:
            raise RuntimeError("backend failure detail")
        return schema.SystemOneResponse(
            model="fake",
            answers={},
            usage=schema.Usage(input_tokens=1, output_tokens=0),
        )


def _bundle(engine):
    return SimpleNamespace(
        engine=engine,
        engine_kind="HybridOptionBranchEngine",
        artifact=SimpleNamespace(
            portable_manifest_sha256=PORTABLE_MANIFEST_SHA256,
            manifest={
                "selected_candidate_ids": SELECTED_CANDIDATE_IDS,
                "historical_lock_sha256": HISTORICAL_LOCK_HASHES,
                "upstream": UPSTREAM_IDENTITIES,
            },
        ),
    )


def _request(server, method, path, payload=None):
    thread = threading.Thread(target=server.handle_request)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    body = None if payload is None else json.dumps(payload)
    headers = {} if body is None else {"content-type": "application/json"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    thread.join(timeout=3)
    return result


def _payload(mode=None):
    payload = {
        "state": "state",
        "questions": {
            "pick": {
                "type": "choice",
                "instructions": "Pick.",
                "criteria": {"left": "L", "right": "R"},
            }
        },
    }
    if mode is not None:
        payload["mode"] = mode
    return payload


def test_loopback_serial_server_modes_health_and_single_load(tmp_path) -> None:
    calls = 0
    engine = FakeEngine()

    def loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return _bundle(engine)

    server = create_server(tmp_path, port=0, device="cpu", loader=loader)
    try:
        assert type(server) is HybridHTTPServer
        assert calls == 1
        status, health = _request(server, "GET", "/health")
        assert status == 200
        assert health == {
            "status": "ok",
            "engine": "HybridOptionBranchEngine",
            "modes": ["normal", "compare_choices"],
            "default_mode": "normal",
            "portable_manifest_sha256": PORTABLE_MANIFEST_SHA256,
            "selected_candidate_ids": SELECTED_CANDIDATE_IDS,
            "historical_lock_sha256": HISTORICAL_LOCK_HASHES,
            "upstream": UPSTREAM_IDENTITIES,
        }
        status, body = _request(server, "POST", "/v1/systemone", _payload())
        assert status == 200 and body["mode"] == "normal"
        assert body["portable_manifest_sha256"] == PORTABLE_MANIFEST_SHA256
        assert body["selected_candidate_id"] == SELECTED_CANDIDATE_IDS["normal"]
        assert body["historical_lock_sha256"] == HISTORICAL_LOCK_HASHES
        assert isinstance(body["latency_ms"], int | float) and body["latency_ms"] >= 0
        assert "artifact_sha256" not in body
        status, body = _request(server, "POST", "/v1/systemone?mode=compare_choices", _payload())
        assert status == 200 and body["mode"] == "compare_choices"
        assert body["portable_manifest_sha256"] == PORTABLE_MANIFEST_SHA256
        assert body["selected_candidate_id"] == SELECTED_CANDIDATE_IDS["compare_choices"]
        assert body["historical_lock_sha256"] == HISTORICAL_LOCK_HASHES
        assert isinstance(body["latency_ms"], int | float) and body["latency_ms"] >= 0
        assert engine.modes == ["normal", "compare_choices"]
    finally:
        server.server_close()


def test_server_rejects_non_loopback_and_mode_conflict(tmp_path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        create_server(
            tmp_path, host="0.0.0.0", loader=lambda *args, **kwargs: _bundle(FakeEngine())
        )
    server = create_server(tmp_path, port=0, loader=lambda *args, **kwargs: _bundle(FakeEngine()))
    try:
        status, _ = _request(
            server,
            "POST",
            "/v1/systemone?mode=normal",
            _payload("compare_choices"),
        )
        assert status == 422
    finally:
        server.server_close()


def test_unexpected_500_is_redacted(tmp_path) -> None:
    server = create_server(
        tmp_path, port=0, loader=lambda *args, **kwargs: _bundle(FakeEngine(fail=True))
    )
    try:
        status, body = _request(server, "POST", "/v1/systemone", _payload())
        assert status == 500
        assert body == {"detail": "internal server error"}
        assert "backend failure detail" not in json.dumps(body)
    finally:
        server.server_close()
