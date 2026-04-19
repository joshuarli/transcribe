"""HTTP download with connection pooling, chunked streaming, retry backoff, and cancellation."""

import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import urllib3

_CHUNK = 65536
_RETRYABLE = {500, 502, 503, 504}
_pool = urllib3.PoolManager()


def download(url: str, dest: Path, cancel: threading.Event | None = None, *, retries: int = 5) -> None:
    tmp = dest.with_suffix(".tmp")
    success = False
    try:
        for attempt in range(retries + 1):
            if cancel is not None and cancel.is_set():
                return
            try:
                resp = _pool.request("GET", url, preload_content=False)
                try:
                    if resp.status not in (200, 206):
                        if resp.status in _RETRYABLE and attempt < retries:
                            raise urllib3.exceptions.HTTPError(f"HTTP {resp.status}")
                        resp.drain_conn()
                        raise urllib3.exceptions.HTTPError(f"HTTP {resp.status}")
                    with tmp.open("wb") as f:
                        for chunk in resp.stream(_CHUNK):
                            if cancel is not None and cancel.is_set():
                                return
                            f.write(chunk)
                finally:
                    resp.drain_conn()
                tmp.rename(dest)
                success = True
                return
            except urllib3.exceptions.HTTPError:
                if attempt == retries:
                    raise
            except OSError:
                if attempt == retries:
                    raise
            _sleep(2**attempt, cancel)
    finally:
        if not success:
            tmp.unlink(missing_ok=True)


def fetch(
    url: str, *, headers: dict[str, str] | None = None, retries: int = 3
) -> tuple[int, bytes, urllib3.HTTPHeaderDict]:
    """GET url, returning (status, body, response_headers). Retries on transient errors; 304 is returned as-is."""
    for attempt in range(retries + 1):
        try:
            resp = _pool.request("GET", url, headers=headers or {}, preload_content=False)
        except urllib3.exceptions.HTTPError:
            if attempt == retries:
                raise
            _sleep(2**attempt, None)
            continue
        try:
            if resp.status == 304 or resp.status in (200, 206):
                return resp.status, resp.read(), resp.headers
            if resp.status not in _RETRYABLE or attempt == retries:
                raise urllib3.exceptions.HTTPError(f"HTTP {resp.status}")
        finally:
            resp.drain_conn()
        _sleep(2**attempt, None)
    raise urllib3.exceptions.HTTPError("all retries exhausted")


def post_json(url: str, body: object) -> Any:  # noqa: ANN401
    """POST JSON body, return parsed JSON response. Raises on non-200."""
    import json

    data = json.dumps(body).encode()
    resp = _pool.request("POST", url, body=data, headers={"Content-Type": "application/json"})
    if resp.status != 200:
        raise urllib3.exceptions.HTTPError(f"HTTP {resp.status}: {resp.data.decode(errors='replace')}")
    return json.loads(resp.data)


def stream_post_json(url: str, body: object) -> Generator[str]:
    """POST JSON body with stream=True, yield text deltas from SSE response."""
    import json

    data = json.dumps({**body, "stream": True} if isinstance(body, dict) else body).encode()  # type: ignore[operator]
    resp = _pool.request("POST", url, body=data, headers={"Content-Type": "application/json"}, preload_content=False)
    if resp.status != 200:
        raise urllib3.exceptions.HTTPError(f"HTTP {resp.status}: {resp.read().decode(errors='replace')}")
    try:
        buf = b""
        for chunk in resp.stream(_CHUNK):
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    return
                event = json.loads(payload)
                delta = event.get("choices", [{}])[0].get("delta", {}).get("content") or ""
                if delta:
                    yield delta
    finally:
        resp.drain_conn()


def _sleep(seconds: int, cancel: threading.Event | None) -> None:
    if cancel is None:
        time.sleep(seconds)
        return
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cancel.is_set():
            return
        time.sleep(0.1)
