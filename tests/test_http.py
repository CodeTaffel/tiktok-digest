"""The 100 MB cap, temp cleanup, and cutting percents before the MP4.

Nothing here builds a 100 MB file. A declared Content-Length over the cap
is the rejection a real oversized body hits before it is stored, and a
patched cap checks that a file which crosses the limit is removed.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from app.errors import ClipError
from app.main import BODY_OVERHEAD_BYTES, MAX_UPLOAD_BYTES, app

_TOO_LARGE = "File is larger than 100 MB."


def test_page_states_the_limit() -> None:
    status, headers, body = asyncio.run(_call("GET", "/", b""))
    assert status == 200
    assert b"MP4 only, 100 MB maximum" in body
    assert headers[b"content-type"].startswith(b"text/html")


def test_declared_body_over_100mb_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    made = _track_tempdirs(monkeypatch)
    body = _form(b"not-a-video", "1", "1")
    status, _headers, payload = asyncio.run(
        _call(
            "POST",
            "/digest",
            body,
            content_length=MAX_UPLOAD_BYTES + BODY_OVERHEAD_BYTES + 1,
        )
    )
    assert status == 400
    assert json.loads(payload)["detail"] == _TOO_LARGE
    # The declared size is refused before a temp directory is created.
    assert made == []


def test_file_over_the_cap_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 64)
    made = _track_tempdirs(monkeypatch)
    body = _form(b"x" * 200, "1", "1")
    status, _headers, payload = asyncio.run(_call("POST", "/digest", body))
    assert status == 400
    assert json.loads(payload)["detail"] == _TOO_LARGE
    assert made
    assert all(not Path(path).exists() for path in made)


def test_failed_source_removes_its_temp_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    made = _track_tempdirs(monkeypatch)
    body = _form(b"not-an-mp4", "1", "1")
    status, _headers, payload = asyncio.run(_call("POST", "/digest", body))
    assert status == 400
    assert json.loads(payload)["detail"] != _TOO_LARGE
    assert made
    assert all(not Path(path).exists() for path in made)


def test_cutting_error_is_a_line_and_the_temp_dir_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made = _track_tempdirs(monkeypatch)

    async def explode(
        source: Path,
        intervals: list[tuple[float, float]],
        dest: Path,
    ) -> AsyncIterator[int]:
        yield 10
        raise ClipError("encoder stopped")

    monkeypatch.setattr("app.main.iter_cutting_progress", explode)
    source = _source(2.0)
    body = _form(source, "1", "1")
    status, _headers, payload = asyncio.run(_call("POST", "/digest", body))
    assert status == 200
    assert payload.startswith(b"10\n")
    assert b"ERR encoder stopped\n" in payload
    assert b"\nOK\n" not in payload
    assert made
    assert all(not Path(path).exists() for path in made)


def test_progress_precedes_a_finished_mp4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    made = _track_tempdirs(monkeypatch)
    source = _source(2.0)
    body = _form(source, "1", "2")
    status, headers, payload = asyncio.run(_call("POST", "/digest", body))
    assert status == 200
    assert b"filename=\"digest.mp4\"" in headers[b"content-disposition"]
    head, marker, tail = payload.partition(b"\nOK\n")
    assert marker == b"\nOK\n"
    percents = [int(line) for line in head.split(b"\n") if line.strip().isdigit()]
    assert percents
    assert percents[-1] == 100
    assert percents == sorted(percents)
    assert any(percent < 100 for percent in percents)
    assert tail[4:8] == b"ftyp"
    output = tmp_path / "digest.mp4"
    output.write_bytes(tail)
    assert _video_duration(output) == pytest.approx(1.0, abs=0.05)
    assert made
    assert all(not Path(path).exists() for path in made)


def test_client_disconnect_removes_the_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made = _track_tempdirs(monkeypatch)
    source = _source(4.0)
    body = _form(source, "3", "1")

    async def abandon(message: Message) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            raise OSError("client closed the response")

    with pytest.raises(ClientDisconnect):
        asyncio.run(asyncio.wait_for(_call("POST", "/digest", body, send_hook=abandon), 30))
    assert made
    assert all(not Path(path).exists() for path in made)


def _track_tempdirs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    made: list[str] = []
    real = tempfile.mkdtemp

    def wrapped(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
    ) -> str:
        path = real(suffix=suffix, prefix=prefix, dir=dir)
        made.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", wrapped)
    return made


def _form(content: bytes, duration: str, fragments: str) -> bytes:
    boundary = b"----digest-test"
    file_part = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
        b"Content-Type: video/mp4\r\n\r\n" + content + b"\r\n"
    )
    duration_part = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="duration"\r\n\r\n'
        + duration.encode()
        + b"\r\n"
    )
    fragments_part = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="fragments"\r\n\r\n'
        + fragments.encode()
        + b"\r\n"
    )
    return file_part + duration_part + fragments_part + b"--" + boundary + b"--\r\n"


def _source(seconds: float) -> bytes:
    path = Path(tempfile.mkdtemp(prefix="digest-fixture-")) / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=30:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    data = path.read_bytes()
    path.unlink()
    path.parent.rmdir()
    return data


def _video_duration(path: Path) -> float:
    import ffmpeg

    info = ffmpeg.probe(str(path))
    for stream in info["streams"]:
        if stream.get("codec_type") == "video":
            return float(stream["duration"])
    raise AssertionError("digest has no video stream")


async def _call(
    method: str,
    path: str,
    body: bytes,
    *,
    content_length: int | None = None,
    send_hook: Callable[[Message], Awaitable[None]] | None = None,
) -> tuple[int, dict[bytes, bytes], bytes]:
    length = len(body) if content_length is None else content_length
    headers = [(b"content-length", str(length).encode())]
    if method == "POST":
        headers.append((b"content-type", b"multipart/form-data; boundary=----digest-test"))
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
        "state": {},
    }
    sent: list[Message] = []
    chunks = [body]

    async def receive() -> Message:
        if chunks:
            chunk = chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if send_hook is not None:
            await send_hook(message)
        sent.append(message)

    await app(scope, receive, send)
    status = 0
    response_headers: dict[bytes, bytes] = {}
    payload = b""
    for message in sent:
        if message["type"] == "http.response.start":
            status = int(message["status"])
            for key, value in message.get("headers", []):
                response_headers[key] = value
        elif message["type"] == "http.response.body":
            payload += message.get("body", b"")
    return status, response_headers, payload
