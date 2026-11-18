"""HTTP boundary for the digest.

The handler writes the upload to a path, then calls probe, select, and
render. It does not choose timestamps or build a filter graph. ClipError
becomes a client error and no MP4 is returned: a failed request must not
look like a shorter digest. GET / only serves the form that calls this route.

A body over 100 MB is rejected before the encode. The response that does
run is not only the finished MP4: cutting percents from ffmpeg are written
first, so the page can follow that work in the same POST.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import BinaryIO

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from python_multipart.exceptions import FormParserError, MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

from app.errors import ClipError
from app.render import iter_cutting_progress, probe_duration
from app.select import select_intervals

app = FastAPI(title="TikTok digest")

# Next to this module, which is where setuptools places package data.
# The form has to be found after install, not only from a checkout cwd.
_PAGE = Path(__file__).with_name("page.html")

# The page refuses a file larger than this before it sends. The same
# number is enforced here so a client that skips the page cannot leave
# a bigger upload on disk.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# A browser wraps the file in multipart fields. A file of exactly
# MAX_UPLOAD_BYTES arrives a few hundred bytes over that. This is room
# for the wrapper and the two number fields, not for more video.
BODY_OVERHEAD_BYTES = 64 * 1024

# Duration and fragment count are short numbers. A larger field is not
# a value we can cut, and it must not be buffered without a limit.
_MAX_FIELD_BYTES = 128

_TOO_LARGE = "File is larger than 100 MB."

# A percent line is only a few bytes. Chrome will not hand those to the
# page until a larger chunk arrives, which would be the MP4, so the
# cutting bar would jump at the end. The extra line is not a percent;
# the page ignores it.
_PROGRESS_PAD = b"#" * 4096 + b"\n"


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    """Serve the one form that posts to ``/digest``.

    The form is not an API operation, so it stays out of the schema. The
    cut still happens only in ``digest``; this handler does not encode.
    """

    return FileResponse(_PAGE, media_type="text/html; charset=utf-8")


@app.post("/digest")
async def digest(request: Request) -> Response:
    """Cut one MP4 into a vertical digest.

    The body is one multipart form: the MP4, the finished length in
    seconds, and how many pieces to cut. The response starts with cutting
    percents and then the MP4, or a 400 with a plain explanation when the
    source cannot supply that cut. An upload over 100 MB is that kind of
    explanation, and the partial file is removed.
    """

    _reject_declared_body(request)
    directory = Path(tempfile.mkdtemp(prefix="tiktok-digest-"))
    # The streaming response outlives this function. The flag is what
    # keeps the directory until the body has been sent: a `finally` here
    # would delete the upload before ffmpeg opened it.
    handed_off = False
    try:
        source, duration, fragments = await _receive_form(request, directory)
        intervals = await _plan(source, duration, fragments)
        handed_off = True
        return StreamingResponse(
            _digest_body(directory, source, intervals),
            # x-user-defined, not UTF-8. The percents are text and the MP4
            # follows in the same body; a browser's UTF-8 decode would
            # change bytes that are not text. ISO-8859-1 is windows-1252
            # there, which has the same problem for bytes 0x80-0x9F.
            media_type="text/plain; charset=x-user-defined",
            headers={"Content-Disposition": 'attachment; filename="digest.mp4"'},
        )
    finally:
        if not handed_off:
            _delete_directory(directory)


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


def _reject_declared_body(request: Request) -> None:
    raw = request.headers.get("content-length")
    if raw is None:
        return
    try:
        declared = int(raw)
    except ValueError:
        return
    if declared > MAX_UPLOAD_BYTES + BODY_OVERHEAD_BYTES:
        raise HTTPException(status_code=400, detail=_TOO_LARGE)


def _reject_bad_name(filename: str | None) -> None:
    # Suffix, not the content type. Browsers often send octet-stream for a
    # real MP4, while a .txt upload should fail before ffmpeg starts.
    # A renamed non-MP4 still fails when the probe cannot read a duration.
    if Path(filename or "").suffix.lower() != ".mp4":
        raise HTTPException(status_code=400, detail="Upload an MP4 file.")


async def _receive_form(
    request: Request,
    directory: Path,
) -> tuple[Path, float, int]:
    """Stream the form to ``directory / source.mp4`` and stop if it is too big.

    The file is written here, not spooled by the framework first. Stopping
    at the cap is what deletes a partial upload instead of keeping a body
    the page would have refused.
    """

    boundary = _boundary(request)
    source = directory / "source.mp4"
    filename: str | None = None
    fields: dict[str, bytearray] = {}
    file_bytes = 0
    body_bytes = 0
    writing_file = False
    field_name = ""
    header_name = bytearray()
    header_value = bytearray()
    headers: dict[bytes, bytes] = {}
    handle: BinaryIO | None = None

    def on_part_begin() -> None:
        nonlocal headers, header_name, header_value, writing_file, field_name
        headers = {}
        header_name = bytearray()
        header_value = bytearray()
        writing_file = False
        field_name = ""

    def on_header_field(data: bytes, start: int, end: int) -> None:
        header_name.extend(data[start:end])

    def on_header_value(data: bytes, start: int, end: int) -> None:
        header_value.extend(data[start:end])

    def on_header_end() -> None:
        headers[bytes(header_name).lower()] = bytes(header_value).strip()
        header_name.clear()
        header_value.clear()

    def on_headers_finished() -> None:
        nonlocal filename, writing_file, field_name, handle
        _media, options = parse_options_header(headers.get(b"content-disposition"))
        field_name = options.get(b"name", b"").decode("utf-8", errors="replace")
        raw_filename = options.get(b"filename")
        if field_name == "file" and raw_filename is not None:
            if filename is not None:
                raise HTTPException(status_code=400, detail="Upload an MP4 file.")
            filename = raw_filename.decode("utf-8", errors="replace")
            _reject_bad_name(filename)
            writing_file = True
            handle = source.open("wb")
            return
        writing_file = False
        fields[field_name] = bytearray()

    def on_part_data(data: bytes, start: int, end: int) -> None:
        nonlocal file_bytes
        chunk = data[start:end]
        if writing_file:
            file_bytes += len(chunk)
            if file_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail=_TOO_LARGE)
            if handle is None:
                raise HTTPException(status_code=400, detail="Upload an MP4 file.")
            handle.write(chunk)
            return
        buf = fields.get(field_name)
        if buf is None:
            return
        if len(buf) + len(chunk) > _MAX_FIELD_BYTES:
            raise HTTPException(status_code=400, detail="The upload could not be read.")
        buf.extend(chunk)

    def on_part_end() -> None:
        _close_upload()

    def _close_upload() -> None:
        nonlocal handle
        if handle is not None:
            handle.close()
            handle = None

    try:
        parser = MultipartParser(
            boundary,
            {
                "on_part_begin": on_part_begin,
                "on_part_data": on_part_data,
                "on_part_end": on_part_end,
                "on_header_field": on_header_field,
                "on_header_value": on_header_value,
                "on_header_end": on_header_end,
                "on_headers_finished": on_headers_finished,
            },
        )
    except FormParserError as exc:
        raise HTTPException(status_code=400, detail="The upload could not be read.") from exc

    try:
        async for chunk in request.stream():
            body_bytes += len(chunk)
            if body_bytes > MAX_UPLOAD_BYTES + BODY_OVERHEAD_BYTES:
                raise HTTPException(status_code=400, detail=_TOO_LARGE)
            try:
                parser.write(chunk)
            except (MultipartParseError, FormParserError) as exc:
                raise HTTPException(
                    status_code=400, detail="The upload could not be read."
                ) from exc
        try:
            parser.finalize()
        except (MultipartParseError, FormParserError) as exc:
            raise HTTPException(
                status_code=400, detail="The upload could not be read."
            ) from exc
    finally:
        _close_upload()

    if filename is None:
        raise HTTPException(status_code=400, detail="Upload an MP4 file.")
    if file_bytes == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    duration = _parse_duration(_field_text(fields, "duration"))
    fragments = _parse_fragments(_field_text(fields, "fragments"))
    return source, duration, fragments


def _boundary(request: Request) -> bytes:
    header = request.headers.get("content-type")
    if not header:
        raise HTTPException(status_code=400, detail="Upload an MP4 file.")
    media, params = parse_options_header(header)
    if media != b"multipart/form-data":
        raise HTTPException(status_code=400, detail="Upload an MP4 file.")
    boundary = params.get(b"boundary")
    if not boundary:
        raise HTTPException(status_code=400, detail="Upload an MP4 file.")
    return boundary


def _field_text(fields: dict[str, bytearray], name: str) -> str | None:
    raw = fields.get(name)
    if raw is None:
        return None
    return bytes(raw).decode("utf-8", errors="replace")


def _parse_duration(raw: str | None) -> float:
    if raw is None or not raw.strip():
        raise HTTPException(status_code=400, detail="Duration must be a number of seconds.")
    try:
        return float(raw)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Duration must be a number of seconds."
        ) from None


def _parse_fragments(raw: str | None) -> int:
    if raw is None or not raw.strip():
        raise HTTPException(
            status_code=400, detail="Fragment count must be a whole number."
        )
    try:
        return int(raw, 10)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Fragment count must be a whole number."
        ) from None


async def _plan(
    source: Path,
    duration: float,
    fragments: int,
) -> list[tuple[float, float]]:
    # Probe and selection can fail the request with a sentence. Doing that
    # before the response starts keeps the explanation a normal 400: once
    # percents are being written, the status is already 200.
    def choose() -> list[tuple[float, float]]:
        try:
            return select_intervals(probe_duration(source), duration, fragments)
        except ClipError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return await asyncio.to_thread(choose)


async def _digest_body(
    directory: Path,
    source: Path,
    intervals: list[tuple[float, float]],
) -> AsyncIterator[bytes]:
    """Write percents, then ``OK``, then the MP4. Always delete the temp dir.

    100 is the last percent and means the file passed the length check.
    Anything short of that — the client leaving, or ffmpeg stopping — removes
    the upload and the partial encode instead of returning them.
    """

    output = directory / "digest.mp4"
    try:
        finished = False
        async for percent in iter_cutting_progress(source, intervals, output):
            if percent == 100:
                finished = True
            yield f"{percent}\n".encode() + _PROGRESS_PAD
        if finished and output.is_file():
            yield b"OK\n"
            for block in _blocks(output):
                yield block
    except ClipError as exc:
        yield _error_line(str(exc))
    finally:
        _delete_directory(directory)


def _blocks(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return
            yield block


def _error_line(message: str) -> bytes:
    # One line. The page reads text until ERR or OK; a newline inside the
    # sentence would look like another percent.
    flat = " ".join(message.split())
    if not flat:
        flat = "The cut could not be made."
    return f"ERR {flat}\n".encode()


def _delete_directory(directory: Path) -> None:
    # The upload, the clips, and the rendered file all live here. Removing
    # the directory is how a failed request, a finished one, and a client
    # that disconnects leave none of the user's video behind.
    shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
