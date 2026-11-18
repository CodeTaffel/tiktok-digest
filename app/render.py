"""Turn chosen intervals into one vertical MP4.

ffmpeg only. The timestamps are already decided; this module cuts each one
with its audio, fits the picture to 9:16, and concatenates in the given
order. It does not pick new ranges when a cut fails — that would hide an
impossible request inside a shorter file.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import math
import subprocess
import tempfile
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import ffmpeg

from app.errors import ClipError
from app.select import OUTPUT_FPS

# 1080x1920 is the 9:16 size TikTok publishes. Width and height stay even
# because the pixel format below subsamples chroma in 2x2 blocks.
WIDTH = 1080
HEIGHT = 1920

# Pad, don't crop. Nothing in this app looks at the picture, so a center
# crop could remove the only subject. force_divisible_by keeps the scaled
# frame even before the pad. setsar clears a non-square pixel aspect from
# the phone recording; leaving it would stretch the 9:16 frame in players.
# fps is the same rate selection counted in, so the file duration stays on
# the intervals we promised instead of drifting to another frame grid.
_VIDEO_FILTER = (
    f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease:"
    "force_divisible_by=2,"
    f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
    "setsar=1,"
    f"fps={OUTPUT_FPS},"
    "format=yuv420p"
)


def probe_duration(source: Path) -> float:
    """Return how many seconds of ``source`` can be cut.

    The container duration, not a single stream's, is the timeline when
    the file is complete: a stream can report a shorter value when its
    timestamps start late. That number is a lie once ``mdat`` is cut out
    from under an intact ``moov``. The index still names the old duration,
    and a later encode exits 0 with a shorter picture. The last video
    packet still on disk is the limit selection has to hear about, or it
    will place a clip in samples that are not there.
    """

    info, tallies = _scan_source(source)
    return _limit_to_readable_video(info, tallies, _declared_duration(info))


def render_digest(
    source: Path,
    intervals: Sequence[tuple[float, float]],
) -> bytes:
    """Cut ``intervals`` from ``source`` and return one MP4.

    Intervals are used in list order. An empty or reversed interval is
    refused here because ffmpeg would still exit successfully and write a
    file with no samples, which the client would treat as the digest. A
    file shorter than those intervals is refused for the same reason: a
    truncated source still makes ffmpeg exit 0.
    """

    with tempfile.TemporaryDirectory(prefix="tiktok-digest-") as directory:
        output = Path(directory) / "digest.mp4"

        async def _encode() -> None:
            # The page streams this same iterator. A second cutter would
            # let the download and the tests drift apart.
            async for _percent in iter_cutting_progress(source, intervals, output):
                pass

        asyncio.run(_encode())
        # The directory deletes this file on the way out, including when
        # the encode raised, so the bytes are read only after 100.
        return output.read_bytes()


async def iter_cutting_progress(
    source: Path,
    intervals: Sequence[tuple[float, float]],
    dest: Path,
) -> AsyncIterator[int]:
    """Yield how much of the requested duration ffmpeg has encoded.

    The number is encoded time over the duration the caller asked for,
    not wall time. Samples stay at 99 or below. 100 is yielded only after
    the file at ``dest`` has passed the same length check as a blocking
    render, because that is the moment the bytes are safe to download.
    """

    checked = _require_intervals(intervals)
    has_audio = await asyncio.to_thread(_source_has_audio, source)
    total = sum(end - start for start, end in checked)
    finished = 0.0
    last = -1
    clips: list[Path] = []

    for index, (start, end) in enumerate(checked):
        clip = dest.parent / f"clip-{index:04d}.mp4"
        length = end - start
        command = _progress_command(
            _clip_stream(source, start, end, clip, has_audio=has_audio)
        )
        async for seconds in _ffmpeg_times(command, "cut a fragment"):
            percent = _percent_before_done(finished + min(seconds, length), total)
            if percent != last:
                last = percent
                yield percent
        finished += length
        percent = _percent_before_done(finished, total)
        if percent != last:
            last = percent
            yield percent
        clips.append(clip)

    await asyncio.to_thread(_concat, clips, dest)
    await asyncio.to_thread(_reject_short_output, dest, checked)
    await asyncio.to_thread(_require_ftyp_file, dest)
    yield 100


def _require_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not intervals:
        raise ClipError("Cannot render a digest with no fragments.")
    checked: list[tuple[float, float]] = []
    for start, end in intervals:
        if end <= start:
            raise ClipError(
                "Cannot render a fragment that does not move forward in time."
            )
        checked.append((float(start), float(end)))
    return checked


def _declared_duration(info: Mapping[str, object]) -> float:
    streams = _streams(info)
    if not any(stream.get("codec_type") == "video" for stream in streams):
        raise ClipError("The file has no video stream.")

    container = info.get("format")
    if isinstance(container, dict):
        duration = _read_duration(cast(Mapping[str, object], container))
        if duration is not None:
            return duration

    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        duration = _read_duration(stream)
        if duration is not None:
            return duration

    raise ClipError(
        "Could not read a duration from this file. It needs to be a valid MP4."
    )


def _limit_to_readable_video(
    info: Mapping[str, object],
    tallies: Mapping[int, _PacketTally],
    declared: float,
) -> float:
    """Cap ``declared`` at the last video packet whose bytes are in the file.

    ``ffprobe`` still reports the indexed frame count and the container
    duration after ``mdat`` is cut in half. The tally stops at the cut.
    When that tally is shorter than the index, or the index has no frame
    count and the timeline stops early, the declared duration would let
    selection ask for samples the encoder cannot produce.
    """

    stream = _longest_video(_streams(info))
    if stream is None:
        return declared
    index = stream.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return declared

    tally = tallies.get(index)
    count = 0 if tally is None else tally.count
    indexed = _optional_int(stream.get("nb_frames"))
    if indexed is not None and indexed > 0 and count == 0:
        raise ClipError(
            "The file has no readable video. It needs to be a complete MP4."
        )
    readable_end = None if tally is None else tally.end
    if readable_end is None:
        return declared

    samples_missing = indexed is not None and count < indexed
    # No frame count to compare. A one-frame gap is normal when the
    # container follows a slightly longer audio track; a truncated file
    # stops well before that.
    timeline_cut_short = indexed is None and declared - readable_end > 1 / OUTPUT_FPS
    if (samples_missing or timeline_cut_short) and readable_end + (
        0.5 / OUTPUT_FPS
    ) < declared:
        return readable_end
    return declared


def _longest_video(
    streams: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Return the video stream with the most indexed frames.

    A poster image is its own video stream and is often the first one.
    Measuring that single frame would make a complete recording look
    truncated. The long stream is the one the cut reads.
    """

    best: Mapping[str, object] | None = None
    best_frames = -1
    for stream in streams:
        if stream.get("codec_type") != "video" or _attached_pic(stream):
            continue
        frames = _optional_int(stream.get("nb_frames"))
        count = frames if frames is not None else 0
        if count >= best_frames:
            best = stream
            best_frames = count
    return best


def _attached_pic(stream: Mapping[str, object]) -> bool:
    disposition = stream.get("disposition")
    if not isinstance(disposition, dict):
        return False
    flag = cast(Mapping[str, object], disposition).get("attached_pic")
    return flag == 1 or flag == "1"


def _reject_short_output(
    path: Path,
    intervals: Sequence[tuple[float, float]],
) -> None:
    """Refuse an encode that did not reach the intervals it was given.

    ffmpeg exits 0 when the source runs out of samples mid-cut and writes
    whatever prefix it could decode. That file plays, and it is shorter
    than the digest the client asked for. A finished encode lands on the
    same frame grid as the intervals, so more than half a frame short is
    a failed cut, not a timestamp wobble.
    """

    expected = sum(end - start for start, end in intervals)
    stream = _longest_video(_streams(_probe(path)))
    actual = _read_duration(stream) if stream is not None else None
    # No measured duration is not a zero-length video. Saying 0.000s would
    # send the client to trim a file the encoder did not finish.
    if actual is None:
        raise ClipError("Could not read a duration from the rendered digest.")
    if expected - actual > 0.5 / OUTPUT_FPS:
        raise ClipError(
            "Source is shorter than the requested digest: "
            f"the readable video is {actual:.3f}s, {expected:.3f}s requested."
        )


@dataclass
class _PacketTally:
    """Packets still on disk for one stream, and where the last one ends."""

    count: int = 0
    end: float | None = None


def _scan_source(source: Path) -> tuple[dict[str, object], dict[int, _PacketTally]]:
    """Read format, streams, and packet ends in one ffprobe pass.

    A second probe would demux the file again. The tally keeps a count and
    a last timestamp per stream. Storing every packet would hold more than
    the upload. A truncated ``mdat`` makes the demuxer stop, so the tally
    is the samples still in the file, which the container duration cannot say.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-show_packets",
        "-show_entries",
        "format=duration:stream=index,codec_type,nb_frames,duration:"
        "stream_disposition=attached_pic:packet=stream_index,pts_time,duration_time",
        "-of",
        "default",
        str(source),
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise ClipError(
            "ffprobe is not available, so the source duration cannot be read."
        ) from exc

    stdout_pipe = process.stdout
    stderr_pipe = process.stderr
    assert stdout_pipe is not None
    assert stderr_pipe is not None
    stderr_parts: list[bytes] = []

    def _drain_stderr() -> None:
        stderr_parts.append(stderr_pipe.read())

    # stderr is drained beside stdout. If that pipe fills, ffprobe stops
    # writing the packet lines and the tally never finishes.
    drain = threading.Thread(target=_drain_stderr)
    drain.start()
    output = io.TextIOWrapper(stdout_pipe, encoding="utf-8", errors="replace")
    try:
        info, tallies = _read_ffprobe(output)
    finally:
        output.close()
        code = process.wait()
        drain.join()
    if code != 0:
        message = stderr_parts[0] if stderr_parts else b""
        raise ClipError(_failure("read the source", message))
    return info, tallies


def _read_ffprobe(
    lines: io.TextIOBase,
) -> tuple[dict[str, object], dict[int, _PacketTally]]:
    streams: list[dict[str, object]] = []
    format_info: dict[str, object] = {}
    tallies: dict[int, _PacketTally] = {}
    section: str | None = None
    fields: dict[str, str] = {}
    allowed = {
        "PACKET": {"stream_index", "pts_time", "duration_time"},
        "STREAM": {
            "index",
            "codec_type",
            "nb_frames",
            "duration",
            "DISPOSITION:attached_pic",
        },
        "FORMAT": {"duration"},
    }
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            closing = line.startswith("[/")
            name = line[2:-1] if closing else line[1:-1]
            if name not in allowed:
                continue
            if closing:
                if section == name:
                    _finish_section(name, fields, streams, format_info, tallies)
                section = None
                fields = {}
            else:
                section = name
                fields = {}
            continue
        if section is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in allowed[section] and key not in fields:
            fields[key] = value
    return {"streams": streams, "format": format_info}, tallies


def _finish_section(
    name: str,
    fields: Mapping[str, str],
    streams: list[dict[str, object]],
    format_info: dict[str, object],
    tallies: dict[int, _PacketTally],
) -> None:
    if name == "PACKET":
        _note_packet(fields, tallies)
        return
    if name == "FORMAT":
        duration = fields.get("duration")
        if duration is not None:
            format_info["duration"] = duration
        return
    streams.append(_stream_record(fields))


def _stream_record(fields: Mapping[str, str]) -> dict[str, object]:
    stream: dict[str, object] = {}
    index = _optional_int(fields.get("index"))
    if index is not None:
        stream["index"] = index
    codec = fields.get("codec_type")
    if codec is not None:
        stream["codec_type"] = codec
    if "nb_frames" in fields:
        stream["nb_frames"] = fields["nb_frames"]
    if "duration" in fields:
        stream["duration"] = fields["duration"]
    flag = fields.get("DISPOSITION:attached_pic")
    if flag is not None:
        parsed = _optional_int(flag)
        stream["disposition"] = {"attached_pic": flag if parsed is None else parsed}
    return stream


def _note_packet(fields: Mapping[str, str], tallies: dict[int, _PacketTally]) -> None:
    index = _optional_int(fields.get("stream_index"))
    if index is None:
        return
    tally = tallies.get(index)
    if tally is None:
        tally = _PacketTally()
        tallies[index] = tally
    tally.count += 1
    pts = _finite_float(fields.get("pts_time"))
    if pts is None:
        return
    duration = _finite_float(fields.get("duration_time"))
    if duration is not None and duration > 0:
        pts += duration
    if tally.end is None or pts > tally.end:
        tally.end = pts


def _probe(source: Path) -> Mapping[str, object]:
    try:
        info = ffmpeg.probe(str(source))
    except FileNotFoundError as exc:
        raise ClipError(
            "ffprobe is not available, so the source duration cannot be read."
        ) from exc
    except ffmpeg.Error as exc:
        raise ClipError(_ffmpeg_message("read the source", exc)) from exc
    if not isinstance(info, dict):
        raise ClipError("Could not read the source video.")
    return cast(Mapping[str, object], info)


def _streams(info: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = info.get("streams")
    if not isinstance(raw, list):
        return []
    streams: list[Mapping[str, object]] = []
    for item in cast(list[object], raw):
        if isinstance(item, dict):
            streams.append(cast(Mapping[str, object], item))
    return streams


def _read_duration(item: Mapping[str, object]) -> float | None:
    raw = item.get("duration")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value > 0 and math.isfinite(value):
        return value
    return None


def _finite_float(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _optional_int(raw: object) -> int | None:
    # bool is an int subclass. A flag must not become a frame count.
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return int(raw, 10)
    except ValueError:
        return None


def _source_has_audio(source: Path) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in _streams(_probe(source)))


def _clip_stream(
    source: Path,
    start: float,
    end: float,
    dest: Path,
    *,
    has_audio: bool,
) -> ffmpeg.Stream:
    # -ss on the input seeks near the cut, and because we re-encode,
    # ffmpeg's default accurate seek discards the partial GOP before
    # `start`. Output -t then keeps the fragment the length we selected.
    # Stream copy cannot do that: it can only cut on keyframes, which would
    # change the duration, and it cannot scale or pad.
    inp = ffmpeg.input(str(source), ss=start)
    output_args: dict[str, object] = {
        "t": end - start,
        "vf": _VIDEO_FILTER,
        "vcodec": "libx264",
        "preset": "veryfast",
        "crf": 23,
        "pix_fmt": "yuv420p",
    }
    if has_audio:
        # Every fragment is encoded with the same layout so the concat
        # demuxer can copy packets. A mono source and a stereo source in
        # the same digest would otherwise refuse to join.
        output_args["acodec"] = "aac"
        output_args["audio_bitrate"] = "128k"
        output_args["ar"] = 48000
        output_args["ac"] = 2
    else:
        # Do not invent a silent track. The source had no audio to preserve,
        # and a silent track would make the digest a different recording.
        output_args["an"] = None

    return ffmpeg.output(inp, str(dest), **output_args)


def _progress_command(stream: ffmpeg.Stream) -> list[str]:
    # The extra flags only change how often ffmpeg reports the encoded
    # timestamp. The default is about a second, which would hide the bar
    # on a short digest. They are not part of the output graph: the
    # picture, the audio, and the 9:16 filter stay the blocking command.
    command = [str(arg) for arg in ffmpeg.compile(stream, overwrite_output=True)]
    # Global options belong before the inputs. After the output path, ffmpeg
    # treats a flag as another output and the progress log never appears.
    return [command[0], "-nostats", "-stats_period", "0.2", "-progress", "pipe:1", *command[1:]]


def _percent_before_done(processed: float, total: float) -> int:
    if total <= 0 or processed <= 0 or not math.isfinite(processed):
        return 0
    # Floor, and hold the last point back. A sample on the final frame
    # would otherwise say 100 before concat and the length check.
    ratio = min(processed, total) / total
    return min(99, math.floor(ratio * 100))


def _output_seconds(line: str) -> float | None:
    # out_time_us is the encoded position in microseconds. out_time_ms is
    # the same unit on current ffmpeg (the name is wrong), so it is not read.
    text = line.strip()
    prefix = "out_time_us="
    if not text.startswith(prefix):
        return None
    raw = text[len(prefix) :]
    if not raw or raw == "N/A":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0:
        return None
    return value / 1_000_000


async def _ffmpeg_times(command: list[str], action: str) -> AsyncIterator[float]:
    """Yield encoded seconds from ffmpeg's progress output.

    stdout is the progress log. stderr is drained beside it: if that pipe
    fills, ffmpeg stops writing progress and the bar would freeze. The
    process is killed if the caller stops reading, which is what happens
    when the browser goes away mid-response.
    """

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise ClipError(
            "ffmpeg is not available, so the digest cannot be encoded."
        ) from exc

    assert process.stdout is not None
    assert process.stderr is not None
    stderr_task = asyncio.create_task(process.stderr.read())
    try:
        while True:
            raw = await process.stdout.readline()
            if not raw:
                break
            seconds = _output_seconds(raw.decode("utf-8", errors="replace"))
            if seconds is not None:
                yield seconds
        code = await process.wait()
        err = await stderr_task
        if code != 0:
            raise ClipError(_failure(action, err))
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                process.kill()
            await process.wait()
        if not stderr_task.done():
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task


def _concat(clips: Sequence[Path], dest: Path) -> None:
    # One code path for one fragment and for many. A special case that just
    # renamed the single clip would skip the faststart pass the multi-clip
    # file gets, so a one-fragment digest would be slower to start playing.
    list_file = dest.with_suffix(".txt")
    lines = [f"file '{_escape_concat_path(clip)}'" for clip in clips]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    joined = ffmpeg.input(str(list_file), format="concat", safe=0)
    # Copy, because the fragments were already encoded with matching
    # parameters. Re-encoding the join would not change the picture and
    # would add another generation of loss. faststart moves the moov atom
    # forward so a client can begin playback without the whole file.
    stream = ffmpeg.output(joined, str(dest), c="copy", movflags="+faststart")
    _run(stream, "join the fragments")


def _escape_concat_path(path: Path) -> str:
    # The concat demuxer treats the list file as a tiny script. Quotes in a
    # path would close the file token early. Temp names do not contain
    # quotes today; the escape is what keeps a later path change from
    # becoming a second, unintended input.
    return str(path).replace("'", r"'\''")


def _run(stream: ffmpeg.Stream, action: str) -> None:
    try:
        ffmpeg.run(
            stream,
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True,
        )
    except FileNotFoundError as exc:
        raise ClipError(
            "ffmpeg is not available, so the digest cannot be encoded."
        ) from exc
    except ffmpeg.Error as exc:
        raise ClipError(_ffmpeg_message(action, exc)) from exc


def _require_ftyp(data: bytes) -> None:
    # A non-zero ffmpeg status is not enough: a truncated mux can still
    # exit 0 on some builds. An MP4 we are willing to return starts with an
    # ftyp box; anything else is an error body, not a video.
    if len(data) < 8 or data[4:8] != b"ftyp":
        raise ClipError("ffmpeg did not produce a readable MP4.")


def _require_ftyp_file(path: Path) -> None:
    with path.open("rb") as handle:
        _require_ftyp(handle.read(8))


def _ffmpeg_message(action: str, exc: ffmpeg.Error) -> str:
    return _failure(action, exc.stderr)


def _failure(action: str, stderr: bytes | str | None) -> str:
    # The full stderr is a banner plus a progress log. The last non-empty
    # line is the reason ffmpeg stopped, which is the part a client can act on.
    line = _last_ffmpeg_line(stderr)
    if not line:
        return f"ffmpeg could not {action}."
    return f"ffmpeg could not {action}: {line}"


def _last_ffmpeg_line(stderr: bytes | str | None) -> str:
    text = ""
    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]
