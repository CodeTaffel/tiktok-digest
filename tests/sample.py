"""One sample video for the HTTP tests and the render tests.

Both suites need the same picture and the same way to read its length.
A second copy of the lavfi command would let those checks drift apart.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def write_sample(path: Path, seconds: float) -> None:
    """Write a short synthetic MP4 with a picture and a tone."""

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


def video_duration(path: Path) -> float:
    """Return the video stream duration of ``path``."""

    import ffmpeg

    info = ffmpeg.probe(str(path))
    for stream in info["streams"]:
        if stream.get("codec_type") == "video":
            return float(stream["duration"])
    raise AssertionError("digest has no video stream")
