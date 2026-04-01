"""
Stage 1: extract still frames from a video at a fixed time interval.

Frames are written as JPEG files under the configured output directory, with
filenames encoding the video timestamp (hh:mm:ss).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard for clearer errors
    raise ImportError(
        "opencv-python is required for extract_frames. Install dependencies from requirements.txt."
    ) from exc

# Allow running as script: `python pipeline/extract_frames.py` from project root
if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DRY_RUN_MAX_VIDEO_SECONDS,
    FRAMES_DIR,
    SAMPLE_RATE,
)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger(__name__)


def ensure_output_directories() -> None:
    """
    Create frame output directory and parent output tree if they do not exist.

    Raises:
        OSError: If the directory cannot be created.
    """
    try:
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured frames directory exists: %s", FRAMES_DIR)
    except OSError as exc:
        logger.exception("Failed to create frames directory %s", FRAMES_DIR)
        raise


def video_timestamp_to_filename_stem(seconds: float) -> str:
    """
    Build the filename stem for a frame at the given video time.

    Args:
        seconds: Elapsed time in the video from the start, in seconds.

    Returns:
        Stem like ``frame_00h03m45s`` (zero-padded hours, minutes, seconds).
    """
    total = int(math.floor(seconds + 1e-9))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"frame_{hours:02d}h{minutes:02d}m{secs:02d}s"


def get_video_metadata(cap: cv2.VideoCapture) -> tuple[float, float]:
    """
    Read frames-per-second and duration in seconds from an open capture.

    Args:
        cap: An opened ``cv2.VideoCapture`` instance.

    Returns:
        Tuple of ``(fps, duration_seconds)``. If FPS is missing, defaults to 30.0.
    """
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        logger.warning("Video FPS missing or invalid; assuming 30.0")
        fps = 30.0
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if frame_count > 0 else 0.0
    return fps, duration


def extract_frames(
    video_path: Path,
    *,
    sample_interval_seconds: float = SAMPLE_RATE,
    max_video_seconds: float | None = None,
    frames_dir: Path = FRAMES_DIR,
) -> list[Path]:
    """
    Sample frames from a video at regular time intervals and save as JPEG files.

    Args:
        video_path: Path to the input video file.
        sample_interval_seconds: Time between consecutive samples (seconds of video time).
        max_video_seconds: If set, stop after this many seconds of video (inclusive start).
        frames_dir: Directory to write ``frame_XXhYYmZZs.jpg`` files into.

    Returns:
        List of paths to written frame files, in chronological order.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        ValueError: If ``sample_interval_seconds`` is not positive.
    """
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")

    video_path = video_path.resolve()
    if not video_path.is_file():
        logger.error("Video file not found: %s", video_path)
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ensure_output_directories()

    written: list[Path] = []

    try:
        cap = cv2.VideoCapture(str(video_path))
    except Exception as exc:
        logger.exception("Failed to open video: %s", video_path)
        raise

    if not cap.isOpened():
        logger.error("OpenCV could not open video: %s", video_path)
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps, reported_duration = get_video_metadata(cap)
        effective_limit = max_video_seconds
        if effective_limit is not None and reported_duration > 0:
            effective_limit = min(effective_limit, reported_duration)

        t = 0.0
        while True:
            if effective_limit is not None and t > effective_limit + 1e-6:
                break

            frame_index = int(round(t * fps))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
            except Exception as exc:
                logger.exception("Error seeking/reading at t=%.3fs frame=%d", t, frame_index)
                break

            if not ok or frame is None:
                logger.debug("End of stream or read failed at t=%.3fs (frame %d)", t, frame_index)
                break

            stem = video_timestamp_to_filename_stem(t)
            out_path = frames_dir / f"{stem}.jpg"

            try:
                if not cv2.imwrite(str(out_path), frame):
                    logger.error("cv2.imwrite returned False for %s", out_path)
                    continue
            except Exception as exc:
                logger.exception("Failed to write frame to %s", out_path)
                continue

            written.append(out_path)
            logger.info("Wrote %s (video time ~%s)", out_path.name, stem.replace("frame_", ""))

            t += sample_interval_seconds

            if effective_limit is not None and t > effective_limit + 1e-6:
                break

    finally:
        cap.release()

    logger.info("Extracted %d frame(s) to %s", len(written), frames_dir)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for standalone frame extraction.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace with ``video``, ``dry_run``, and optional overrides.
    """
    parser = argparse.ArgumentParser(
        description="Extract frames from a wildlife camera video at a fixed interval.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Only process the first {DRY_RUN_MAX_VIDEO_SECONDS:.0f} seconds of video.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        metavar="SECONDS",
        help=f"Seconds between frames (default: {SAMPLE_RATE} from config).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for Stage 1.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    args = parse_args(argv)
    interval = args.sample_rate if args.sample_rate is not None else SAMPLE_RATE
    max_sec = DRY_RUN_MAX_VIDEO_SECONDS if args.dry_run else None

    if args.dry_run:
        logger.info(
            "Dry-run mode: limiting extraction to first %.0f seconds of video",
            DRY_RUN_MAX_VIDEO_SECONDS,
        )

    try:
        extract_frames(
            args.video,
            sample_interval_seconds=interval,
            max_video_seconds=max_sec,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during frame extraction")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
