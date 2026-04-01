"""
Master script to run pipeline stages end-to-end.

Runs Stages 1–5 (through analytics charts); Stage 6 (Dash) is separate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Project root on sys.path when executed as `python run_pipeline.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DRY_RUN_MAX_VIDEO_SECONDS, SAMPLE_RATE
from pipeline.analyze import run_analyze
from pipeline.build_sessions import run_build_sessions
from pipeline.classify_species import run_classifications
from pipeline.detect_animals import run_detection
from pipeline.extract_frames import extract_frames

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the full pipeline runner.

    Returns:
        Parsed arguments including ``video`` and ``dry_run``.
    """
    parser = argparse.ArgumentParser(
        description="Run Nature Tracker pipeline (frame extraction and later stages).",
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
        help=(
            "Process only the first "
            f"{DRY_RUN_MAX_VIDEO_SECONDS:.0f} seconds of video (testing)."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive API cost confirmation before Stage 3 (classify_species).",
    )
    return parser.parse_args()


def main() -> int:
    """
    Run configured pipeline stages.

    Returns:
        Exit code 0 on success, non-zero on failure.
    """
    args = parse_args()
    max_sec = DRY_RUN_MAX_VIDEO_SECONDS if args.dry_run else None

    if args.dry_run:
        logger.info(
            "Dry-run: processing first %.0f seconds only",
            DRY_RUN_MAX_VIDEO_SECONDS,
        )

    logger.info("Stage 1: frame extraction")
    try:
        extract_frames(
            args.video,
            sample_interval_seconds=SAMPLE_RATE,
            max_video_seconds=max_sec,
        )
    except Exception as exc:
        logger.exception("Stage 1 failed: %s", exc)
        return 1

    logger.info("Stage 1 complete.")

    logger.info("Stage 2: animal detection (YOLOv8)")
    try:
        run_detection(dry_run=args.dry_run)
    except Exception as exc:
        logger.exception("Stage 2 failed: %s", exc)
        return 1

    logger.info("Stage 2 complete.")

    logger.info("Stage 3: species classification (Claude Vision)")
    try:
        run_classifications(dry_run=args.dry_run, skip_confirm=args.yes)
    except Exception as exc:
        logger.exception("Stage 3 failed: %s", exc)
        return 1

    logger.info("Stage 3 complete.")

    logger.info("Stage 4: session construction")
    try:
        run_build_sessions()
    except Exception as exc:
        logger.exception("Stage 4 failed: %s", exc)
        return 1

    logger.info("Stage 4 complete.")

    logger.info("Stage 5: analytics and charts")
    try:
        run_analyze()
    except Exception as exc:
        logger.exception("Stage 5 failed: %s", exc)
        return 1

    logger.info("Stage 5 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
