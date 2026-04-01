"""
Stage 2: YOLOv8 animal-related detection on extracted frames.

Loads JPEGs from the frames directory, runs inference, filters to wildlife-relevant
classes (plus person for feeder placement), writes annotated images and detections.csv.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    DATA_DIR,
    DETECTED_DIR,
    DETECTIONS_CSV,
    DRY_RUN_MAX_VIDEO_SECONDS,
    FRAMES_DIR,
    YOLO_ANIMAL_CLASS_IDS,
    YOLO_CONFIDENCE,
    YOLO_MODEL_NAME,
    YOLO_PERSON_CLASS_ID,
    YOLO_WILDLIFE_NAME_KEYWORDS,
    YOLO_DEVICE,
)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger(__name__)

FRAME_STEM_PATTERN = re.compile(
    r"^frame_(\d{2})h(\d{2})m(\d{2})s$",
    re.IGNORECASE,
)


def ensure_output_directories() -> None:
    """
    Create detected-frames and data directories if missing.

    Raises:
        OSError: If a directory cannot be created.
    """
    try:
        DETECTED_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Failed to create output directories")
        raise


def parse_frame_stem(stem: str) -> tuple[str, int] | None:
    """
    Parse a frame filename stem into a display timestamp and total seconds.

    Args:
        stem: Filename without extension, e.g. ``frame_00h03m45s``.

    Returns:
        Tuple ``(timestamp_hhmmss, total_seconds)`` like ``("00:03:45", 225)``,
        or ``None`` if the stem does not match the expected pattern.
    """
    m = FRAME_STEM_PATTERN.match(stem)
    if not m:
        return None
    h, mi, s = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    total = h * 3600 + mi * 60 + s
    return f"{h:02d}:{mi:02d}:{s:02d}", total


def load_processed_basenames(csv_path: Path) -> set[str]:
    """
    Load basenames of annotated frames already recorded for resume support.

    Args:
        csv_path: Path to ``detections.csv``.

    Returns:
        Set of filename basenames (e.g. ``frame_00h03m45s.jpg``) already present.
    """
    if not csv_path.is_file():
        return set()
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "frame_path" not in reader.fieldnames:
                return set()
            out: set[str] = set()
            for row in reader:
                fp = row.get("frame_path", "").strip()
                if fp:
                    out.add(Path(fp).name)
            return out
    except OSError:
        logger.exception("Could not read %s for resume; starting fresh", csv_path)
        return set()


def choose_yolo_device() -> str:
    """
    Pick YOLO device: MPS on Apple Silicon when available, else CPU.

    Returns:
        Device string accepted by Ultralytics (``'mps'`` or ``'cpu'``).
    """
    pref = (YOLO_DEVICE or "mps").lower()
    if pref == "cpu":
        return "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        logger.debug("torch MPS check failed", exc_info=True)
    return "cpu"


def detection_keeps_frame(
    class_id: int,
    class_name: str,
    *,
    animal_ids: set[int],
    person_id: int,
    name_keywords: tuple[str, ...],
) -> bool:
    """
    Return True if this detection is a reason to keep the frame (animal, keyword, or person).

    Args:
        class_id: COCO (or model) class index.
        class_name: Human-readable class name for the index.
        animal_ids: Set of COCO animal class IDs to treat as wildlife.
        person_id: Person class ID (feeder / hand).
        name_keywords: Lowercase substrings; if any appear in ``class_name``, keep.

    Returns:
        Whether this box qualifies for keeping the frame.
    """
    if class_id == person_id:
        return True
    if class_id in animal_ids:
        return True
    lower = class_name.lower()
    return any(kw in lower for kw in name_keywords)


def is_animal_detection(
    class_id: int,
    class_name: str,
    *,
    animal_ids: set[int],
    person_id: int,
    name_keywords: tuple[str, ...],
) -> bool:
    """
    Return True if this detection counts toward ``num_animals`` (excludes person).

    Args:
        class_id: Model class index.
        class_name: Class label string.
        animal_ids: COCO animal IDs.
        person_id: Person ID (never counted as animal).
        name_keywords: Substrings for extra wildlife labels.

    Returns:
        True if the detection is an animal / keyword wildlife match, not a person.
    """
    if class_id == person_id:
        return False
    if class_id in animal_ids:
        return True
    lower = class_name.lower()
    return any(kw in lower for kw in name_keywords)


def draw_detections_bgr(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """
    Draw axis-aligned boxes and labels on a BGR image copy.

    Args:
        image_bgr: Source image in BGR layout.
        boxes_xyxy: Array of shape ``(N, 4)`` with ``x1,y1,x2,y2`` in pixels.
        labels: One text label per box.

    Returns:
        New image array with drawings applied.
    """
    out = image_bgr.copy()
    for (x1, y1, x2, y2), label in zip(boxes_xyxy, labels, strict=True):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            out,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def append_detection_row(csv_path: Path, row: dict[str, Any], write_header: bool) -> None:
    """
    Append one row to detections CSV with a consistent schema.

    Args:
        csv_path: Destination CSV path.
        row: Mapping with keys matching ``DETECTION_FIELDNAMES``.
        write_header: Whether to write the header row (new file).

    Raises:
        OSError: On write failure.
    """
    fieldnames = [
        "timestamp",
        "frame_path",
        "detected_classes",
        "confidence_scores",
        "num_animals",
    ]
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    except OSError:
        logger.exception("Failed to append row to %s", csv_path)
        raise


def run_yolo_predict(
    model: Any,
    source: str | Path,
    *,
    device: str,
    conf: float,
) -> tuple[Any, str]:
    """
    Run YOLO predict with error handling; fall back from MPS to CPU on failure.

    Args:
        model: Loaded Ultralytics YOLO model.
        source: Image path.
        device: Device to try first (typically ``mps`` or ``cpu``).
        conf: Confidence threshold.

    Returns:
        Tuple of (Ultralytics Results list, device string that succeeded).

    Raises:
        RuntimeError: If prediction fails on all attempted devices.
    """
    devices_to_try = [device]
    if device == "mps":
        devices_to_try.append("cpu")

    last_error: Exception | None = None
    for dev in devices_to_try:
        try:
            res = model.predict(
                source=str(source),
                conf=conf,
                device=dev,
                verbose=False,
            )
            if dev == "cpu" and device == "mps":
                logger.info("Using CPU for this and subsequent frames after MPS failure")
            return res, dev
        except Exception as exc:
            last_error = exc
            logger.warning(
                "YOLO predict failed on device=%s (%s): %s",
                dev,
                source,
                exc,
            )

    raise RuntimeError("YOLO predict failed on all devices") from last_error


def collect_frame_paths(
    frames_dir: Path,
    *,
    dry_run: bool,
    max_video_seconds: float,
) -> list[Path]:
    """
    List ``.jpg`` frames under ``frames_dir``, sorted by parsed video time.

    Args:
        frames_dir: Directory containing extracted JPEGs.
        dry_run: If True, only include frames whose timestamp is within ``max_video_seconds``.
        max_video_seconds: Upper bound on video time for dry-run filtering.

    Returns:
        Sorted list of frame paths to process.
    """
    paths = sorted(frames_dir.glob("*.jpg"), key=lambda p: p.name.lower())
    scored: list[tuple[int, Path]] = []
    for p in paths:
        parsed = parse_frame_stem(p.stem)
        if parsed is None:
            logger.warning("Skipping frame with unexpected name: %s", p.name)
            continue
        _ts, secs = parsed
        if dry_run and secs > max_video_seconds:
            continue
        scored.append((secs, p))
    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored]


def run_detection(*, dry_run: bool = False) -> dict[str, Any]:
    """
    Run YOLOv8 on all frames (respecting dry-run window and resume state).

    Args:
        dry_run: Limit to frames within ``DRY_RUN_MAX_VIDEO_SECONDS`` of video time.

    Returns:
        Statistics dict with keys ``total_scanned``, ``frames_with_detections``,
        ``detection_rate_pct``, ``animal_class_counter``, ``all_kept_class_counter``,
        ``device_used``.
    """
    ensure_output_directories()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required. Install dependencies from requirements.txt."
        ) from exc

    animal_ids = set(YOLO_ANIMAL_CLASS_IDS)
    person_id = YOLO_PERSON_CLASS_ID
    keywords = YOLO_WILDLIFE_NAME_KEYWORDS

    frame_paths = collect_frame_paths(
        FRAMES_DIR,
        dry_run=dry_run,
        max_video_seconds=DRY_RUN_MAX_VIDEO_SECONDS,
    )
    processed_names = load_processed_basenames(DETECTIONS_CSV)
    csv_exists = DETECTIONS_CSV.is_file() and DETECTIONS_CSV.stat().st_size > 0
    write_header = not csv_exists

    device = choose_yolo_device()
    logger.info("Loading YOLO model %s on device=%s", YOLO_MODEL_NAME, device)

    try:
        model = YOLO(YOLO_MODEL_NAME)
    except Exception:
        logger.exception("Failed to load YOLO model %s", YOLO_MODEL_NAME)
        raise

    total_scanned = 0
    frames_with_detections = 0
    animal_class_counter: Counter[str] = Counter()
    all_kept_class_counter: Counter[str] = Counter()
    device_used = device

    pbar = tqdm(frame_paths, desc="YOLO detect", unit="frame")
    for frame_path in pbar:
        if frame_path.name in processed_names:
            logger.debug("Skip (already in CSV): %s", frame_path.name)
            continue

        parsed = parse_frame_stem(frame_path.stem)
        if parsed is None:
            continue
        timestamp_str, _secs = parsed

        total_scanned += 1

        try:
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                logger.error("Could not read image: %s", frame_path)
                continue
        except Exception:
            logger.exception("OpenCV failed reading %s", frame_path)
            continue

        try:
            results, used_dev = run_yolo_predict(
                model,
                frame_path,
                device=device_used,
                conf=YOLO_CONFIDENCE,
            )
            if used_dev != device_used:
                logger.info("Inference device set to %s", used_dev)
            device_used = used_dev
        except Exception:
            logger.exception("Inference failed for %s", frame_path)
            continue

        if not results:
            continue
        r0 = results[0]
        names_map: dict[int, str] = dict(r0.names) if getattr(r0, "names", None) else {}

        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            continue

        cls_tensor = boxes.cls
        conf_tensor = boxes.conf
        xyxy_tensor = boxes.xyxy

        keep_indices: list[int] = []
        for i in range(len(boxes)):
            cid = int(cls_tensor[i].item())
            cname = names_map.get(cid, str(cid))
            if detection_keeps_frame(
                cid,
                cname,
                animal_ids=animal_ids,
                person_id=person_id,
                name_keywords=keywords,
            ):
                keep_indices.append(i)

        if not keep_indices:
            continue

        frames_with_detections += 1

        labels_for_draw: list[str] = []
        boxes_np: list[list[float]] = []
        class_names_row: list[str] = []
        confs_row: list[str] = []
        num_animals = 0

        for i in keep_indices:
            cid = int(cls_tensor[i].item())
            conf = float(conf_tensor[i].item())
            cname = names_map.get(cid, str(cid))
            logger.debug("Detection: %s id=%s conf=%.4f file=%s", cname, cid, conf, frame_path.name)
            x1, y1, x2, y2 = xyxy_tensor[i].cpu().numpy().tolist()
            boxes_np.append([x1, y1, x2, y2])
            labels_for_draw.append(f"{cname} {conf:.2f}")
            class_names_row.append(cname)
            confs_row.append(f"{conf:.2f}")
            all_kept_class_counter[cname] += 1
            if is_animal_detection(
                cid,
                cname,
                animal_ids=animal_ids,
                person_id=person_id,
                name_keywords=keywords,
            ):
                num_animals += 1
                animal_class_counter[cname] += 1

        annotated = draw_detections_bgr(
            image_bgr,
            np.array(boxes_np, dtype=np.float32),
            labels_for_draw,
        )
        out_path = DETECTED_DIR / frame_path.name
        try:
            if not cv2.imwrite(str(out_path), annotated):
                logger.error("cv2.imwrite failed for %s", out_path)
                continue
        except Exception:
            logger.exception("Failed to write annotated image %s", out_path)
            continue

        row = {
            "timestamp": timestamp_str,
            "frame_path": str(out_path.resolve()),
            "detected_classes": ",".join(class_names_row),
            "confidence_scores": ",".join(confs_row),
            "num_animals": num_animals,
        }
        try:
            append_detection_row(DETECTIONS_CSV, row, write_header=write_header)
            write_header = False
            processed_names.add(frame_path.name)
        except OSError:
            continue

        logger.info(
            "Kept frame %s classes=[%s] num_animals=%d",
            frame_path.name,
            row["detected_classes"],
            num_animals,
        )

    detection_rate_pct = (
        (100.0 * frames_with_detections / total_scanned) if total_scanned else 0.0
    )

    stats: dict[str, Any] = {
        "total_scanned": total_scanned,
        "frames_with_detections": frames_with_detections,
        "detection_rate_pct": detection_rate_pct,
        "animal_class_counter": animal_class_counter,
        "all_kept_class_counter": all_kept_class_counter,
        "device_used": device_used,
    }
    log_summary(stats)
    return stats


def log_summary(stats: dict[str, Any]) -> None:
    """
    Log and print end-of-run detection summary (INFO + stdout).

    Args:
        stats: Return value from :func:`run_detection`.
    """
    total = stats["total_scanned"]
    kept = stats["frames_with_detections"]
    rate = stats["detection_rate_pct"]
    counter: Counter[str] = stats["all_kept_class_counter"]
    most_common = counter.most_common(10)
    device = stats.get("device_used", "?")

    lines = [
        "--- Detection summary ---",
        f"Total frames scanned: {total}",
        f"Frames with relevant detections: {kept}",
        f"Detection rate: {rate:.2f}%",
        (
            "Most commonly detected classes: "
            + (", ".join(f"{n} ({c})" for n, c in most_common) if most_common else "(none)")
        ),
        f"YOLO device used: {device}",
    ]
    for line in lines:
        logger.info("%s", line)
    print()
    for line in lines:
        print(line)
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for standalone animal detection.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace with ``dry_run`` flag.
    """
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 wildlife-related detection on extracted frames.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only process frames in the first "
            f"{DRY_RUN_MAX_VIDEO_SECONDS:.0f} seconds of video (by filename timestamp)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for Stage 2.

    Args:
        argv: Optional arguments for tests.

    Returns:
        Exit code (0 on success).
    """
    args = parse_args(argv)
    try:
        run_detection(dry_run=args.dry_run)
    except Exception:
        logger.exception("Stage 2 failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
