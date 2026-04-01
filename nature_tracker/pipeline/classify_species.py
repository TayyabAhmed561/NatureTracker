"""
Stage 3: Claude Vision classification for YOLO-screened feeding-station frames.

Reads detections.csv, calls the Anthropic API with base64 JPEGs, and appends
per-animal rows to classifications.csv.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ANTHROPIC_API_KEY,
    CLASSIFICATIONS_CSV,
    CLASSIFICATION_API_FAILURE_MAX_RETRIES,
    CLASSIFICATION_ESTIMATED_COST_USD_PER_IMAGE,
    CLASSIFICATION_INTER_REQUEST_DELAY_SECONDS,
    CLASSIFICATION_MAX_TOKENS,
    CLASSIFICATION_RATE_LIMIT_MAX_RETRIES,
    CLAUDE_CLASSIFICATION_MODEL,
    DETECTIONS_CSV,
    DRY_RUN_MAX_VIDEO_SECONDS,
)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger(__name__)

CLASSIFICATION_SYSTEM_PROMPT = """You are an expert wildlife biologist and naturalist specializing in North American backyard wildlife. You are analyzing frames from a wildlife camera feeding station. The YOLO detector that pre-screened these images is trained on COCO classes and frequently mislabels small mammals — a chipmunk might be labeled 'bear', a squirrel labeled 'elephant', etc. Ignore the YOLO bounding box labels entirely and identify what you actually see.

Respond ONLY with a valid JSON object, no markdown, no explanation, no code blocks. Use this exact schema:
{
  'animals': [
    {
      'species': 'Eastern Chipmunk',
      'common_name': 'Chipmunk',
      'confidence': 'high',
      'food_item': 'sunflower seeds',
      'behavior': 'actively stuffing cheek pouches with sunflower seeds',
      'count': 2
    }
  ],
  'food_items_visible': ['sunflower seeds', 'peanuts'],
  'scene_notes': 'Two chipmunks sharing the feeding stone',
  'human_present': false
}

Rules:
- 'confidence' must be one of: high, medium, low
- 'food_item' is what that specific animal is interacting with, null if unclear
- 'species' should be as specific as possible (e.g. Eastern Chipmunk not just Chipmunk)
- If no animals are visible despite YOLO detection, return animals as empty array
- 'count' is the number of that species visible in the frame"""

CLASSIFICATION_FIELDNAMES = [
    "timestamp",
    "frame_path",
    "species",
    "common_name",
    "confidence",
    "food_item",
    "behavior",
    "count",
    "food_items_visible",
    "scene_notes",
    "human_present",
]

USER_PROMPT_TEXT = (
    "Analyze this camera frame and produce the JSON response exactly as specified in your instructions."
)


def ensure_data_dir() -> None:
    """
    Ensure the data directory and parent output tree exist.

    Raises:
        OSError: If directories cannot be created.
    """
    try:
        CLASSIFICATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Failed to create data directory %s", CLASSIFICATIONS_CSV.parent)
        raise


def timestamp_to_seconds(ts: str) -> int | None:
    """
    Parse a ``HH:MM:SS`` timestamp string into seconds from midnight.

    Args:
        ts: Timestamp as stored in detections.csv.

    Returns:
        Total seconds, or ``None`` if parsing fails.
    """
    parts = ts.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]))
        return h * 3600 + m * 60 + s
    except ValueError:
        return None


def load_detections_rows(csv_path: Path) -> list[dict[str, str]]:
    """
    Load all rows from detections.csv.

    Args:
        csv_path: Path to detections.csv.

    Returns:
        List of row dicts keyed by column name.

    Raises:
        OSError: If the file cannot be read.
    """
    if not csv_path.is_file():
        logger.error("Detections file not found: %s", csv_path)
        return []
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except OSError:
        logger.exception("Failed to read %s", csv_path)
        raise


def row_should_classify(row: dict[str, str]) -> bool:
    """
    Return True if this detection row should be sent to Claude.

    Rows qualify when ``num_animals > 0`` or ``detected_classes`` includes ``person``.

    Args:
        row: A row from detections.csv.

    Returns:
        Whether the row passes the filter.
    """
    try:
        num_animals = int(str(row.get("num_animals", "0") or "0").strip())
    except ValueError:
        num_animals = 0
    raw_classes = row.get("detected_classes", "") or ""
    classes_lower = [c.strip().lower() for c in raw_classes.split(",") if c.strip()]
    has_person = "person" in classes_lower
    return num_animals > 0 or has_person


def load_classified_frame_paths(csv_path: Path) -> set[str]:
    """
    Load normalized frame paths already present in classifications.csv for resume.

    Args:
        csv_path: Path to classifications.csv.

    Returns:
        Set of ``str(Path.resolve())`` for each distinct ``frame_path`` seen.
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
                fp = (row.get("frame_path") or "").strip()
                if fp:
                    try:
                        out.add(str(Path(fp).resolve()))
                    except OSError:
                        out.add(fp)
            return out
    except OSError:
        logger.exception("Could not read %s for resume", csv_path)
        return set()


def normalize_frame_path(path_str: str) -> str:
    """
    Resolve a frame path string to an absolute normalized path.

    Args:
        path_str: Path as stored in detections.csv.

    Returns:
        Resolved path string.
    """
    p = Path(path_str).expanduser()
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


def encode_image_base64_jpeg(image_path: Path) -> str:
    """
    Read a JPEG file and return standard base64-encoded bytes as ASCII string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 string suitable for the Anthropic image block.

    Raises:
        OSError: If the file cannot be read.
    """
    try:
        data = image_path.read_bytes()
    except OSError:
        logger.exception("Failed to read image %s", image_path)
        raise
    return base64.standard_b64encode(data).decode("ascii")


def strip_json_fences(text: str) -> str:
    """
    Remove optional markdown code fences and trim whitespace from a model response.

    Args:
        text: Raw assistant text.

    Returns:
        Stripped text more likely to parse as JSON.
    """
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def parse_classification_json(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Parse the model JSON response.

    Args:
        raw_text: Raw assistant output.

    Returns:
        Tuple of (parsed dict, None) on success, or (None, error message) on failure.
    """
    cleaned = strip_json_fences(raw_text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(obj, dict):
        return None, "root is not an object"
    return obj, None


def _status_code(exc: BaseException) -> int | None:
    """Return HTTP status code from an Anthropic/HTTP error if present."""
    code = getattr(exc, "status_code", None)
    if code is not None:
        return int(code)
    resp = getattr(exc, "response", None)
    if resp is not None:
        return int(getattr(resp, "status_code", 0) or 0) or None
    return None


def call_claude_vision(
    client: Any,
    *,
    model: str,
    system: str,
    b64_jpeg: str,
) -> str:
    """
    Call Claude Vision with retries: up to five 429 retries, then up to three
    general failure retries with exponential backoff.

    Args:
        client: ``anthropic.Anthropic`` client.
        model: Model identifier.
        system: System prompt string.
        b64_jpeg: Base64-encoded JPEG (no data URL prefix).

    Returns:
        Concatenated text blocks from the assistant message.

    Raises:
        Exception: If all retries are exhausted.
    """
    last_error: BaseException | None = None
    general_attempts = CLASSIFICATION_API_FAILURE_MAX_RETRIES + 1

    for gen in range(general_attempts):
        rate_attempts = 0
        max_rate = CLASSIFICATION_RATE_LIMIT_MAX_RETRIES + 1
        while rate_attempts < max_rate:
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=CLASSIFICATION_MAX_TOKENS,
                    system=system,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": b64_jpeg,
                                    },
                                },
                                {"type": "text", "text": USER_PROMPT_TEXT},
                            ],
                        }
                    ],
                )
                parts: list[str] = []
                for block in message.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(getattr(block, "text", "") or "")
                return "".join(parts).strip()
            except Exception as exc:
                last_error = exc
                code = _status_code(exc)
                if code == 429 and rate_attempts < CLASSIFICATION_RATE_LIMIT_MAX_RETRIES:
                    logger.warning(
                        "Rate limited (429); retry %d/%d — %s",
                        rate_attempts + 1,
                        CLASSIFICATION_RATE_LIMIT_MAX_RETRIES,
                        exc,
                    )
                    time.sleep(min(2.0**rate_attempts, 60.0))
                    rate_attempts += 1
                    continue
                if code == 429:
                    logger.warning("Rate limit retries exhausted for this attempt: %s", exc)
                break

        if gen < CLASSIFICATION_API_FAILURE_MAX_RETRIES:
            wait = float(2**gen)
            logger.warning(
                "API call failed; general retry %d/%d after %.1fs: %s",
                gen + 1,
                CLASSIFICATION_API_FAILURE_MAX_RETRIES,
                wait,
                last_error,
            )
            time.sleep(wait)
            continue
        break

    assert last_error is not None
    raise last_error


def append_classification_rows(
    csv_path: Path,
    rows: list[dict[str, Any]],
    *,
    write_header: bool,
) -> None:
    """
    Append classification rows to CSV.

    Args:
        csv_path: classifications.csv path.
        rows: Row dicts with keys matching ``CLASSIFICATION_FIELDNAMES``.
        write_header: Whether to write the header row.

    Raises:
        OSError: On write failure.
    """
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CLASSIFICATION_FIELDNAMES, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in CLASSIFICATION_FIELDNAMES})
    except OSError:
        logger.exception("Failed to append to %s", csv_path)
        raise


def serialize_top_level_fields(data: dict[str, Any]) -> tuple[str, str, str]:
    """
    Serialize top-level JSON fields for repeated CSV columns.

    Args:
        data: Parsed assistant JSON object.

    Returns:
        Tuple ``(food_items_visible_json, scene_notes, human_present_str)``.
    """
    food_items_visible = data.get("food_items_visible", [])
    if isinstance(food_items_visible, list):
        fav_str = json.dumps(food_items_visible, ensure_ascii=False)
    else:
        fav_str = str(food_items_visible)
    scene_notes = str(data.get("scene_notes", "") or "")
    human_present = data.get("human_present", False)
    if isinstance(human_present, bool):
        hp_str = "true" if human_present else "false"
    else:
        hp_str = str(human_present).lower()
    return fav_str, scene_notes, hp_str


def build_rows_from_parsed(
    *,
    timestamp: str,
    frame_path_resolved: str,
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build one CSV row per animal from a parsed JSON object.

    Args:
        timestamp: Video timestamp string.
        frame_path_resolved: Absolute path to the annotated JPEG.
        data: Parsed top-level JSON object.

    Returns:
        List of row dicts ready for CSV writing (empty if ``animals`` is empty).
    """
    fav_str, scene_notes, hp_str = serialize_top_level_fields(data)

    animals = data.get("animals", [])
    rows: list[dict[str, Any]] = []
    if not isinstance(animals, list):
        return rows

    for animal in animals:
        if not isinstance(animal, dict):
            continue
        count_val = animal.get("count", 1)
        try:
            count_int = int(count_val)
        except (TypeError, ValueError):
            count_int = 1
        food_item = animal.get("food_item")
        if food_item is None:
            food_item_str = ""
        else:
            food_item_str = str(food_item)
        rows.append(
            {
                "timestamp": timestamp,
                "frame_path": frame_path_resolved,
                "species": str(animal.get("species", "") or ""),
                "common_name": str(animal.get("common_name", "") or ""),
                "confidence": str(animal.get("confidence", "") or ""),
                "food_item": food_item_str,
                "behavior": str(animal.get("behavior", "") or ""),
                "count": count_int,
                "food_items_visible": fav_str,
                "scene_notes": scene_notes,
                "human_present": hp_str,
            }
        )
    return rows


def build_no_animals_visible_row(
    *,
    timestamp: str,
    frame_path_resolved: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a single CSV row when the model returns an empty ``animals`` array.

    Ensures resume tracking and preserves scene-level fields.

    Args:
        timestamp: Video timestamp string.
        frame_path_resolved: Absolute path to the annotated JPEG.
        data: Parsed top-level JSON object.

    Returns:
        One row dict for classifications.csv.
    """
    fav_str, scene_notes, hp_str = serialize_top_level_fields(data)
    return {
        "timestamp": timestamp,
        "frame_path": frame_path_resolved,
        "species": "no_animals_visible",
        "common_name": "",
        "confidence": "",
        "food_item": "",
        "behavior": "",
        "count": 0,
        "food_items_visible": fav_str,
        "scene_notes": scene_notes,
        "human_present": hp_str,
    }


def run_classifications(
    *,
    dry_run: bool = False,
    skip_confirm: bool = False,
) -> dict[str, Any]:
    """
    Run Stage 3: classify qualifying frames and append to classifications.csv.

    Args:
        dry_run: Only process rows whose timestamp falls within the dry-run window.
        skip_confirm: If False, prompt before spending API budget (unless zero frames).

    Returns:
        Statistics dict for summaries and tests.
    """
    ensure_data_dir()

    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example)."
        )

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError("anthropic package required; install from requirements.txt.") from exc

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    all_rows = load_detections_rows(DETECTIONS_CSV)
    filtered: list[dict[str, str]] = [r for r in all_rows if row_should_classify(r)]

    done_paths = load_classified_frame_paths(CLASSIFICATIONS_CSV)
    pending: list[dict[str, str]] = []
    for r in filtered:
        fp = normalize_frame_path((r.get("frame_path") or "").strip())
        if not fp:
            continue
        if fp in done_paths:
            logger.debug("Skip (already classified): %s", fp)
            continue
        if dry_run:
            sec = timestamp_to_seconds(r.get("timestamp", "") or "")
            if sec is None or sec > DRY_RUN_MAX_VIDEO_SECONDS:
                continue
        pending.append(r)

    n = len(pending)
    est_cost = n * CLASSIFICATION_ESTIMATED_COST_USD_PER_IMAGE
    print()
    print(
        f"Frames queued for Claude Vision: {n} "
        f"(~${est_cost:.2f} USD at ${CLASSIFICATION_ESTIMATED_COST_USD_PER_IMAGE} per image)."
    )
    logger.info(
        "Cost estimate: %d frame(s), ~$%.4f USD @ $%.4f/image",
        n,
        est_cost,
        CLASSIFICATION_ESTIMATED_COST_USD_PER_IMAGE,
    )

    if n > 0 and not skip_confirm:
        try:
            ans = input("Proceed with API calls? [y/N]: ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            logger.info("Aborted by user before API calls.")
            print("Aborted (no API calls made).")
            return {
                "frames_processed": 0,
                "total_animals": 0,
                "species_counter": Counter(),
                "food_counter": Counter(),
                "parse_errors": 0,
                "api_errors": 0,
                "aborted": True,
            }

    csv_exists = CLASSIFICATIONS_CSV.is_file() and CLASSIFICATIONS_CSV.stat().st_size > 0
    write_header = not csv_exists

    frames_processed = 0
    total_animals = 0
    species_counter: Counter[str] = Counter()
    food_counter: Counter[str] = Counter()
    parse_errors = 0
    api_errors = 0
    first_request = True

    pbar = tqdm(pending, desc="Classify species", unit="frame")
    for row in pbar:
        if not first_request:
            time.sleep(CLASSIFICATION_INTER_REQUEST_DELAY_SECONDS)
        first_request = False

        ts = (row.get("timestamp") or "").strip()
        fp_raw = (row.get("frame_path") or "").strip()
        fp_res = normalize_frame_path(fp_raw)
        pbar.set_postfix_str(Path(fp_res).name[:40], refresh=False)

        image_path = Path(fp_res)
        if not image_path.is_file():
            logger.error("Frame file missing, skipping: %s", fp_res)
            continue

        logger.debug("Classifying frame %s", fp_res)

        try:
            b64 = encode_image_base64_jpeg(image_path)
        except OSError:
            continue

        try:
            raw_text = call_claude_vision(
                client,
                model=CLAUDE_CLASSIFICATION_MODEL,
                system=CLASSIFICATION_SYSTEM_PROMPT,
                b64_jpeg=b64,
            )
        except Exception as exc:
            api_errors += 1
            logger.exception("Claude API failed for %s", fp_res)
            err_rows = [
                {
                    "timestamp": ts,
                    "frame_path": fp_res,
                    "species": "api_error",
                    "common_name": "",
                    "confidence": "",
                    "food_item": "",
                    "behavior": str(exc)[:500],
                    "count": 0,
                    "food_items_visible": "",
                    "scene_notes": "",
                    "human_present": "",
                }
            ]
            try:
                append_classification_rows(CLASSIFICATIONS_CSV, err_rows, write_header=write_header)
                write_header = False
                done_paths.add(fp_res)
            except OSError:
                pass
            continue

        parsed, err = parse_classification_json(raw_text)
        frames_processed += 1

        if parsed is None:
            parse_errors += 1
            logger.error("JSON parse failed for %s: %s | raw (truncated): %.500s", fp_res, err, raw_text)
            err_rows = [
                {
                    "timestamp": ts,
                    "frame_path": fp_res,
                    "species": "parse_error",
                    "common_name": "",
                    "confidence": "",
                    "food_item": "",
                    "behavior": (raw_text or "")[:2000],
                    "count": 0,
                    "food_items_visible": "",
                    "scene_notes": "",
                    "human_present": "",
                }
            ]
            try:
                append_classification_rows(CLASSIFICATIONS_CSV, err_rows, write_header=write_header)
                write_header = False
                done_paths.add(fp_res)
            except OSError:
                pass
            continue

        out_rows = build_rows_from_parsed(
            timestamp=ts,
            frame_path_resolved=fp_res,
            data=parsed,
        )
        if not out_rows:
            logger.info("No animals in JSON for frame %s (empty array); writing placeholder row", fp_res)
            out_rows = [
                build_no_animals_visible_row(
                    timestamp=ts,
                    frame_path_resolved=fp_res,
                    data=parsed,
                )
            ]

        for r in out_rows:
            sp = (r.get("species") or "").strip()
            try:
                cnt = int(r.get("count", 0))
            except (TypeError, ValueError):
                cnt = 1
            if sp in ("no_animals_visible", "parse_error", "api_error"):
                cnt = 0
            if sp and sp not in ("no_animals_visible", "parse_error", "api_error"):
                species_counter[sp] += max(cnt, 0)
            total_animals += max(cnt, 0)
            fi = (r.get("food_item") or "").strip()
            if fi and cnt > 0:
                food_counter[fi.lower()] += cnt
            try:
                visible = json.loads(r.get("food_items_visible", "[]") or "[]")
                if isinstance(visible, list):
                    for item in visible:
                        if item:
                            food_counter[str(item).lower()] += 1
            except json.JSONDecodeError:
                pass

        try:
            append_classification_rows(CLASSIFICATIONS_CSV, out_rows, write_header=write_header)
            write_header = False
            done_paths.add(fp_res)
        except OSError:
            continue

        n_animal_rows = sum(
            1
            for r in out_rows
            if (r.get("species") or "").strip() not in ("no_animals_visible", "parse_error", "api_error")
        )
        logger.info(
            "Classified %s — %d animal row(s)",
            Path(fp_res).name,
            n_animal_rows,
        )

    stats = {
        "frames_processed": frames_processed,
        "total_animals": total_animals,
        "species_counter": species_counter,
        "food_counter": food_counter,
        "parse_errors": parse_errors,
        "api_errors": api_errors,
        "aborted": False,
    }
    log_classification_summary(stats)
    return stats


def log_classification_summary(stats: dict[str, Any]) -> None:
    """
    Log and print a Stage 3 summary.

    Args:
        stats: Output of :func:`run_classifications`.
    """
    if stats.get("aborted"):
        return

    species: Counter[str] = stats["species_counter"]
    foods: Counter[str] = stats["food_counter"]
    skip_species = {"parse_error", "api_error", "no_animals_visible"}
    species_filtered = Counter({k: v for k, v in species.items() if k not in skip_species})
    top_species = species_filtered.most_common(20)
    top_foods = foods.most_common(15)

    lines = [
        "--- Classification summary ---",
        f"Total frames processed (API success): {stats['frames_processed']}",
        f"Total individual animal detections (sum of counts): {stats['total_animals']}",
        "Species breakdown: "
        + (", ".join(f"{n} ({c})" for n, c in top_species) if top_species else "(none)"),
        "Most common food items: "
        + (", ".join(f"{n} ({c})" for n, c in top_foods) if top_foods else "(none)"),
        f"Parse errors: {stats['parse_errors']}",
        f"API failures (after retries): {stats.get('api_errors', 0)}",
    ]
    for line in lines:
        logger.info("%s", line)
    print()
    for line in lines:
        print(line)
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for standalone classification.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Classify species in detected frames via Claude Vision (Stage 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only rows with timestamp within the first "
            f"{DRY_RUN_MAX_VIDEO_SECONDS:.0f}s of video (see config)."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive cost confirmation before API calls.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for Stage 3.

    Args:
        argv: Optional argv for tests.

    Returns:
        Exit code.
    """
    args = parse_args(argv)
    try:
        run_classifications(dry_run=args.dry_run, skip_confirm=args.yes)
    except Exception:
        logger.exception("Stage 3 failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
