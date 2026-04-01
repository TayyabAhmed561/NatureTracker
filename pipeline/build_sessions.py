"""
Stage 4: build visit sessions from per-frame species classifications.

Groups consecutive detections of the same species separated by more than
``SESSION_GAP_SECONDS`` and writes ``sessions.csv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CLASSIFICATIONS_CSV,
    SESSIONS_CSV,
    SESSION_GAP_SECONDS,
    SPECIES_NORMALIZATION,
)

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger(__name__)

SKIP_SPECIES: frozenset[str] = frozenset(
    {"parse_error", "api_error", "no_animals_visible"},
)

SESSION_FIELDNAMES = [
    "session_id",
    "species",
    "common_name",
    "arrival_time",
    "departure_time",
    "duration_seconds",
    "duration_formatted",
    "preferred_food",
    "avg_count",
    "peak_count",
    "confidence_avg",
    "concurrent_species",
    "hour_of_video",
    "frame_count",
]


@dataclass
class SessionDraft:
    """Mutable accumulation of rows belonging to one species session."""

    species: str
    common_name: str
    rows: list[dict[str, str]] = field(default_factory=list)

    @property
    def arrival_sec(self) -> int:
        """Seconds of first frame in the session."""
        return parse_timestamp_to_seconds(self.rows[0]["timestamp"])

    @property
    def departure_sec(self) -> int:
        """Seconds of last frame in the session."""
        return parse_timestamp_to_seconds(self.rows[-1]["timestamp"])


def parse_timestamp_to_seconds(ts: str) -> int:
    """
    Convert an ``HH:MM:SS`` timestamp string to seconds from midnight.

    Args:
        ts: Timestamp as in ``classifications.csv``.

    Returns:
        Total seconds; ``0`` if parsing fails.
    """
    parts = (ts or "").strip().split(":")
    if len(parts) != 3:
        return 0
    try:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    except ValueError:
        return 0


def seconds_to_hhmmss(total: int) -> str:
    """
    Format integer seconds since midnight as ``HH:MM:SS``.

    Args:
        total: Non-negative seconds.

    Returns:
        Zero-padded time string.
    """
    total = max(0, int(total))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_duration_human(seconds: int) -> str:
    """
    Build a short human-readable duration like ``4m 30s``.

    Args:
        seconds: Duration in whole seconds.

    Returns:
        Compact English duration string.
    """
    seconds = max(0, int(seconds))
    if seconds == 0:
        return "0s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return " ".join(parts)


def _clean_species_key(raw: object) -> str:
    """
    Normalize a cell value to a stripped species string for lookup.

    Args:
        raw: Value from a ``species`` column (may be NaN or non-string).

    Returns:
        Stripped string, or empty if missing/invalid.
    """
    if raw is None:
        return ""
    if isinstance(raw, float) and raw != raw:
        return ""
    s = str(raw).strip()
    if s.lower() == "nan":
        return ""
    return s


def normalize_classification_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply ``SPECIES_NORMALIZATION`` so duplicate labels merge before session logic.

    Matching is case-insensitive on ``species`` after stripping whitespace.
    Updates ``species`` and ``common_name`` in place.

    Args:
        rows: Raw rows from ``classifications.csv`` (dicts from CSV or DataFrame).

    Returns:
        The same list instance (mutated).
    """
    if not SPECIES_NORMALIZATION:
        return rows
    lookup = {
        k.strip().casefold(): (species, common)
        for k, (species, common) in SPECIES_NORMALIZATION.items()
    }
    changed = 0
    for row in rows:
        sp = _clean_species_key(row.get("species"))
        if not sp:
            continue
        repl = lookup.get(sp.casefold())
        if repl is None:
            continue
        new_sp, new_cn = repl
        row["species"] = new_sp
        row["common_name"] = new_cn
        changed += 1
    if changed:
        logger.info("Normalized species labels for %d classification row(s).", changed)
    return rows


def load_classifications(path: Path) -> list[dict[str, str]]:
    """
    Load classification rows from CSV.

    Args:
        path: Path to ``classifications.csv``.

    Returns:
        List of row dicts.

    Raises:
        OSError: If the file cannot be read.
    """
    if not path.is_file():
        logger.error("Classifications file not found: %s", path)
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        logger.exception("Failed to read %s", path)
        raise


def filter_session_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Keep rows suitable for session construction (real species only).

    Args:
        rows: Raw CSV rows.

    Returns:
        Filtered rows sorted by timestamp then species.
    """
    out: list[dict[str, str]] = []
    for row in rows:
        sp = (row.get("species") or "").strip()
        if not sp or sp in SKIP_SPECIES:
            continue
        out.append(row)
    out.sort(
        key=lambda r: (
            parse_timestamp_to_seconds(r.get("timestamp", "")),
            (r.get("species") or "").lower(),
        )
    )
    return out


def build_species_sessions(rows: list[dict[str, str]]) -> list[SessionDraft]:
    """
    Group rows into sessions per species using the session gap threshold.

    Args:
        rows: Filtered, time-sorted classification rows.

    Returns:
        List of session drafts in chronological order of first arrival.
    """
    by_species: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        sp = (row.get("species") or "").strip()
        by_species.setdefault(sp, []).append(row)

    sessions: list[SessionDraft] = []
    for species in sorted(by_species.keys(), key=str.lower):
        species_rows = by_species[species]
        if not species_rows:
            continue
        common0 = (species_rows[0].get("common_name") or "").strip() or species
        current = SessionDraft(species=species, common_name=common0)
        prev_sec = parse_timestamp_to_seconds(species_rows[0].get("timestamp", ""))
        current.rows.append(species_rows[0])

        for row in species_rows[1:]:
            sec = parse_timestamp_to_seconds(row.get("timestamp", ""))
            gap = sec - prev_sec
            cn = (row.get("common_name") or "").strip() or common0
            if gap > SESSION_GAP_SECONDS:
                sessions.append(current)
                current = SessionDraft(species=species, common_name=cn)
            current.rows.append(row)
            if cn and not current.common_name:
                current.common_name = cn
            prev_sec = sec
        sessions.append(current)

    sessions.sort(key=lambda s: (s.arrival_sec, s.species.lower()))
    return sessions


def mode_most_common(values: list[str]) -> str:
    """
    Return the most frequent non-empty string; ties broken alphabetically.

    Args:
        values: String values (e.g. confidence labels).

    Returns:
        Mode string, or empty if none.
    """
    cleaned = [v.strip() for v in values if v and str(v).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    best = max(counts.values())
    candidates = sorted(k for k, v in counts.items() if v == best)
    return candidates[0] if candidates else ""


def preferred_food_for_session(rows: list[dict[str, str]]) -> str:
    """
    Pick the most frequent non-empty ``food_item`` in a session.

    Args:
        rows: Rows belonging to one session.

    Returns:
        Preferred food label or empty string.
    """
    items = [(r.get("food_item") or "").strip() for r in rows]
    items = [i for i in items if i]
    if not items:
        return ""
    counts = Counter(items)
    best = max(counts.values())
    return sorted(k for k, v in counts.items() if v == best)[0]


def finalize_sessions(drafts: list[SessionDraft]) -> list[dict[str, Any]]:
    """
    Convert session drafts to output rows with metrics and overlap metadata.

    Args:
        drafts: Ordered session drafts.

    Returns:
        List of dicts ready for ``sessions.csv`` (before ``session_id``).
    """
    intervals: list[tuple[str, int, int]] = [
        (d.species, d.arrival_sec, d.departure_sec) for d in drafts
    ]

    rows_out: list[dict[str, Any]] = []
    for d in drafts:
        arr = d.arrival_sec
        dep = d.departure_sec
        duration = max(0, dep - arr)
        counts = []
        for r in d.rows:
            try:
                counts.append(int(float(r.get("count", 0) or 0)))
            except (TypeError, ValueError):
                counts.append(0)
        avg_c = sum(counts) / len(counts) if counts else 0.0
        peak_c = max(counts) if counts else 0
        confs = [r.get("confidence", "") or "" for r in d.rows]
        conc: set[str] = set()
        for sp, a, b in intervals:
            if sp == d.species:
                continue
            if arr <= b and a <= dep:
                conc.add(sp)
        concurrent_sorted = ", ".join(sorted(conc, key=str.lower))
        hour = min(7, max(0, arr // 3600))

        cn_list = [(r.get("common_name") or "").strip() for r in d.rows]
        cn_list = [c for c in cn_list if c]
        common_display = mode_most_common(cn_list) if cn_list else (d.common_name or d.species)

        rows_out.append(
            {
                "species": d.species,
                "common_name": common_display,
                "arrival_time": seconds_to_hhmmss(arr),
                "departure_time": seconds_to_hhmmss(dep),
                "duration_seconds": duration,
                "duration_formatted": format_duration_human(duration),
                "preferred_food": preferred_food_for_session(d.rows),
                "avg_count": round(avg_c, 3),
                "peak_count": peak_c,
                "confidence_avg": mode_most_common(confs),
                "concurrent_species": concurrent_sorted,
                "hour_of_video": hour,
                "frame_count": len(d.rows),
            }
        )
    return rows_out


def assign_session_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add monotonic ``session_id`` keys to session rows.

    Args:
        rows: Session dicts without ``session_id``.

    Returns:
        Same rows with ``session_id`` from ``1`` upward.
    """
    for i, row in enumerate(rows, start=1):
        row["session_id"] = i
    return rows


def write_sessions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write sessions to CSV.

    Args:
        path: Destination ``sessions.csv``.

        rows: Rows including ``session_id``.

    Raises:
        OSError: On write failure.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SESSION_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in SESSION_FIELDNAMES})
    except OSError:
        logger.exception("Failed to write %s", path)
        raise


def run_build_sessions() -> dict[str, Any]:
    """
    Build ``sessions.csv`` from ``classifications.csv``.

    Returns:
        Summary statistics dict.
    """
    raw = load_classifications(CLASSIFICATIONS_CSV)
    normalize_classification_rows(raw)
    filtered = filter_session_rows(raw)
    if not filtered:
        logger.warning("No classification rows to build sessions from.")
        write_sessions_csv(SESSIONS_CSV, [])
        stats = {
            "total_sessions": 0,
            "avg_duration": 0.0,
            "longest": ("", 0),
            "most_sessions_species": ("", 0),
            "peak_concurrent": 0,
        }
        print_session_summary(stats)
        return stats

    drafts = build_species_sessions(filtered)
    rows = finalize_sessions(drafts)
    rows = assign_session_ids(rows)
    try:
        write_sessions_csv(SESSIONS_CSV, rows)
    except OSError:
        raise

    total = len(rows)
    durations = [int(r["duration_seconds"]) for r in rows]
    avg_d = sum(durations) / total if total else 0.0
    longest = max(rows, key=lambda r: int(r["duration_seconds"]))
    species_session_counts = Counter(r["species"] for r in rows)
    top_visitor, top_n = species_session_counts.most_common(1)[0] if rows else ("", 0)
    peak_conc = 0
    for r in rows:
        others = [s for s in (r.get("concurrent_species") or "").split(",") if s.strip()]
        peak_conc = max(peak_conc, 1 + len(others))

    stats = {
        "total_sessions": total,
        "avg_duration": avg_d,
        "longest": (str(longest.get("species", "")), int(longest.get("duration_seconds", 0))),
        "most_sessions_species": (top_visitor, top_n),
        "peak_concurrent": peak_conc,
    }
    logger.info("Wrote %d session(s) to %s", total, SESSIONS_CSV)
    print_session_summary(stats)
    return stats


def print_session_summary(stats: dict[str, Any]) -> None:
    """
    Log and print Stage 4 summary statistics.

    Args:
        stats: Output of :func:`run_build_sessions`.
    """
    lines = [
        "--- Session build summary ---",
        f"Total sessions: {stats['total_sessions']}",
        f"Average session duration: {stats['avg_duration']:.1f}s",
        (
            f"Longest session: {stats['longest'][0]} "
            f"({stats['longest'][1]}s, {format_duration_human(stats['longest'][1])})"
        ),
        (
            f"Most frequent visitor (by session count): {stats['most_sessions_species'][0]} "
            f"({stats['most_sessions_species'][1]} sessions)"
        ),
        f"Peak concurrent species count: {stats['peak_concurrent']}",
    ]
    for line in lines:
        logger.info("%s", line)
    print()
    for line in lines:
        print(line)
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for Stage 4.

    Args:
        argv: Optional argv for tests.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Build visit sessions from classifications.csv.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry for Stage 4.

    Args:
        argv: Optional argv.

    Returns:
        Exit code.
    """
    parse_args(argv)
    try:
        run_build_sessions()
    except Exception:
        logger.exception("Stage 4 failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
