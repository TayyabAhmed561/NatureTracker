"""
Central configuration for the Nature Tracker wildlife camera pipeline.

All paths and tunable constants live here; pipeline modules must not hardcode paths.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env in the project root (this file's directory).
load_dotenv(Path(__file__).resolve().parent / ".env")

# --- Paths (project root = directory containing this file) ---
BASE_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = BASE_DIR / "output"
FRAMES_DIR: Path = OUTPUT_DIR / "frames"
DETECTED_DIR: Path = OUTPUT_DIR / "detected"
CHARTS_DIR: Path = OUTPUT_DIR / "charts"
DATA_DIR: Path = OUTPUT_DIR / "data"

DETECTIONS_CSV: Path = DATA_DIR / "detections.csv"
CLASSIFICATIONS_CSV: Path = DATA_DIR / "classifications.csv"
SESSIONS_CSV: Path = DATA_DIR / "sessions.csv"

# --- Frame extraction ---
# Extract one frame every N seconds of video time (interval in seconds).
SAMPLE_RATE: float = 5.0

# --- Dry-run / testing ---
# When dry-run mode is enabled, only the first N seconds of video are processed.
DRY_RUN_MAX_VIDEO_SECONDS: float = 10 * 60.0

# --- Session construction (Stage 4) ---
SESSION_GAP_SECONDS: float = 30.0
# Map raw ``species`` values from classifications to canonical (species, common_name).
# Keys are matched case-insensitively after strip. Omit entries that should stay distinct
# (e.g. "Eurasian Red Squirrel" is intentionally not listed).
SPECIES_NORMALIZATION: dict[str, tuple[str, str]] = {
    "Cyanocitta cristata": ("Blue Jay", "Blue Jay"),
    "Sciurus carolinensis": ("Eastern Gray Squirrel", "Eastern Gray Squirrel"),
    "Tamias striatus": ("Eastern Chipmunk", "Eastern Chipmunk"),
    "Eastern Red Squirrel": ("American Red Squirrel", "American Red Squirrel"),
}
# --- Analytics (Stage 5) ---
VIDEO_LENGTH_HOURS: int = 8
ROLLING_WINDOW_MINUTES: int = 15
MAX_VIDEO_MINUTES: int = VIDEO_LENGTH_HOURS * 60
# Flag sessions longer than this in the written insights (e.g. extended visits vs. sampling gaps).
ANOMALY_THRESHOLD_SECONDS: float = 300.0

# --- API (Anthropic) ---
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
# Stage 3 — Claude Vision species classification
CLAUDE_CLASSIFICATION_MODEL: str = "claude-opus-4-5"
CLASSIFICATION_ESTIMATED_COST_USD_PER_IMAGE: float = 0.003
CLASSIFICATION_INTER_REQUEST_DELAY_SECONDS: float = 0.5
# Retries after the first failed attempt (three retries = four attempts total).
CLASSIFICATION_API_FAILURE_MAX_RETRIES: int = 3
# Additional 429 attempts after the first rate-limit response (five retries).
CLASSIFICATION_RATE_LIMIT_MAX_RETRIES: int = 5
CLASSIFICATION_MAX_TOKENS: int = 4096
CLASSIFICATION_MAX_REQUESTS_PER_SECOND: float = 5.0

# --- YOLO (Stage 2 detection) ---
YOLO_MODEL_NAME: str = "yolov8s.pt"
YOLO_CONFIDENCE: float = 0.25
# Preferred inference device; detect_animals falls back to CPU if MPS fails or is unavailable.
YOLO_DEVICE: str = "mps"
# COCO IDs 14–23: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe.
YOLO_ANIMAL_CLASS_IDS: tuple[int, ...] = tuple(range(14, 24))
YOLO_PERSON_CLASS_ID: int = 0
# Extra class-name substring matches (case-insensitive) for models / names that include these.
YOLO_WILDLIFE_NAME_KEYWORDS: tuple[str, ...] = (
    "squirrel",
    "chipmunk",
    "rabbit",
    "deer",
    "fox",
    "raccoon",
    "mouse",
    "rat",
    "duck",
    "goose",
)
