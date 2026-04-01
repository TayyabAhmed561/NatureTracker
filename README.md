# 🦫 Nature Tracker — Wildlife Feeding Station Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Claude](https://img.shields.io/badge/Claude-Vision%20API-orange)
![Dash](https://img.shields.io/badge/Plotly-Dash-teal)
![License](https://img.shields.io/badge/License-MIT-green)

An automated computer vision pipeline that analyzes 8 hours of wildlife camera footage to identify species, track visits, and generate behavioral insights — built as a proof of concept for the Avia retail analytics platform.

## 📸 Dashboard Preview

<!-- Add dashboard screenshot here -->

---

## 🧠 Project Overview

Nature Tracker ingests long-form feeding-station video, samples frames at a fixed interval, runs **YOLOv8** to find animals, sends crops to **Claude Vision** for species and behavior metadata, merges detections into **visit sessions**, and produces **Plotly** analytics plus an interactive **Dash** dashboard. On a full **8-hour** run, the pipeline yielded **5,469** animal detections across **30** unique species, consolidated into **1,303** visit sessions — enough signal for frequency, temporal, food-preference, and co-occurrence views.

---

## ⚙️ Pipeline Architecture

```
Video
  → Frame Extraction
  → YOLO Detection
  → Claude Vision Classification
  → Session Construction
  → Analytics & Charts
  → Dash Dashboard
```

| Stage | Tool | Input | Output |
|-------|------|--------|--------|
| 1 — Frame extraction | OpenCV (`pipeline/extract_frames.py`) | MP4 video | Timestamped JPEGs in `output/frames/` |
| 2 — Detection | Ultralytics YOLOv8 (`pipeline/detect_animals.py`) | Frame images | Cropped detections, `output/data/detections.csv` |
| 3 — Classification | Anthropic Claude Vision (`pipeline/classify_species.py`) | Detection crops | Species labels, `output/data/classifications.csv` |
| 4 — Sessions | Pandas (`pipeline/build_sessions.py`) | Classifications | Visit rows, `output/data/sessions.csv` |
| 5 — Analytics | Plotly (`pipeline/analyze.py`) | Sessions + classifications | Charts in `output/charts/` |
| 6 — Dashboard | Dash + Dash Bootstrap Components (`app.py`) | CSV snapshots | Web UI at port `8050` (or `PORT` on hosting) |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Anthropic API key** (for Stage 3 — classification)
- **(Optional)** Apple Silicon Mac with **MPS** for faster YOLO inference (see `requirements.txt` notes for PyTorch)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-org>/NatureTracker.git
   cd NatureTracker
   ```

2. **Create a virtual environment**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env and set ANTHROPIC_API_KEY=...
   ```

### Running the Pipeline

```bash
# Dry run (first 10 minutes only — recommended first)
python run_pipeline.py --video path/to/video.mp4 --dry-run

# Full run
python run_pipeline.py --video path/to/video.mp4 --yes

# Launch dashboard (from repo root)
python app.py
# Open http://localhost:8050
```

> **Note:** Run pipeline and dashboard commands from the repository root so `config.py` paths and imports resolve correctly.

---

## 📊 Results (from 8-hour run)

| Metric | Value |
|--------|-------|
| Total frames extracted | 5,696 |
| Frames with animals detected | 3,487 (61%) |
| Total animal detections | 5,469 |
| Unique species identified | 30 |
| Total visit sessions | 1,303 |
| Peak activity hour | Hour 5 |
| Most frequent visitor | Black-capped Chickadee (246 sessions) |
| Most popular food | Sunflower seeds (6,423 interactions) |
| Top co-occurring pair | Chickadee + Chipmunk (300 overlaps) |

---

## 🗂️ Project Structure

```
NatureTracker/                    # Git repository root
├── render.yaml                   # Render Blueprint (web service)
├── Procfile
├── app.py                        # Dash dashboard (Stage 6)
├── config.py                     # Central configuration
├── run_pipeline.py               # End-to-end orchestration
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── __init__.py
│   ├── extract_frames.py         # Stage 1
│   ├── detect_animals.py         # Stage 2
│   ├── classify_species.py      # Stage 3
│   ├── build_sessions.py         # Stage 4
│   └── analyze.py                # Stage 5
└── output/                       # Generated artifacts (gitignored as needed)
    ├── frames/
    ├── detected/
    ├── data/
    │   ├── detections.csv
    │   ├── classifications.csv
    │   └── sessions.csv
    └── charts/
```

---

## ⚙️ Configuration

All tunable parameters and paths live in **`config.py`** (loaded at import time; `.env` is read for secrets). Key settings:

| Setting | Role |
|---------|------|
| `SAMPLE_RATE` | Seconds between extracted frames |
| `YOLO_CONFIDENCE` | Minimum detection confidence for YOLO |
| `SESSION_GAP_SECONDS` | Gap used to split/merge visit sessions |
| `CLAUDE_CLASSIFICATION_MODEL` | Claude Vision model id for classification (the pipeline’s **Claude model** setting) |
| `DRY_RUN_MAX_VIDEO_SECONDS` | Video length cap when `--dry-run` is used |

Related: `ANTHROPIC_API_KEY`, `YOLO_MODEL_NAME`, `SPECIES_NORMALIZATION`, and chart windows (`ROLLING_WINDOW_MINUTES`, etc.).

---

## 💰 Cost Estimate

| Component | Cost |
|-----------|------|
| Claude Vision API (full 8 hr run) | ~$40 |
| YOLOv8 inference | $0 (local) |
| Tip: use Claude Haiku instead | ~$4 |

*Figures are approximate; actual spend depends on model, pricing, and frame/detection volume.*

---

## 🔗 Connection to Avia

This project mirrors the **shape** of a retail analytics stack: ingest continuous video, detect entities of interest, enrich with a vision LLM, aggregate into time-bounded “sessions,” and surface KPIs in a dashboard. For **Avia**, you would swap the wildlife detector for a **people / shopper** detector, replace the species prompt with **retail behavior** prompts (dwell, queue, engagement), and wire in **Shopify POS** (or similar) for transaction context — reusing the same staged pipeline and dashboard patterns.

---

## 📄 License

MIT
