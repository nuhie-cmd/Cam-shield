# Project Overview

**Key Code Files**

- `main.py` – entry point; runs the camera capture loop and invokes the YOLO‑v8 detection model.
- `requirement.txt.txt` – lists required Python packages.
- `yolov8n.pt` – pre‑trained YOLO‑v8‑Nano model weights.
- `backend/` – server‑side scripts (e.g., Flask/FastAPI) exposing APIs for the dashboard and handling inference.
- `dashboard/` – front‑end HTML/JS/CSS displaying live feeds, detection results, and system status.
- `camera_integrity/` – utilities for checking camera health and connectivity.
- `camera_health/` – scripts monitoring and logging camera performance metrics.
- `detection/` – core detection logic wrapping the YOLO model and processing frames.
- `evidence/` – stores snapshots/video clips of detected events.
- `analysis/` – post‑processing tools generating stats and reports from logs.
- `data/` – sample video files for development/testing.
- `test_videos/` – short clips for unit‑testing the detection pipeline.
- `.env` – environment‑variable file with secrets and configuration values.
- `.gitignore` – specifies files/folders to exclude from version control.

