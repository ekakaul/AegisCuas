# Changelog

All notable changes to **AEGIS-CUAS** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha] - 2026-11-20

### Added
- **Unified Aerial Dataset (`datasets/aerial/unified/`):** Integrated and validated 35,363 image/label pairs (28,715 train, 6,648 validation) across 5 standardized classes (`drone`, `bird`, `airplane`, `helicopter`, `missile`).
- **Baseline Detection Model (`models/aegis_v0.1-alpha.pt`):** Trained `yolo26n` for 30 epochs at `640x640` resolution, achieving `0.830 mAP50` (`0.868` precision, `0.781` recall) at `2.4 ms` per-frame inference speed.
- **Core Inference & Tracking Engine (`src/detector.py`):** Implemented `AegisDetector` with ByteTrack target ID persistence, automated threat-level classification (`CRITICAL` to `IGNORE`), and real-time OpenCV HUD overlay.
- **Offline Tactical Console (`dashboard/app.py`):** Built a local Streamlit dashboard supporting Image Upload, Video File playback, Live Webcam feeds, dynamic confidence thresholding, and model checkpoint switching.
- **Automated Verification Suite (`tests/test_pipeline.py`):** Added `pytest` checks validating release artifacts, YAML class schemas, synthetic frame inference, and headless Streamlit UI rendering.