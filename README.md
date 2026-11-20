# AEGIS-CUAS

**AEGIS-CUAS** is an offline-capable Counter-Unmanned Aerial System (C-UAS) detection, multi-object tracking, and threat-classification pipeline built on YOLO (`yolo26n`), OpenCV, and Streamlit.

---

## Key Capabilities

* **5-Class Aerial Target Detection:** Real-time inference across `drone`, `bird`, `airplane`, `helicopter`, and `missile`.
* **Persistent Multi-Target Tracking & Threat Triage:** Integrated ByteTrack ID persistence with automated threat severity tagging (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `IGNORE`).
* **Offline Tactical Console:** Local Streamlit dashboard supporting static imagery, recorded surveillance video, and live webcam feeds with dynamic confidence tuning.
* **Modular Training & Verification Pipeline:** Unified dataset schema, YAML-driven training configurations, and automated `pytest` verification.

---

## Baseline Performance (`v0.1.0-alpha`)

Trained for 30 epochs at `640x640` resolution (`AdamW`, `lr0=0.002`, `AMP` enabled) on **35,363** unified aerial image/label pairs (28,715 train / 6,648 val) on an NVIDIA GeForce RTX 3050 Ti Laptop GPU:

| Class | Precision (`P`) | Recall (`R`) | `mAP50` | `mAP50-95` |
| :--- | :---: | :---: | :---: | :---: |
| **All Classes** | **0.868** | **0.781** | **0.830** | **0.557** |
| **Helicopter** | 0.948 | 0.975 | 0.983 | 0.740 |
| **Airplane** | 0.931 | 0.881 | 0.930 | 0.809 |
| **Missile** | 0.804 | 0.789 | 0.851 | 0.495 |
| **Bird** | 0.754 | 0.658 | 0.721 | 0.328 |
| **Drone** | 0.906 | 0.602 | 0.662 | 0.415 |

* **Inference Speed:** `2.4 ms` per frame (~285 FPS end-to-end)
* **Model Footprint:** `5.4 MB` (`models/aegis_v0.1-alpha.pt`)

---

## Project Structure

```text
AegisCuas/
├── configs/
│   └── default.yaml              # Model, training, class schema, and video parameters
├── dashboard/
│   └── app.py                    # Local Streamlit tactical surveillance console
├── datasets/
│   └── aerial/unified/           # Unified YOLO train/val images, labels, and data.yaml
├── models/
│   └── aegis_v0.1-alpha.pt       # Locked release model weights
├── scripts/
│   ├── export_model.py           # Model export utilities
│   ├── prepare_dataset.py        # Dataset unification and label normalization
│   └── train.py                  # YOLO training entrypoint
├── src/
│   └── detector.py               # Core inference, ByteTrack tracking, and HUD renderer
├── tests/
│   └── test_pipeline.py          # Automated pytest artifact, inference, and UI tests
├── CHANGELOG.md
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-ui.txt
```

---

## Quickstart

### 1. Environment Setup
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -r requirements-ui.txt
```

### 2. Run Automated Tests
```cmd
pytest -v
```

### 3. Launch Tactical Console
```cmd
streamlit run dashboard\app.py
```

---

## Development Roadmap

* **`v0.1.0-alpha` (Completed):** Unified 5-class aerial baseline dataset (~35.3k pairs), initial `yolo26n` training (`0.830 mAP50`), `src/detector.py` tracking/HUD engine, Streamlit console, and `pytest` suite.
* **`v0.2.0-alpha` (Next):** Small-object and long-range UAV fine-tuning (Module 2) to improve distant `drone` recall above `0.80`.
* **`v0.3.0-alpha` (Planned):** Hard-negative and adverse-condition fine-tuning (Module 3) to suppress false positives in cluttered backgrounds.