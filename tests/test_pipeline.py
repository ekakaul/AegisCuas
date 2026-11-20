import numpy as np
import yaml
from pathlib import Path
from streamlit.testing.v1 import AppTest
from src.detector import AegisDetector


ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT_DIR / "models/aegis_v0.1-alpha.pt"
CONFIG_PATH = ROOT_DIR / "configs/default.yaml"
DATA_YAML_PATH = ROOT_DIR / "datasets/aerial/unified/data.yaml"
APP_PATH = ROOT_DIR / "dashboard/app.py"


def test_module1_artifacts_exist():
    assert WEIGHTS_PATH.exists(), f"Missing locked weights: {WEIGHTS_PATH}"
    assert CONFIG_PATH.exists(), f"Missing config: {CONFIG_PATH}"
    assert DATA_YAML_PATH.exists(), f"Missing dataset yaml: {DATA_YAML_PATH}"


def test_config_and_class_schema():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    expected_classes = {
        0: "drone",
        1: "bird",
        2: "airplane",
        3: "helicopter",
        4: "missile",
    }
    assert cfg["classes"] == expected_classes


def test_detector_end_to_end_smoke():
    detector = AegisDetector(config_path=str(CONFIG_PATH))
    assert len(detector.model.names) == 5
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    annotated, detections, fps = detector.process_frame(dummy_frame, conf_override=0.30)
    assert annotated.shape == (640, 640, 3)
    assert isinstance(detections, list)
    assert fps > 0


def test_streamlit_dashboard_renders():
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, f"Dashboard raised an exception: {at.exception}"