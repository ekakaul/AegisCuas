import yaml
from PIL import Image
import ultralytics.data.dataset as dataset_mod
from ultralytics import YOLO

# 1. Pre-initialize PIL image plugins safely on the main thread
Image.init()

# 2. Prevent multi-threaded PIL C-extension race condition on Python 3.15 Windows
dataset_mod.NUM_THREADS = 1


def train():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    t_cfg = cfg["training"]
    m_cfg = cfg["model"]

    model = YOLO(m_cfg["base_weights"])

    model.train(
        data=t_cfg["data_yaml"],
        epochs=t_cfg["epochs"],
        imgsz=m_cfg["imgsz"],
        batch=t_cfg["batch"],
        device=m_cfg["device"],
        amp=t_cfg["amp"],
        workers=t_cfg["workers"],
        patience=t_cfg["patience"],
        project=t_cfg["project"],
        name=t_cfg["name"],
    )


if __name__ == "__main__":
    train()