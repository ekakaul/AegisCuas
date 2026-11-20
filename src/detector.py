import time
from pathlib import Path
import cv2
import yaml
from ultralytics import YOLO

THREAT_LEVELS = {
    "missile": ("CRITICAL", (0, 0, 255)),      # Red
    "drone": ("HIGH", (0, 140, 255)),          # Orange
    "helicopter": ("MEDIUM", (0, 215, 255)),   # Yellow
    "airplane": ("LOW", (255, 180, 0)),        # Cyan-Blue
    "bird": ("IGNORE", (0, 200, 80)),          # Green
}


class AegisDetector:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        weights_path = self.cfg["model"]["weights"]
        if not Path(weights_path).exists():
            weights_path = self.cfg["model"]["base_weights"]

        self.model = YOLO(weights_path)
        self.conf = self.cfg["model"]["conf_threshold"]
        self.iou = self.cfg["model"]["iou_threshold"]
        self.imgsz = self.cfg["model"]["imgsz"]
        self.device = self.cfg["model"]["device"]

    def process_frame(self, frame, conf_override: float = None):
        conf = conf_override if conf_override is not None else self.conf
        t0 = time.perf_counter()

        results = self.model.track(
            source=frame,
            conf=conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            persist=True,
            verbose=False,
        )[0]

        annotated = frame.copy()
        detections = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            ids = (
                results.boxes.id.cpu().numpy().astype(int)
                if results.boxes.id is not None
                else [-1] * len(boxes)
            )

            for box, score, cls_id, track_id in zip(boxes, confs, clss, ids):
                x1, y1, x2, y2 = map(int, box)
                label = self.model.names[cls_id]
                threat, color = THREAT_LEVELS.get(label, ("UNKNOWN", (200, 200, 200)))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = (
                    f"#{track_id} {label.upper()} {score:.2f} [{threat}]"
                    if track_id != -1
                    else f"{label.upper()} {score:.2f} [{threat}]"
                )
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(
                    annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1
                )
                cv2.putText(
                    annotated,
                    tag,
                    (x1 + 3, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                detections.append(
                    {
                        "track_id": int(track_id),
                        "class": label,
                        "confidence": round(float(score), 3),
                        "threat": threat,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        fps = 1.0 / max(1e-6, (time.perf_counter() - t0))
        cv2.putText(
            annotated,
            f"AEGIS-CUAS | FPS: {fps:.1f} | Targets: {len(detections)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )
        return annotated, detections, fps


if __name__ == "__main__":
    detector = AegisDetector()
    cap = cv2.VideoCapture(detector.cfg["video"]["source"])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out, _, _ = detector.process_frame(frame)
        cv2.imshow("AEGIS-CUAS Tactical Feed", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()