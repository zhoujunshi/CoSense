from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox2D:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class DetectionRecord:
    frame_index: int
    bbox: BoundingBox2D
    confidence: float | None = None
    track_id: int | str | None = None
    class_name: str | None = None


@dataclass(frozen=True)
class DetectionTable:
    records: tuple[DetectionRecord, ...]


@dataclass(frozen=True)
class YoloV8DetectorConfig:
    model_path: str
    confidence_threshold: float = 0.25
    image_size: int = 640
    vehicle_class_names: tuple[str, ...] = ("car", "truck", "bus", "motorcycle")


class YoloV8DetectionAdapter:
    """Thin configurable adapter for YOLOv8 image-based vehicle detection."""

    def __init__(self, config: YoloV8DetectorConfig) -> None:
        self._config = config
        self._model = None

    def detect_image(
        self,
        image_path: str | Path,
        frame_index: int,
        frame_metadata=None,
    ) -> DetectionTable:
        _ = frame_metadata
        model = self._get_model()
        results = model.predict(
            source=str(image_path),
            conf=self._config.confidence_threshold,
            imgsz=self._config.image_size,
            verbose=False,
        )
        if not results:
            return DetectionTable(records=tuple())

        result = results[0]
        names = result.names
        records = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(names[class_id])
            if class_name not in self._config.vehicle_class_names:
                continue
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf.item())
            records.append(
                DetectionRecord(
                    frame_index=frame_index,
                    bbox=BoundingBox2D(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                    confidence=confidence,
                    class_name=class_name,
                )
            )
        return DetectionTable(records=tuple(records))

    def _get_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("ultralytics is required for YOLOv8 detection but is not installed.") from exc
            self._model = YOLO(self._config.model_path)
        return self._model


def build_detector(config: dict) -> YoloV8DetectionAdapter:
    return YoloV8DetectionAdapter(
        YoloV8DetectorConfig(
            model_path=config["paths"]["yolo_model_path"],
            confidence_threshold=float(config["detection"]["target_candidate_confidence"]),
            image_size=int(config["detection"]["image_size"]),
            vehicle_class_names=tuple(config["detection"]["vehicle_class_names"]),
        )
    )
