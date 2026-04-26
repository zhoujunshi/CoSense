from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detection import DetectionRecord


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraMetadata:
    position: Point3D
    yaw: float | None
    pitch: float | None
    roll: float | None
    fov: float | None
    max_range: float | None
    intrinsics: CameraIntrinsics | None


@dataclass(frozen=True)
class FrameMetadata:
    source_name: str
    timestamp: float
    camera: CameraMetadata


@dataclass(frozen=True)
class GridCell:
    grid_id: str
    center: Point2D
    heading: float
    width: float
    length: float
    ground_truth_label: int | None


@dataclass(frozen=True)
class GridDefinition:
    name: str
    cells: tuple[GridCell, ...]


@dataclass(frozen=True)
class PointStatistics:
    raw_point_count: int
    filtered_point_count: int
    cluster_point_count: int


@dataclass(frozen=True)
class ProjectedVehicleObject:
    object_id: str
    source_frame_name: str
    source_timestamp: float
    source_detection_index: int
    center_world: Point2D
    heading_rad: float
    length_m: float
    width_m: float
    footprint_world: tuple[Point2D, ...]
    confidence: float | None = None
    class_name: str | None = None
    point_statistics: PointStatistics | None = None


def build_grid_definition(config: dict) -> GridDefinition:
    cells = []
    for grid_id, item in config["grid_geometry"].items():
        cells.append(
            GridCell(
                grid_id=str(grid_id),
                center=Point2D(x=float(item["center"]["x"]), y=float(item["center"]["y"])),
                heading=float(item["heading_deg"]),
                width=float(item["width_m"]),
                length=float(item["length_m"]),
                ground_truth_label=int(config["grid_labels"][str(grid_id)]),
            )
        )
    return GridDefinition(name="final-production-grids", cells=tuple(cells))


def load_case_data(case_id: str, config: dict, detector) -> tuple[tuple[FrameMetadata, ...], tuple[ProjectedVehicleObject, ...]]:
    dataset_root = Path(config["paths"]["dataset_root"])
    case_dir = dataset_root / case_id
    metadata_glob = str(config["data"]["metadata_glob"])
    image_glob = str(config["data"]["image_glob"])
    metadata_paths = sorted(case_dir.glob(metadata_glob), key=lambda path: _timestamp_from_name(path.name))
    image_by_name = {path.name: path for path in case_dir.glob(image_glob)}

    frames = tuple(_load_frame_metadata(path) for path in metadata_paths)
    projected_records: list[ProjectedVehicleObject] = []
    projection_cfg = config["projection"]

    for frame_index, frame in enumerate(frames):
        image_name = frame.source_name.replace("_meta.json", "_rgb.png")
        image_path = image_by_name.get(image_name)
        if image_path is None:
            continue
        detections = detector.detect_image(image_path=image_path, frame_index=frame_index, frame_metadata=frame)
        for detection_index, detection in enumerate(detections.records):
            edge_points = _project_detection_bottom_edge_to_world_points(
                frame_metadata=frame,
                detection=detection,
                pixel_stride=int(projection_cfg["pixel_stride"]),
                min_downward_ray_component=float(projection_cfg["min_downward_ray_component"]),
            )
            footprint_estimate = _estimate_vehicle_footprint_from_bottom_edge(
                edge_points_xy=edge_points,
                road_heading_deg=float(projection_cfg["road_heading_deg"]),
                min_vehicle_width_m=float(projection_cfg["min_vehicle_width_m"]),
                max_vehicle_width_m=float(projection_cfg["max_vehicle_width_m"]),
                vehicle_aspect_ratio=float(projection_cfg["vehicle_aspect_ratio"]),
            )
            if footprint_estimate is None:
                continue
            center, heading, length, width, footprint = footprint_estimate
            projected_records.append(
                ProjectedVehicleObject(
                    object_id=f"{case_id}:{frame.source_name}:{detection_index}",
                    source_frame_name=frame.source_name,
                    source_timestamp=frame.timestamp,
                    source_detection_index=detection_index,
                    center_world=center,
                    heading_rad=heading,
                    length_m=length,
                    width_m=width,
                    footprint_world=footprint,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                    point_statistics=PointStatistics(
                        raw_point_count=int(len(edge_points)),
                        filtered_point_count=int(len(edge_points)),
                        cluster_point_count=int(len(edge_points)),
                    ),
                )
            )
    return frames, tuple(projected_records)


def _timestamp_from_name(name: str) -> float:
    stem = name.replace("_meta.json", "")
    try:
        return float(stem)
    except ValueError:
        return math.inf


def _load_frame_metadata(path: Path) -> FrameMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    intrinsics_payload = payload["camera"].get("intrinsics")
    intrinsics = None
    if intrinsics_payload is not None:
        intrinsics = CameraIntrinsics(
            fx=float(intrinsics_payload["fx"]),
            fy=float(intrinsics_payload["fy"]),
            cx=float(intrinsics_payload["cx"]),
            cy=float(intrinsics_payload["cy"]),
        )
    camera = CameraMetadata(
        position=Point3D(
            x=float(payload["camera"]["x"]),
            y=float(payload["camera"]["y"]),
            z=float(payload["camera"]["z"]),
        ),
        yaw=_maybe_float(payload["camera"].get("yaw")),
        pitch=_maybe_float(payload["camera"].get("pitch")),
        roll=_maybe_float(payload["camera"].get("roll")),
        fov=_maybe_float(payload["camera"].get("fov")),
        max_range=_maybe_float(payload["camera"].get("max_range")),
        intrinsics=intrinsics,
    )
    return FrameMetadata(
        source_name=path.name,
        timestamp=float(payload["timestamp"]),
        camera=camera,
    )


def _maybe_float(value) -> float | None:
    return None if value is None else float(value)


def _project_detection_bottom_edge_to_world_points(
    *,
    frame_metadata: FrameMetadata,
    detection: DetectionRecord,
    pixel_stride: int,
    min_downward_ray_component: float,
) -> np.ndarray:
    x1 = max(0, int(math.floor(detection.bbox.x1)))
    x2 = int(math.ceil(detection.bbox.x2))
    if x2 <= x1:
        return np.empty((0, 2), dtype=float)

    pixel_y = max(detection.bbox.y1, detection.bbox.y2 - 1.0)
    world_points = []
    stride = max(1, pixel_stride)
    for pixel_x in range(x1, x2, stride):
        point = _project_pixel_ray_to_ground(
            frame_metadata=frame_metadata,
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            min_downward_ray_component=min_downward_ray_component,
        )
        if point is not None:
            world_points.append((point.x, point.y))
    if not world_points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(world_points, dtype=float)


def _project_pixel_ray_to_ground(
    *,
    frame_metadata: FrameMetadata,
    pixel_x: float,
    pixel_y: float,
    min_downward_ray_component: float,
) -> Point2D | None:
    intrinsics = frame_metadata.camera.intrinsics
    if intrinsics is None:
        raise ValueError("Frame metadata must include camera intrinsics for projection.")

    x_norm = (float(pixel_x) - intrinsics.cx) / intrinsics.fx
    y_norm = (float(pixel_y) - intrinsics.cy) / intrinsics.fy
    ray_local = np.asarray([1.0, x_norm, -y_norm], dtype=float)
    ray_world = _rotation_world_from_local(frame_metadata) @ ray_local
    ray_world_z = float(ray_world[2])
    camera_position = frame_metadata.camera.position
    downward_component = -ray_world_z
    if downward_component < min_downward_ray_component:
        return None

    t = -camera_position.z / ray_world_z
    if t <= 0.0:
        return None

    world_x = camera_position.x + t * float(ray_world[0])
    world_y = camera_position.y + t * float(ray_world[1])
    return Point2D(x=float(world_x), y=float(world_y))


def _estimate_vehicle_footprint_from_bottom_edge(
    *,
    edge_points_xy: np.ndarray,
    road_heading_deg: float,
    min_vehicle_width_m: float,
    max_vehicle_width_m: float,
    vehicle_aspect_ratio: float,
) -> tuple[Point2D, float, float, float, tuple[Point2D, ...]] | None:
    if len(edge_points_xy) == 0:
        return None

    center_array = edge_points_xy.mean(axis=0)
    center = Point2D(x=float(center_array[0]), y=float(center_array[1]))
    vehicle_heading = math.radians(road_heading_deg)
    lateral_heading = _normalize_angle(vehicle_heading + math.pi / 2.0)
    lateral_axis = np.asarray([math.cos(lateral_heading), math.sin(lateral_heading)], dtype=float)
    lateral_positions = edge_points_xy @ lateral_axis
    width_raw = float(np.percentile(lateral_positions, 95.0) - np.percentile(lateral_positions, 5.0))
    width_m = float(np.clip(width_raw, min_vehicle_width_m, max_vehicle_width_m))
    length_m = vehicle_aspect_ratio * width_m
    footprint = _rectangular_footprint(center, vehicle_heading, length_m, width_m)
    return center, vehicle_heading, length_m, width_m, footprint


def _rectangular_footprint(
    center: Point2D,
    heading_rad: float,
    length_m: float,
    width_m: float,
) -> tuple[Point2D, ...]:
    forward = np.asarray([math.cos(heading_rad), math.sin(heading_rad)], dtype=float)
    lateral = np.asarray([-math.sin(heading_rad), math.cos(heading_rad)], dtype=float)
    center_array = np.asarray([center.x, center.y], dtype=float)
    half_length = 0.5 * length_m
    half_width = 0.5 * width_m
    offsets = (
        forward * half_length + lateral * half_width,
        forward * half_length - lateral * half_width,
        -forward * half_length - lateral * half_width,
        -forward * half_length + lateral * half_width,
    )
    return tuple(Point2D(x=float((center_array + offset)[0]), y=float((center_array + offset)[1])) for offset in offsets)


def _normalize_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _rotation_world_from_local(frame_metadata: FrameMetadata) -> np.ndarray:
    yaw_rad = math.radians(frame_metadata.camera.yaw or 0.0)
    pitch_rad = math.radians(frame_metadata.camera.pitch or 0.0)
    roll_rad = math.radians(frame_metadata.camera.roll or 0.0)
    return _rotation_z(yaw_rad) @ _rotation_y(pitch_rad) @ _rotation_x(roll_rad)


def _rotation_x(angle_rad: float) -> np.ndarray:
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_angle, -sin_angle],
            [0.0, sin_angle, cos_angle],
        ],
        dtype=float,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return np.asarray(
        [
            [cos_angle, 0.0, sin_angle],
            [0.0, 1.0, 0.0],
            [-sin_angle, 0.0, cos_angle],
        ],
        dtype=float,
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return np.asarray(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
