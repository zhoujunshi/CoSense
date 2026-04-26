from __future__ import annotations

import math
from dataclasses import dataclass, replace

from projection import GridCell, GridDefinition, Point2D, ProjectedVehicleObject

TAU = 2.0 * math.pi


@dataclass(frozen=True)
class AngularSpan:
    start_rad: float
    end_rad: float

    @property
    def width_rad(self) -> float:
        if self.end_rad >= self.start_rad:
            return self.end_rad - self.start_rad
        return (math.pi - self.start_rad) + (self.end_rad + math.pi)


@dataclass(frozen=True)
class GridVisibilityResult:
    grid_id: str
    ground_truth_label: int | None
    ideal_span: AngularSpan
    occlusion_spans: tuple[AngularSpan, ...]
    visibility_score: float
    in_fov_frame_count: int


def build_visibility_objects(frames, projected_objects, config: dict):
    visibility = config["visibility"]
    viewpoints_by_frame = {
        frame.source_name: Point2D(x=frame.camera.position.x, y=frame.camera.position.y)
        for frame in frames
        if frame.source_name is not None
    }
    return build_visibility_blocker_objects(
        projected_objects,
        viewpoints_by_frame=viewpoints_by_frame,
        road_heading_deg=float(config["projection"]["road_heading_deg"]),
        geometry_mode=str(visibility["blocker_geometry"]),
        coarse_width_m=float(visibility["coarse_width_m"]),
        coarse_length_m=float(visibility["coarse_length_m"]),
        range_inflate_start_m=float(visibility["range_inflate_start_m"]),
        range_inflate_end_m=float(visibility["range_inflate_end_m"]),
        range_extra_width_m=float(visibility["range_extra_width_m"]),
        range_extra_length_m=float(visibility["range_extra_length_m"]),
    )


def build_visibility_blocker_objects(
    projected_objects: tuple[ProjectedVehicleObject, ...],
    *,
    viewpoints_by_frame: dict[str, Point2D],
    road_heading_deg: float,
    geometry_mode: str = "current_blocker_geometry",
    coarse_width_m: float = 2.6,
    coarse_length_m: float = 5.85,
    range_inflate_start_m: float = 15.0,
    range_inflate_end_m: float = 60.0,
    range_extra_width_m: float = 1.0,
    range_extra_length_m: float = 2.0,
) -> tuple[ProjectedVehicleObject, ...]:
    if geometry_mode == "current_blocker_geometry":
        return projected_objects

    road_heading_rad = math.radians(road_heading_deg)
    transformed = []
    for obj in projected_objects:
        if geometry_mode == "road_aligned_coarse_proxy":
            length_m = coarse_length_m
            width_m = coarse_width_m
        elif geometry_mode == "road_aligned_range_inflated_proxy":
            viewpoint = viewpoints_by_frame.get(obj.source_frame_name)
            if viewpoint is None:
                length_m = coarse_length_m
                width_m = coarse_width_m
            else:
                distance_m = _distance(viewpoint, obj.center_world)
                scale = _linear_range_scale(distance_m, range_inflate_start_m, range_inflate_end_m)
                length_m = coarse_length_m + range_extra_length_m * scale
                width_m = coarse_width_m + range_extra_width_m * scale
        else:
            raise ValueError(f"Unsupported visibility blocker geometry mode: {geometry_mode}")

        transformed.append(
            replace(
                obj,
                heading_rad=road_heading_rad,
                length_m=length_m,
                width_m=width_m,
                footprint_world=_rectangular_footprint(
                    center=obj.center_world,
                    heading_rad=road_heading_rad,
                    length_m=length_m,
                    width_m=width_m,
                ),
            )
        )
    return tuple(transformed)


def compute_case_visibility(frames, grids: GridDefinition, projected_objects: tuple[ProjectedVehicleObject, ...], config: dict):
    scores_by_grid = {cell.grid_id: [] for cell in grids.cells}
    in_fov_scores_by_grid = {cell.grid_id: [] for cell in grids.cells}
    grid_by_id = {cell.grid_id: cell for cell in grids.cells}
    first_result_by_grid = {}
    objects_by_frame: dict[str, list[ProjectedVehicleObject]] = {}
    for record in projected_objects:
        objects_by_frame.setdefault(record.source_frame_name, []).append(record)

    aggregation = str(config["visibility"]["aggregation"])
    min_in_fov = int(config["visibility"]["min_in_fov_frame_count"])
    use_distance_ordering = bool(config["visibility"]["use_distance_ordering"])

    for frame in frames:
        if frame.source_name is None:
            continue
        frame_objects = tuple(objects_by_frame.get(frame.source_name, []))
        viewpoint = Point2D(x=frame.camera.position.x, y=frame.camera.position.y)
        frame_results = compute_grid_visibility(
            viewpoint=viewpoint,
            grids=grids,
            occluding_vehicles=frame_objects,
            camera_yaw_deg=frame.camera.yaw,
            camera_fov_deg=frame.camera.fov,
            use_distance_ordering=use_distance_ordering,
        )
        for result in frame_results:
            scores_by_grid[result.grid_id].append(result.visibility_score)
            if _grid_intersects_camera_fov(
                viewpoint=viewpoint,
                camera_yaw_deg=frame.camera.yaw,
                camera_fov_deg=frame.camera.fov,
                grid=grid_by_id[result.grid_id],
            ):
                in_fov_scores_by_grid[result.grid_id].append(result.visibility_score)
            first_result_by_grid.setdefault(result.grid_id, result)

    averaged = []
    for cell in grids.cells:
        base = first_result_by_grid[cell.grid_id]
        score = _aggregate_visibility(
            all_values=scores_by_grid[cell.grid_id],
            in_fov_values=in_fov_scores_by_grid[cell.grid_id],
            mode=aggregation,
            min_in_fov_frame_count=min_in_fov,
        )
        averaged.append(replace(base, occlusion_spans=tuple(), visibility_score=score))
        averaged[-1] = replace(averaged[-1], in_fov_frame_count=len(in_fov_scores_by_grid[cell.grid_id]))
    return tuple(averaged)


def compute_grid_visibility(
    *,
    viewpoint: Point2D,
    grids: GridDefinition,
    occluding_vehicles: tuple[ProjectedVehicleObject, ...],
    camera_yaw_deg: float | None = None,
    camera_fov_deg: float | None = None,
    use_distance_ordering: bool = True,
) -> tuple[GridVisibilityResult, ...]:
    occluders = tuple(
        (
            vehicle.object_id,
            _distance(viewpoint, vehicle.center_world),
            angular_span_from_points(viewpoint, vehicle.footprint_world),
        )
        for vehicle in occluding_vehicles
    )
    fov_span = _camera_fov_span(camera_yaw_deg, camera_fov_deg)
    results = []
    for grid in grids.cells:
        ideal_span = angular_span_from_points(viewpoint, _grid_corners(grid))
        visible_ideal_span = _clip_span_to_span(ideal_span, fov_span) if fov_span is not None else ideal_span
        if visible_ideal_span is None:
            results.append(
                GridVisibilityResult(
                    grid_id=grid.grid_id,
                    ground_truth_label=grid.ground_truth_label,
                    ideal_span=ideal_span,
                    occlusion_spans=tuple(),
                    visibility_score=0.0,
                    in_fov_frame_count=0,
                )
            )
            continue

        grid_distance = _distance(viewpoint, grid.center)
        candidate_spans = []
        for _, occluder_distance, occluder_span in occluders:
            if use_distance_ordering and occluder_distance >= grid_distance:
                continue
            candidate_spans.append(occluder_span)
        relevant_occlusions = tuple(_clip_span_to_span(span, visible_ideal_span) for span in candidate_spans)
        relevant_occlusions = tuple(span for span in relevant_occlusions if span is not None)
        merged_relevant = merge_angular_spans(relevant_occlusions)
        occluded_width = _intersection_width(visible_ideal_span, merged_relevant)
        ideal_width = ideal_span.width_rad
        visible_width = max(0.0, visible_ideal_span.width_rad - occluded_width)
        visibility_score = 0.0 if ideal_width <= 0.0 else visible_width / ideal_width
        results.append(
            GridVisibilityResult(
                grid_id=grid.grid_id,
                ground_truth_label=grid.ground_truth_label,
                ideal_span=ideal_span,
                occlusion_spans=merged_relevant,
                visibility_score=visibility_score,
                in_fov_frame_count=1,
            )
        )
    return tuple(results)


def angular_span_from_points(viewpoint: Point2D, points: tuple[Point2D, ...]) -> AngularSpan:
    if not points:
        raise ValueError("At least one point is required to compute an angular span.")
    angles = sorted(_angle_0_tau(viewpoint, point) for point in points)
    if len(angles) == 1:
        angle = _normalize_pi(angles[0])
        return AngularSpan(start_rad=angle, end_rad=angle)
    gaps = []
    for index, angle in enumerate(angles):
        next_angle = angles[(index + 1) % len(angles)]
        if index == len(angles) - 1:
            next_angle += TAU
        gaps.append((next_angle - angle, index))
    _, largest_gap_index = max(gaps, key=lambda item: item[0])
    start = angles[(largest_gap_index + 1) % len(angles)]
    end = angles[largest_gap_index]
    if end < start:
        end += TAU
    return AngularSpan(start_rad=_normalize_pi(start), end_rad=_normalize_pi(end))


def merge_angular_spans(spans: tuple[AngularSpan, ...]) -> tuple[AngularSpan, ...]:
    intervals = []
    for span in spans:
        intervals.extend(_span_to_intervals(span))
    if not intervals:
        return tuple()
    intervals.sort(key=lambda item: item[0])

    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    if len(merged) > 1 and merged[0][0] <= 0.0 and merged[-1][1] >= TAU:
        first_start, first_end = merged.pop(0)
        last_start, last_end = merged.pop(-1)
        merged.insert(0, (last_start, first_end + TAU))

    return tuple(_interval_to_span(start, end) for start, end in merged)


def _aggregate_visibility(*, all_values: list[float], in_fov_values: list[float], mode: str, min_in_fov_frame_count: int) -> float:
    if mode == "mean_all_frames":
        return sum(all_values) / len(all_values) if all_values else 0.0
    if mode == "mean_in_fov_frames":
        return sum(in_fov_values) / len(in_fov_values) if in_fov_values else 0.0
    if mode == "max_in_fov_frames":
        return max(in_fov_values) if in_fov_values else 0.0
    if mode == "mean_in_fov_min_count":
        if len(in_fov_values) < min_in_fov_frame_count:
            return 0.0
        return sum(in_fov_values) / len(in_fov_values)
    raise ValueError(f"Unsupported visibility aggregation mode: {mode}")


def _grid_intersects_camera_fov(*, viewpoint: Point2D, camera_yaw_deg: float | None, camera_fov_deg: float | None, grid: GridCell) -> bool:
    if camera_yaw_deg is None or camera_fov_deg is None:
        return True
    center = math.radians(camera_yaw_deg)
    half = math.radians(camera_fov_deg) * 0.5
    fov_span = AngularSpan(
        start_rad=_normalize_pi(center - half),
        end_rad=_normalize_pi(center + half),
    )
    ideal_span = angular_span_from_points(viewpoint, _grid_corners(grid))
    return _clip_span_to_span(ideal_span, fov_span) is not None


def _camera_fov_span(camera_yaw_deg: float | None, camera_fov_deg: float | None) -> AngularSpan | None:
    if camera_yaw_deg is None or camera_fov_deg is None:
        return None
    center = math.radians(camera_yaw_deg)
    half_width = math.radians(camera_fov_deg) * 0.5
    return AngularSpan(
        start_rad=_normalize_pi(center - half_width),
        end_rad=_normalize_pi(center + half_width),
    )


def _linear_range_scale(distance_m: float, start_m: float, end_m: float) -> float:
    if distance_m <= start_m:
        return 0.0
    if distance_m >= end_m:
        return 1.0
    return (distance_m - start_m) / (end_m - start_m)


def _grid_corners(grid: GridCell) -> tuple[Point2D, ...]:
    heading_rad = math.radians(grid.heading or 0.0)
    forward = (math.cos(heading_rad), math.sin(heading_rad))
    lateral = (-math.sin(heading_rad), math.cos(heading_rad))
    half_length = 0.5 * grid.length
    half_width = 0.5 * grid.width
    offsets = (
        (forward[0] * half_length + lateral[0] * half_width, forward[1] * half_length + lateral[1] * half_width),
        (forward[0] * half_length - lateral[0] * half_width, forward[1] * half_length - lateral[1] * half_width),
        (-forward[0] * half_length - lateral[0] * half_width, -forward[1] * half_length - lateral[1] * half_width),
        (-forward[0] * half_length + lateral[0] * half_width, -forward[1] * half_length + lateral[1] * half_width),
    )
    return tuple(Point2D(x=grid.center.x + dx, y=grid.center.y + dy) for dx, dy in offsets)


def _rectangular_footprint(*, center: Point2D, heading_rad: float, length_m: float, width_m: float) -> tuple[Point2D, ...]:
    forward = (math.cos(heading_rad), math.sin(heading_rad))
    lateral = (-math.sin(heading_rad), math.cos(heading_rad))
    half_length = 0.5 * length_m
    half_width = 0.5 * width_m
    offsets = (
        (forward[0] * half_length + lateral[0] * half_width, forward[1] * half_length + lateral[1] * half_width),
        (forward[0] * half_length - lateral[0] * half_width, forward[1] * half_length - lateral[1] * half_width),
        (-forward[0] * half_length - lateral[0] * half_width, -forward[1] * half_length - lateral[1] * half_width),
        (-forward[0] * half_length + lateral[0] * half_width, -forward[1] * half_length + lateral[1] * half_width),
    )
    return tuple(Point2D(x=center.x + dx, y=center.y + dy) for dx, dy in offsets)


def _clip_span_to_span(span: AngularSpan, container: AngularSpan) -> AngularSpan | None:
    intersections = []
    for start_a, end_a in _span_to_intervals(span):
        for start_b, end_b in _span_to_intervals(container):
            start = max(start_a, start_b)
            end = min(end_a, end_b)
            if end > start:
                intersections.append((start, end))
    if not intersections:
        return None
    merged = merge_angular_spans(tuple(_interval_to_span(start, end) for start, end in intersections))
    if not merged:
        return None
    if len(merged) == 1:
        return merged[0]
    return max(merged, key=lambda item: item.width_rad)


def _intersection_width(span: AngularSpan, spans: tuple[AngularSpan, ...]) -> float:
    width = 0.0
    for candidate in spans:
        for start_a, end_a in _span_to_intervals(span):
            for start_b, end_b in _span_to_intervals(candidate):
                width += max(0.0, min(end_a, end_b) - max(start_a, start_b))
    return width


def _span_to_intervals(span: AngularSpan) -> list[tuple[float, float]]:
    start = _normalize_0_tau(span.start_rad)
    end = _normalize_0_tau(span.end_rad)
    if span.width_rad == 0.0:
        return [(start, start)]
    if end >= start:
        return [(start, end)]
    return [(start, TAU), (0.0, end)]


def _interval_to_span(start: float, end: float) -> AngularSpan:
    return AngularSpan(start_rad=_normalize_pi(start), end_rad=_normalize_pi(end))


def _angle_0_tau(viewpoint: Point2D, point: Point2D) -> float:
    return _normalize_0_tau(math.atan2(point.y - viewpoint.y, point.x - viewpoint.x))


def _distance(point_a: Point2D, point_b: Point2D) -> float:
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


def _normalize_0_tau(angle_rad: float) -> float:
    return angle_rad % TAU


def _normalize_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % TAU - math.pi
