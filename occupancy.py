from __future__ import annotations

import math
from dataclasses import dataclass

from projection import GridCell, GridDefinition, ProjectedVehicleObject


@dataclass(frozen=True)
class FrameGridOverlap:
    frame_name: str
    matched_object_id: str | None
    overlap_ratio: float


@dataclass(frozen=True)
class GridOccupancyResult:
    grid_id: str
    ground_truth_label: int | None
    matched_object_ids: tuple[str, ...]
    per_frame_overlap_ratios: tuple[FrameGridOverlap, ...]
    occupancy_score: float
    hit_frame_count: int


def compute_case_occupancy(grids: GridDefinition, projected_objects: tuple[ProjectedVehicleObject, ...], config: dict):
    occupancy = config["occupancy"]
    return compute_grid_occupancy_selected_candidates(
        grids=grids,
        projected_objects=projected_objects,
        aggregation=str(occupancy["temporal_aggregation"]),
        distance_threshold_m=float(occupancy["distance_threshold_m"]),
        selection_rule=str(occupancy["target_selector"]),
    )


def compute_grid_occupancy_selected_candidates(
    *,
    grids: GridDefinition,
    projected_objects: tuple[ProjectedVehicleObject, ...],
    aggregation: str = "max",
    distance_threshold_m: float = 4.0,
    selection_rule: str = "grid_centered_road_prior",
) -> tuple[GridOccupancyResult, ...]:
    objects_by_frame: dict[str, list[ProjectedVehicleObject]] = {}
    for vehicle in projected_objects:
        objects_by_frame.setdefault(vehicle.source_frame_name, []).append(vehicle)

    per_grid_rows: dict[str, list[FrameGridOverlap]] = {grid.grid_id: [] for grid in grids.cells}
    matched_ids_by_grid: dict[str, list[str]] = {grid.grid_id: [] for grid in grids.cells}

    for frame_name, frame_objects in sorted(objects_by_frame.items()):
        matched_pairs = _nearest_center_distance_matches(
            grids=grids.cells,
            projected_objects=tuple(frame_objects),
            distance_threshold_m=distance_threshold_m,
        )
        for grid in grids.cells:
            selected = _select_grid_candidate(
                grid=grid,
                frame_objects=tuple(frame_objects),
                selection_rule=selection_rule,
            )
            matched_object_id = None
            if selected is not None and matched_pairs.get(grid.grid_id) == selected.object_id:
                matched_object_id = selected.object_id
                matched_ids_by_grid[grid.grid_id].append(matched_object_id)
            per_grid_rows[grid.grid_id].append(
                FrameGridOverlap(
                    frame_name=frame_name,
                    matched_object_id=matched_object_id,
                    overlap_ratio=1.0 if matched_object_id is not None else 0.0,
                )
            )

    results = []
    for grid in grids.cells:
        frame_rows = per_grid_rows[grid.grid_id]
        occupancy_score = aggregate_occupancy_score(frame_rows, mode=aggregation)
        results.append(
            GridOccupancyResult(
                grid_id=grid.grid_id,
                ground_truth_label=grid.ground_truth_label,
                matched_object_ids=tuple(matched_ids_by_grid[grid.grid_id]),
                per_frame_overlap_ratios=tuple(frame_rows),
                occupancy_score=occupancy_score,
                hit_frame_count=sum(1 for row in frame_rows if row.matched_object_id is not None),
            )
        )
    return tuple(results)


def _nearest_center_distance_matches(
    *,
    grids: tuple[GridCell, ...],
    projected_objects: tuple[ProjectedVehicleObject, ...],
    distance_threshold_m: float,
) -> dict[str, str]:
    candidate_pairs: list[tuple[float, int, str, str]] = []
    for object_index, vehicle in enumerate(projected_objects):
        best_distance = None
        best_grid_id = None
        for grid in grids:
            distance = math.hypot(vehicle.center_world.x - grid.center.x, vehicle.center_world.y - grid.center.y)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_grid_id = grid.grid_id
        if best_distance is not None and best_grid_id is not None and best_distance < distance_threshold_m:
            candidate_pairs.append((best_distance, object_index, best_grid_id, vehicle.object_id))

    candidate_pairs.sort(key=lambda item: item[0])
    used_objects: set[int] = set()
    used_grids: set[str] = set()
    matches: dict[str, str] = {}
    for _, object_index, grid_id, object_id in candidate_pairs:
        if object_index in used_objects or grid_id in used_grids:
            continue
        used_objects.add(object_index)
        used_grids.add(grid_id)
        matches[grid_id] = object_id
    return matches


def _select_grid_candidate(
    *,
    grid: GridCell,
    frame_objects: tuple[ProjectedVehicleObject, ...],
    selection_rule: str,
) -> ProjectedVehicleObject | None:
    if not frame_objects:
        return None
    if selection_rule == "grid_centered":
        return min(
            frame_objects,
            key=lambda item: math.hypot(item.center_world.x - grid.center.x, item.center_world.y - grid.center.y),
        )
    if selection_rule == "grid_centered_road_prior":
        return min(frame_objects, key=lambda item: _grid_centered_road_prior_score(grid=grid, projected_object=item))
    raise ValueError(f"Unsupported grid candidate selection rule: {selection_rule}")


def _grid_centered_road_prior_score(*, grid: GridCell, projected_object: ProjectedVehicleObject) -> float:
    distance = math.hypot(projected_object.center_world.x - grid.center.x, projected_object.center_world.y - grid.center.y)
    grid_heading = math.radians(grid.heading or 0.0)
    heading_diff = abs(_wrap_pi(projected_object.heading_rad - grid_heading))
    heading_penalty = min(heading_diff, abs(math.pi - heading_diff))
    size_penalty = abs(projected_object.length_m - 4.5) + abs(projected_object.width_m - 2.0)
    return distance + 1.25 * heading_penalty + 0.25 * size_penalty


def _wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def aggregate_occupancy_score(frame_rows: list[FrameGridOverlap], *, mode: str = "max") -> float:
    overlaps = [row.overlap_ratio for row in frame_rows]
    if not overlaps:
        return 0.0
    if mode == "mean_all_frames":
        return sum(overlaps) / float(len(overlaps))
    if mode == "max":
        return max(overlaps)
    raise ValueError(f"Unsupported occupancy aggregation mode: {mode}")
