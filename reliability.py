from __future__ import annotations

import math
from dataclasses import dataclass

from occupancy import GridOccupancyResult
from visibility import GridVisibilityResult

STATE_FREE = 0
STATE_OCCUPIED = 1


@dataclass(frozen=True)
class ExtrinsicReliabilityConfig:
    visibility_threshold_eta: float
    occupancy_threshold_tau: float
    hit_ratio_threshold_tau_h: float
    lambda1: float
    lambda2: float
    epsilon: float


@dataclass(frozen=True)
class GridExtrinsicReliabilityResult:
    grid_id: str
    ground_truth_label: int | None
    visibility_score: float
    occupancy_score: float
    hit_ratio: float
    detection_state: int
    extrinsic_reliability: float


def build_case_reliability(visibility_results, occupancy_results, config: dict):
    reliability_config = ExtrinsicReliabilityConfig(
        visibility_threshold_eta=float(config["reliability"]["visibility_threshold_eta"]),
        occupancy_threshold_tau=float(config["reliability"]["occupancy_threshold_tau_u"]),
        hit_ratio_threshold_tau_h=float(config["reliability"]["hit_ratio_threshold_tau_h"]),
        lambda1=float(config["reliability"]["lambda1"]),
        lambda2=float(config["reliability"]["lambda2"]),
        epsilon=float(config["reliability"]["epsilon"]),
    )
    return build_extrinsic_reliability_results(
        visibility_results=visibility_results,
        occupancy_results=occupancy_results,
        config=reliability_config,
    )


def assign_detection_state(*, occupancy_score: float, config: ExtrinsicReliabilityConfig) -> int:
    if occupancy_score >= config.occupancy_threshold_tau:
        return STATE_OCCUPIED
    return STATE_FREE


def compute_extrinsic_reliability(
    *,
    visibility_score: float,
    hit_ratio: float,
    detection_state: int,
    config: ExtrinsicReliabilityConfig,
) -> float:
    visibility_term = config.lambda1 * (visibility_score - config.visibility_threshold_eta)
    if detection_state == STATE_OCCUPIED:
        reliability = _sigmoid(visibility_term + config.lambda2 * (hit_ratio - config.hit_ratio_threshold_tau_h))
    elif detection_state == STATE_FREE:
        reliability = _sigmoid(visibility_term)
    else:
        raise ValueError(f"Unsupported binary detection state: {detection_state}")
    return _clamp_probability(reliability, config.epsilon)


def build_extrinsic_reliability_results(
    *,
    visibility_results: tuple[GridVisibilityResult, ...],
    occupancy_results: tuple[GridOccupancyResult, ...],
    config: ExtrinsicReliabilityConfig,
) -> tuple[GridExtrinsicReliabilityResult, ...]:
    occupancy_by_grid = {result.grid_id: result for result in occupancy_results}
    results = []
    for visibility in visibility_results:
        occupancy = occupancy_by_grid.get(visibility.grid_id)
        if occupancy is None:
            raise ValueError(f"Missing occupancy result for grid: {visibility.grid_id}")
        state = assign_detection_state(
            occupancy_score=occupancy.occupancy_score,
            config=config,
        )
        in_fov_frame_count = max(0, int(visibility.in_fov_frame_count))
        hit_frame_count = max(0, int(occupancy.hit_frame_count))
        hit_ratio = 0.0 if in_fov_frame_count == 0 else hit_frame_count / float(in_fov_frame_count)
        reliability = compute_extrinsic_reliability(
            visibility_score=visibility.visibility_score,
            hit_ratio=hit_ratio,
            detection_state=state,
            config=config,
        )
        results.append(
            GridExtrinsicReliabilityResult(
                grid_id=visibility.grid_id,
                ground_truth_label=visibility.ground_truth_label,
                visibility_score=visibility.visibility_score,
                occupancy_score=occupancy.occupancy_score,
                hit_ratio=hit_ratio,
                detection_state=state,
                extrinsic_reliability=reliability,
            )
        )
    return tuple(results)


def _clamp_probability(value: float, epsilon: float) -> float:
    eps = max(0.0, min(0.5, epsilon))
    return max(eps, min(1.0 - eps, value))


def _sigmoid(value: float) -> float:
    value = max(min(value, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-value))
