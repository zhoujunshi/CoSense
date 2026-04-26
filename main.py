from __future__ import annotations

import json
from pathlib import Path

from detection import build_detector
from evaluation import write_grid_posteriors_csv, write_matrix_csv, write_post_em_decisions
from intrinsic_em import run_em
from occupancy import compute_case_occupancy
from projection import build_grid_definition, load_case_data
from reliability import build_case_reliability
from visibility import build_visibility_objects, compute_case_visibility


def load_config() -> dict:
    with Path(__file__).resolve().parent.joinpath("config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    config = load_config()
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(config)
    grids = build_grid_definition(config)
    grid_ids = tuple(str(cell.grid_id) for cell in grids.cells)
    row_names = []
    detection_rows = []
    reliability_rows = []

    for case_id in config["case_ids"]:
        frames, raw_projected_objects = load_case_data(case_id, config, detector)
        visibility_objects = build_visibility_objects(frames, raw_projected_objects, config)
        occupancy_results = compute_case_occupancy(grids, raw_projected_objects, config)
        visibility_results = compute_case_visibility(frames, grids, visibility_objects, config)
        reliability_results = build_case_reliability(visibility_results, occupancy_results, config)
        by_grid = {str(item.grid_id): item for item in reliability_results}
        row_names.append(case_id)
        detection_rows.append([int(by_grid[grid_id].detection_state) for grid_id in grid_ids])
        reliability_rows.append([float(by_grid[grid_id].extrinsic_reliability) for grid_id in grid_ids])

    detection_csv = output_dir / "detection.csv"
    reliability_csv = output_dir / "reliability.csv"
    write_matrix_csv(detection_csv, grid_ids, row_names, detection_rows)
    write_matrix_csv(reliability_csv, grid_ids, row_names, reliability_rows)

    em_result = run_em(detection_rows, reliability_rows, row_names, grid_ids, config)
    posterior_csv = output_dir / "grid_posteriors.csv"
    decisions_csv = output_dir / "grid_decisions_post_em.csv"
    write_grid_posteriors_csv(posterior_csv, em_result.grid_posteriors)
    metrics = write_post_em_decisions(decisions_csv, em_result.grid_posteriors, config)

    print(f"outputs={output_dir.resolve()}")
    print("files=detection.csv,reliability.csv,grid_posteriors.csv,grid_decisions_post_em.csv")
    print(f"post_em TP={metrics.tp} FP={metrics.fp} FN={metrics.fn} TN={metrics.tn} F1={metrics.f1:.3f}")


if __name__ == "__main__":
    main()
