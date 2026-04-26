from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def f1(self) -> float:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def write_matrix_csv(path: str | Path, grid_ids, row_names, rows) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["VideoFolderName", *grid_ids])
        for row_name, row in zip(row_names, rows):
            writer.writerow([row_name, *row])


def write_grid_posteriors_csv(path: str | Path, grid_posteriors) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["grid_id", "posterior_score", "pred_label"])
        writer.writeheader()
        for item in grid_posteriors:
            writer.writerow(
                {
                    "grid_id": item.grid_id,
                    "posterior_score": f"{item.posterior_score:.12g}",
                    "pred_label": int(item.posterior_score >= 0.60),
                }
            )


def write_post_em_decisions(path: str | Path, grid_posteriors, config: dict) -> Metrics:
    labels = {str(key): int(value) for key, value in config["grid_labels"].items()}
    threshold = float(config["intrinsic_em"]["posterior_threshold"])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tp = fp = fn = tn = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["grid_id", "pred_label", "gt_label", "correct", "posterior_score"],
        )
        writer.writeheader()
        for item in grid_posteriors:
            grid_id = str(item.grid_id)
            pred = int(item.posterior_score >= threshold)
            gt = labels[grid_id]
            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 1 and gt == 0:
                fp += 1
            elif pred == 0 and gt == 1:
                fn += 1
            else:
                tn += 1
            writer.writerow(
                {
                    "grid_id": grid_id,
                    "pred_label": pred,
                    "gt_label": gt,
                    "correct": int(pred == gt),
                    "posterior_score": f"{item.posterior_score:.12g}",
                }
            )
    return Metrics(tp=tp, fp=fp, fn=fn, tn=tn)
