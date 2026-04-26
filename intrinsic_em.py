from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IntrinsicEMConfig:
    max_iterations: int = 300
    tolerance: float = 1e-6
    sigma2: float = 1.0
    epsilon: float = 1e-8
    q_epsilon: float = 1e-4
    posterior_threshold: float = 0.5
    mu_init_clip: float = 0.05
    agreement_init_low: float = 0.55
    agreement_init_high: float = 0.95
    newton_tolerance: float = 1e-8
    newton_max_iterations: int = 50


@dataclass(frozen=True)
class MatrixAxis:
    name: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class MatrixOutput:
    matrix_name: str
    row_axis: MatrixAxis
    column_axis: MatrixAxis
    values: tuple[tuple[float, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.values), len(self.column_axis.labels)


@dataclass(frozen=True)
class GridPosterior:
    grid_id: str
    posterior_score: float
    pred_label: int


@dataclass(frozen=True)
class CaseIntrinsicReliability:
    case_id: str
    gamma: float
    intrinsic_reliability: float


@dataclass(frozen=True)
class IntrinsicEMResult:
    grid_posteriors: tuple[GridPosterior, ...]
    case_reliabilities: tuple[CaseIntrinsicReliability, ...]
    iterations: int
    converged: bool
    prior_probability: float
    max_delta: float


def run_em(detection_rows, reliability_rows, row_names, grid_ids, config: dict):
    em_cfg = config["intrinsic_em"]
    em_config = IntrinsicEMConfig(
        max_iterations=int(em_cfg["max_iterations"]),
        tolerance=float(em_cfg["tolerance"]),
        sigma2=float(em_cfg["sigma2"]),
        epsilon=float(em_cfg["epsilon"]),
        q_epsilon=float(em_cfg["q_epsilon"]),
        posterior_threshold=float(em_cfg["posterior_threshold"]),
        mu_init_clip=float(em_cfg["mu_init_clip"]),
        agreement_init_low=float(em_cfg["agreement_init_low"]),
        agreement_init_high=float(em_cfg["agreement_init_high"]),
        newton_tolerance=float(em_cfg["newton_tolerance"]),
        newton_max_iterations=int(em_cfg["newton_max_iterations"]),
    )
    detection_matrix = MatrixOutput(
        matrix_name="detection",
        row_axis=MatrixAxis(name="VideoFolderName", labels=tuple(row_names)),
        column_axis=MatrixAxis(name="columns", labels=tuple(grid_ids)),
        values=tuple(tuple(float(value) for value in row) for row in detection_rows),
    )
    reliability_matrix = MatrixOutput(
        matrix_name="reliability",
        row_axis=detection_matrix.row_axis,
        column_axis=detection_matrix.column_axis,
        values=tuple(tuple(float(value) for value in row) for row in reliability_rows),
    )
    return run_intrinsic_em(
        detection_matrix=detection_matrix,
        reliability_matrix=reliability_matrix,
        config=em_config,
    )


def run_intrinsic_em(
    *,
    detection_matrix: MatrixOutput,
    reliability_matrix: MatrixOutput,
    config: IntrinsicEMConfig,
) -> IntrinsicEMResult:
    _validate_aligned_matrices(detection_matrix, reliability_matrix)
    c_values = [[int(value) for value in row] for row in detection_matrix.values]
    q_values = [
        [_clip(float(value), config.q_epsilon, 1.0 - config.q_epsilon) for value in row]
        for row in reliability_matrix.values
    ]
    row_count = len(c_values)
    column_count = len(detection_matrix.column_axis.labels)
    observed = [[state in (0, 1) for state in row] for row in c_values]

    mu = _initial_mu(c_values, observed, column_count, config)
    prior = _clip(sum(mu) / len(mu), config.epsilon, 1.0 - config.epsilon)
    gamma = _initial_gamma(c_values, observed, mu, row_count, config)

    converged = False
    max_delta = math.inf
    iteration = 0
    for iteration in range(1, config.max_iterations + 1):
        previous_mu = list(mu)
        previous_gamma = list(gamma)
        previous_prior = prior

        mu = _e_step(c_values, q_values, observed, gamma, prior, config)
        prior = _clip(sum(mu) / len(mu), config.epsilon, 1.0 - config.epsilon)
        gamma = _update_gamma(c_values, q_values, observed, mu, gamma, config)

        max_delta = max(
            max(abs(new - old) for new, old in zip(mu, previous_mu)),
            max(abs(new - old) for new, old in zip(gamma, previous_gamma)),
            abs(prior - previous_prior),
        )
        if max_delta < config.tolerance:
            converged = True
            break

    grid_posteriors = tuple(
        GridPosterior(
            grid_id=str(grid_id),
            posterior_score=score,
            pred_label=int(score >= config.posterior_threshold),
        )
        for grid_id, score in zip(detection_matrix.column_axis.labels, mu)
    )
    case_reliabilities = tuple(
        CaseIntrinsicReliability(
            case_id=str(case_id),
            gamma=value,
            intrinsic_reliability=_sigmoid(value),
        )
        for case_id, value in zip(detection_matrix.row_axis.labels, gamma)
    )
    return IntrinsicEMResult(
        grid_posteriors=grid_posteriors,
        case_reliabilities=case_reliabilities,
        iterations=iteration,
        converged=converged,
        prior_probability=prior,
        max_delta=max_delta,
    )


def _validate_aligned_matrices(detection_matrix: MatrixOutput, reliability_matrix: MatrixOutput) -> None:
    if detection_matrix.row_axis.labels != reliability_matrix.row_axis.labels:
        raise ValueError("Detection and reliability matrices must have the same row labels.")
    if detection_matrix.column_axis.labels != reliability_matrix.column_axis.labels:
        raise ValueError("Detection and reliability matrices must have the same grid columns.")
    if detection_matrix.shape != reliability_matrix.shape:
        raise ValueError("Detection and reliability matrices must have the same shape.")


def _initial_mu(c_values, observed, column_count: int, config: IntrinsicEMConfig) -> list[float]:
    values = []
    for column_index in range(column_count):
        known_states = [
            c_values[row_index][column_index]
            for row_index in range(len(c_values))
            if observed[row_index][column_index]
        ]
        if not known_states:
            values.append(0.5)
        else:
            values.append(sum(known_states) / len(known_states))
    return [_clip(value, config.mu_init_clip, 1.0 - config.mu_init_clip) for value in values]


def _initial_gamma(c_values, observed, mu: list[float], row_count: int, config: IntrinsicEMConfig) -> list[float]:
    gamma = []
    for row_index in range(row_count):
        agreements = []
        for column_index, mu_value in enumerate(mu):
            if not observed[row_index][column_index]:
                continue
            state = c_values[row_index][column_index]
            agreements.append(mu_value if state == 1 else 1.0 - mu_value)
        agreement = sum(agreements) / len(agreements) if agreements else 0.5
        agreement = _clip(agreement, config.agreement_init_low, config.agreement_init_high)
        gamma.append(_logit(agreement, config.epsilon))
    return gamma


def _e_step(c_values, q_values, observed, gamma, prior: float, config: IntrinsicEMConfig) -> list[float]:
    column_count = len(c_values[0]) if c_values else 0
    mu = []
    for column_index in range(column_count):
        ll_occupied = math.log(_clip(prior, config.epsilon, 1.0 - config.epsilon))
        ll_free = math.log(_clip(1.0 - prior, config.epsilon, 1.0 - config.epsilon))
        for row_index, row in enumerate(c_values):
            if not observed[row_index][column_index]:
                continue
            reliability = _adjusted_reliability(q_values[row_index][column_index], gamma[row_index], config)
            agreement_probability = _agreement_probability(reliability, config)
            log_agree = math.log(_clip(agreement_probability, config.epsilon, 1.0 - config.epsilon))
            log_disagree = math.log(_clip(1.0 - agreement_probability, config.epsilon, 1.0 - config.epsilon))
            if row[column_index] == 1:
                ll_occupied += log_agree
                ll_free += log_disagree
            else:
                ll_occupied += log_disagree
                ll_free += log_agree
        mu.append(_sigmoid(ll_occupied - ll_free))
    return mu


def _update_gamma(c_values, q_values, observed, mu, gamma_old, config: IntrinsicEMConfig) -> list[float]:
    gamma_new = []
    for row_index, row in enumerate(c_values):
        gamma_value = gamma_old[row_index]
        observed_columns = [
            column_index for column_index in range(len(row)) if observed[row_index][column_index]
        ]
        if not observed_columns:
            gamma_new.append(0.0)
            continue

        for _ in range(config.newton_max_iterations):
            gradient, hessian = _gamma_objective_derivatives(
                row=row,
                q_row=q_values[row_index],
                observed_columns=observed_columns,
                mu=mu,
                gamma_value=gamma_value,
                config=config,
            )
            step = gradient / (hessian + 1e-12)
            next_gamma = gamma_value - step
            if abs(next_gamma - gamma_value) < config.newton_tolerance:
                gamma_value = next_gamma
                break
            gamma_value = next_gamma
        gamma_new.append(gamma_value)
    return gamma_new


def _adjusted_reliability(q_value: float, gamma_value: float, config: IntrinsicEMConfig) -> float:
    return _sigmoid(_logit(q_value, config.epsilon) + gamma_value)


def _agreement_probability(reliability: float, config: IntrinsicEMConfig) -> float:
    return _clip(0.5 + 0.5 * reliability, config.epsilon, 1.0 - config.epsilon)


def _gamma_objective_derivatives(
    *,
    row,
    q_row,
    observed_columns,
    mu,
    gamma_value: float,
    config: IntrinsicEMConfig,
) -> tuple[float, float]:
    step = 1e-4
    center = _gamma_objective(
        row=row,
        q_row=q_row,
        observed_columns=observed_columns,
        mu=mu,
        gamma_value=gamma_value,
        config=config,
    )
    plus = _gamma_objective(
        row=row,
        q_row=q_row,
        observed_columns=observed_columns,
        mu=mu,
        gamma_value=gamma_value + step,
        config=config,
    )
    minus = _gamma_objective(
        row=row,
        q_row=q_row,
        observed_columns=observed_columns,
        mu=mu,
        gamma_value=gamma_value - step,
        config=config,
    )
    gradient = (plus - minus) / (2.0 * step)
    hessian = (plus - 2.0 * center + minus) / (step * step)
    if hessian >= -1e-9:
        hessian = -1e-9
    return gradient, hessian


def _gamma_objective(
    *,
    row,
    q_row,
    observed_columns,
    mu,
    gamma_value: float,
    config: IntrinsicEMConfig,
) -> float:
    value = -(gamma_value * gamma_value) / (2.0 * config.sigma2)
    for column_index in observed_columns:
        state = row[column_index]
        expected_agreement = mu[column_index] if state == 1 else 1.0 - mu[column_index]
        reliability = _adjusted_reliability(q_row[column_index], gamma_value, config)
        agreement_probability = _agreement_probability(reliability, config)
        value += expected_agreement * math.log(_clip(agreement_probability, config.epsilon, 1.0 - config.epsilon))
        value += (1.0 - expected_agreement) * math.log(
            _clip(1.0 - agreement_probability, config.epsilon, 1.0 - config.epsilon)
        )
    return value


def _sigmoid(value: float) -> float:
    value = _clip(value, -50.0, 50.0)
    return 1.0 / (1.0 + math.exp(-value))


def _logit(value: float, epsilon: float) -> float:
    value = _clip(value, epsilon, 1.0 - epsilon)
    return math.log(value / (1.0 - value))


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
