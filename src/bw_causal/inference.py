"""Shared-multiplier confidence regions for declared finite target families."""

from dataclasses import dataclass

import numpy as np

from .geometry import projector, spectral_projector_if


@dataclass(frozen=True)
class Region:
    estimate: np.ndarray | float
    radius: float

    def contains(self, target):
        return np.linalg.norm(np.asarray(self.estimate) - target) <= self.radius

    def norm_interval(self):
        magnitude = np.linalg.norm(self.estimate)
        return max(0.0, magnitude - self.radius), magnitude + self.radius


@dataclass(frozen=True)
class ConfidenceResult:
    regions: dict
    abstentions: dict
    critical_value: float | None

    def covers(self, targets):
        return (not self.abstentions and self.regions.keys() == targets.keys()
                and all(self.regions[key].contains(value) for key, value in targets.items()))


def spectral_targets(matrices, *, eigenvalues=(1, 2, 3), projector_rank=1):
    targets = {}
    for row, (cov0, cov1) in enumerate(matrices):
        eval0 = np.linalg.eigvalsh(cov0)[::-1]
        eval1 = np.linalg.eigvalsh(cov1)[::-1]
        targets[(row, "matrix")] = cov1 - cov0
        for j in eigenvalues:
            targets[(row, f"eigenvalue_{j}")] = eval1[j - 1] - eval0[j - 1]
        if projector_rank is not None:
            targets[(row, f"projector_{projector_rank}")] = (
                projector(cov1, projector_rank) - projector(cov0, projector_rank)
            )
    return targets


def _first_order_contributions(prediction, *, eigenvalues, projector_rank, max_gap_ratio):
    contributions, abstentions = {}, {}
    p = prediction.matrices.shape[-1]
    if not all(1 <= j <= p for j in eigenvalues):
        raise ValueError("Eigenvalue indices are one based and must be within dimension")
    if projector_rank is not None and not 1 <= projector_rank < p:
        raise ValueError("Projector rank must be in [1, p-1]")
    for row, (cov0, cov1) in enumerate(prediction.matrices):
        labels = [f"eigenvalue_{number}" for number in eigenvalues]
        if projector_rank is not None:
            labels.append(f"projector_{projector_rank}")
        if not prediction.accepted[row].all():
            reason = f"root_failure:{','.join(prediction.statuses[row])}"
            for label in ["matrix", *labels]:
                abstentions[(row, label)] = reason
            continue
        arms = (cov0, cov1)
        effects = prediction.influences[row]
        contributions[(row, "matrix")] = cov1 - cov0, effects[1] - effects[0]
        eigensystems = [np.linalg.eigh(matrix) for matrix in arms]
        absolute_threshold = np.asarray([1e-10 * max(1, np.linalg.norm(matrix, ord=2)) for matrix in arms])
        for j in eigenvalues:
            index = p - j
            other = np.delete(np.arange(p), index)
            gaps = np.asarray([min(abs(values[other] - values[index]), default=np.inf)
                               for values, _ in eigensystems])
            key = row, f"eigenvalue_{j}"
            if np.any(gaps <= absolute_threshold):
                abstentions[key] = "weak_eigenvalue_gap"
                continue
            mixing = np.asarray([
                np.sqrt(np.sum(((basis[:, other].T @ influence @ basis[:, index])
                                / (values[index] - values[other])[None, :]) ** 2))
                for (values, basis), influence in zip(eigensystems, effects)
            ])
            if np.any(max_gap_ratio * mixing >= 1.0):
                abstentions[key] = "weak_eigenvalue_gap"
                continue
            vectors = [basis[:, index] for _, basis in eigensystems]
            scores = [np.einsum("i,nij,j->n", vector, influence, vector)
                      for vector, influence in zip(vectors, effects)]
            estimate = eigensystems[1][0][index] - eigensystems[0][0][index]
            contributions[key] = estimate, scores[1] - scores[0]
        if projector_rank is not None:
            key = row, f"projector_{projector_rank}"
            gaps = np.asarray([values[-projector_rank] - values[-projector_rank - 1]
                               for values, _ in eigensystems])
            if np.any(gaps <= absolute_threshold):
                abstentions[key] = "weak_projector_gap"
                continue
            mixing = np.asarray([
                np.sqrt(np.sum(((basis[:, :-projector_rank].T @ influence @ basis[:, -projector_rank:])
                                / (values[-projector_rank:][None, :] - values[:-projector_rank, None])[None, :, :]) ** 2))
                for (values, basis), influence in zip(eigensystems, effects)
            ])
            if np.any(max_gap_ratio * mixing >= 1.0):
                abstentions[key] = "weak_projector_gap"
            else:
                estimate = projector(cov1, projector_rank) - projector(cov0, projector_rank)
                directions = [spectral_projector_if(matrix, influence, projector_rank)
                              for matrix, influence in zip(arms, effects)]
                contributions[key] = estimate, directions[1] - directions[0]
    return contributions, abstentions


def simultaneous_regions(prediction, *, eigenvalues=(1, 2, 3), projector_rank=1,
                         n_multiplier=999, level=0.95, seed=0, max_gap_ratio=1.0,
                         joint=True):
    """Treat the frozen design as fixed and abstain on unstable spectral directions.

    max_gap_ratio multiplies the estimated Frobenius standard error of each arm
    as a prespecified stability diagnostic. It is not an operator-error bound.
    """
    if prediction.influences is None:
        raise ValueError("Prediction must include fitted root influences")
    if not (0 < level < 1) or n_multiplier < 1 or max_gap_ratio < 0:
        raise ValueError("Invalid bootstrap settings")
    contributions, abstentions = _first_order_contributions(
        prediction, eigenvalues=eigenvalues, projector_rank=projector_rank,
        max_gap_ratio=max_gap_ratio,
    )
    n = prediction.influences.shape[2]
    multipliers = np.random.default_rng(seed).normal(size=(n_multiplier, n))
    max_statistics = np.zeros(n_multiplier)
    active = {}
    for key, (estimate, individual) in contributions.items():
        individual = individual.reshape(n, -1)
        scale = np.linalg.norm(individual)
        if not np.isfinite(scale) or scale <= 1e-15:
            abstentions[key] = "zero_or_nonfinite_variance"
            continue
        replicates = multipliers @ individual
        statistics = np.linalg.norm(replicates, axis=1) / scale
        max_statistics = np.maximum(max_statistics, statistics)
        individual_critical = None if joint else np.quantile(statistics, level, method="higher")
        active[key] = (estimate, scale, individual_critical)
    critical = float(np.quantile(max_statistics, level, method="higher")) if active else None
    regions = {key: Region(estimate, (critical if joint else individual_critical) * scale)
               for key, (estimate, scale, individual_critical) in active.items()}
    return ConfidenceResult(regions, abstentions, critical if joint else None)
