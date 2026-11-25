"""Matched embedding baselines and nonaugmented BW Fréchet learners."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .forest import (
    AnchorCoordinates, ForestConfig, MatrixDRForest, MatrixPrediction, fit_forest, forest_kernel,
    normalize_kernel,
)
from .geometry import (
    CovarianceSample, bw_barycenter, bw_distance, project_psd, smat,
    solve_augmented_barycenter, svec, sym,
)


class SqrtEuclideanDRForest(MatrixDRForest):
    def fit(self, cohort):
        return self._fit(cohort, AnchorCoordinates(np.eye(cohort.dimension)))


@dataclass(frozen=True)
class _LogCoordinates:
    dimension: int
    ridge: float

    def encode(self, matrices):
        values, vectors = np.linalg.eigh(sym(matrices))
        if np.min(values) < -1e-10 * max(1, np.max(values)):
            raise ValueError("Expected PSD matrices in Log-Euclidean encoding")
        logged = np.log(np.maximum(values, 0) + self.ridge)
        return svec(sym((vectors * logged[:, None, :]) @ vectors.swapaxes(-1, -2)))

    def decode(self, vector):
        values, vectors = np.linalg.eigh(smat(vector, self.dimension))
        return sym((vectors * np.maximum(np.exp(values) - self.ridge, 0.0)) @ vectors.T)


class LogEuclideanDRForest(MatrixDRForest):
    def __init__(self, config=ForestConfig(), *, ridge=1e-6):
        super().__init__(config)
        if ridge <= 0:
            raise ValueError("Log-Euclidean encoding requires a positive ridge for PSD data")
        self.ridge = ridge

    def fit(self, cohort):
        return self._fit(cohort, _LogCoordinates(cohort.dimension, self.ridge))


@dataclass(frozen=True)
class _LogCholeskyCoordinates:
    dimension: int
    ridge: float

    def encode(self, matrices):
        chol = np.linalg.cholesky(matrices + self.ridge * np.eye(self.dimension))
        rows, cols = np.tril_indices(self.dimension, k=-1)
        return np.column_stack((chol[:, rows, cols], np.log(np.diagonal(chol, axis1=-2, axis2=-1))))

    def decode(self, vector):
        factor = np.diag(np.exp(vector[-self.dimension:]))
        factor[np.tril_indices(self.dimension, k=-1)] = vector[:-self.dimension]
        return project_psd(factor @ factor.T - self.ridge * np.eye(self.dimension))


class LogCholeskyDRForest(MatrixDRForest):
    def __init__(self, config=ForestConfig(), *, ridge=1e-6):
        super().__init__(config)
        if ridge <= 0:
            raise ValueError("Log-Cholesky encoding requires a positive ridge for PSD data")
        self.ridge = ridge

    def fit(self, cohort):
        return self._fit(cohort, _LogCholeskyCoordinates(cohort.dimension, self.ridge))


@dataclass(frozen=True)
class _ElementwiseCoordinates:
    dimension: int

    def encode(self, matrices):
        return svec(matrices)

    def decode(self, vector):
        return project_psd(smat(vector, self.dimension))


class ElementwiseDRForest(MatrixDRForest):
    def fit(self, cohort):
        return self._fit(cohort, _ElementwiseCoordinates(cohort.dimension))


class BWFrechetTLearner(MatrixDRForest):
    """Same design and score covariates, but no orthogonalization in either layer."""

    def fit(self, cohort, *, anchor=None):
        if anchor is None:
            anchor = np.eye(cohort.dimension)
        split = cohort.split
        coordinates = AnchorCoordinates(anchor)
        self.cohort = cohort
        responses = coordinates.encode(cohort.Y[split.design])
        fitted = [cohort.design_scores.regression(responses, arm) for arm in (0, 1)]
        self.trees = fit_forest(cohort.X[split.design], fitted[1] - fitted[0], self.config)
        self.arm_samples = _arm_samples(cohort)
        return self

    def predict(self, queries=None, *, subgroup_masks=None):
        alpha = self.alpha(queries, subgroup_masks=subgroup_masks)
        a = self.cohort.evaluation_scores.a
        return _nonnegative_prediction(self.arm_samples, [alpha[:, a == arm] for arm in (0, 1)])


def _arm_samples(cohort):
    a = cohort.evaluation_scores.a
    return [CovarianceSample(cohort.Y[cohort.split.evaluation][a == arm]) for arm in (0, 1)]


def _nonnegative_prediction(samples, arm_weights):
    n, p = len(arm_weights[0]), samples[0].dimension
    matrices = np.full((n, 2, p, p), np.nan)
    status = [["unsupported", "unsupported"] for _ in range(n)]
    diagnostics = [[{}, {}] for _ in range(n)]
    for arm in (0, 1):
        for row, weights in enumerate(arm_weights[arm]):
            fit = bw_barycenter(samples[arm], weights)
            status[row][arm] = fit.status
            diagnostics[row][arm] = {
                "local_arm_count": int(np.count_nonzero(weights)),
                "solver_iterations": fit.iterations,
                "solver_residual": float(fit.residual) if np.isfinite(fit.residual) else None,
            }
            if fit.accepted:
                matrices[row, arm] = fit.covariance
    return MatrixPrediction(matrices, status, diagnostics)


def _frechet_variance(outcomes):
    fit = bw_barycenter(outcomes, tolerance=1e-4, max_iter=30)
    if not fit.accepted:
        return np.inf
    values = bw_distance(outcomes, fit.covariance)
    return values @ values


def _two_means_cut(values, min_leaf):
    centers = np.array([values[0], values[-1]])
    for _ in range(25):
        cut = np.searchsorted(values, centers.mean(), side="right")
        if not 0 < cut < len(values):
            return None
        updated = np.array([values[:cut].mean(), values[cut:].mean()])
        if np.max(abs(updated - centers)) < 1e-10:
            break
        centers = updated
    if cut < min_leaf or len(values) - cut < min_leaf:
        return None
    return cut


def _frechet_split(X, Y, indices, features, config):
    best = None
    parent = _frechet_variance(Y[indices])
    if not np.isfinite(parent):
        return best
    for feature in features:
        order = indices[np.argsort(X[indices, feature], kind="stable")]
        values = X[order, feature]
        cut = _two_means_cut(values, config.min_leaf_size)
        if cut is None:
            continue
        left, right = order[:cut], order[cut:]
        reduction = (parent - _frechet_variance(Y[left]) - _frechet_variance(Y[right])) / len(indices)
        if best is None or reduction > best[0]:
            best = reduction, feature, (values[cut - 1] + values[cut]) / 2, left, right
    return best


class BWRFWLFRTLearner:
    """Arm-stratified honest local-constant BW RFWLFR with direct Fréchet split loss."""

    def __init__(self, config=ForestConfig()):
        self.config = config

    def fit(self, cohort):
        self.cohort = cohort
        split = cohort.split
        self.forests = []
        self.arm_samples = _arm_samples(cohort)
        for arm in (0, 1):
            index = split.design[cohort.a[split.design] == arm]
            self.forests.append(fit_forest(cohort.X[index], cohort.Y[index], self.config,
                                           split_rule=_frechet_split))
        return self

    def predict(self, queries=None, *, subgroup_masks=None):
        X, a = self.cohort.evaluation_scores.X, self.cohort.evaluation_scores.a
        if subgroup_masks is not None:
            weights = [np.asarray(subgroup_masks)[:, a == arm] for arm in (0, 1)]
        else:
            weights = [
                normalize_kernel(forest_kernel(self.forests[arm], X[a == arm], np.atleast_2d(queries)))
                for arm in (0, 1)
            ]
        return _nonnegative_prediction(self.arm_samples, weights)


class BWFrechetGlobalTLearner:
    """Signed global Fréchet regression per arm with ridge-stabilized covariates."""

    def __init__(self, *, ridge=1e-6):
        self.ridge = ridge

    def fit(self, cohort):
        self.cohort = cohort
        self.samples = _arm_samples(cohort)
        self.designs = []
        X, a = cohort.evaluation_scores.X, cohort.evaluation_scores.a
        for arm, sample in enumerate(self.samples):
            features = X[a == arm]
            mean = features.mean(axis=0)
            centered = features - mean
            covariance = centered.T @ centered / len(features) + self.ridge * np.eye(X.shape[1])
            operator = np.linalg.solve(covariance, centered.T)
            pilot = bw_barycenter(sample)
            self.designs.append((mean, operator, pilot))
        return self

    def predict(self, queries=None, *, subgroup_masks=None):
        if subgroup_masks is not None:
            raise ValueError("The global regression baseline takes query covariates")
        queries = np.atleast_2d(queries)
        p = self.cohort.dimension
        matrices = np.full((len(queries), 2, p, p), np.nan)
        status = [["unsupported", "unsupported"] for _ in queries]
        diagnostics = [[{}, {}] for _ in queries]
        for arm, (mean, operator, pilot) in enumerate(self.designs):
            if not pilot.accepted:
                continue
            for row, weights in enumerate(1 + (queries - mean) @ operator):
                weights = weights / weights.sum()
                fit = solve_augmented_barycenter(self.samples[arm], weights, pilot.covariance)
                status[row][arm] = fit.status
                diagnostics[row][arm] = {
                    "solver_iterations": fit.iterations,
                    "solver_residual": float(fit.residual) if np.isfinite(fit.residual) else None,
                }
                if fit.accepted:
                    matrices[row, arm] = fit.covariance
        return MatrixPrediction(matrices, status, diagnostics)


def entropy_balanced_weights(prior, features, target, *, tolerance=1e-7):
    """Finite-moment calibration on the positive-prior support."""
    prior = np.asarray(prior, dtype=float)
    active = prior > 0
    if not np.any(active):
        return None
    differences = features[active] - target
    scale = np.maximum(np.max(np.abs(differences), axis=0), 1e-12)
    differences = differences / scale
    _, singular, vectors = np.linalg.svd(differences, full_matrices=False)
    rank = np.count_nonzero(singular > 1e-12)
    reduced = differences @ vectors[:rank].T
    log_prior = np.log(prior[active] / prior[active].sum())

    def objective(beta):
        log_weight = log_prior + reduced @ beta
        normalizer = logsumexp(log_weight)
        weight = np.exp(log_weight - normalizer)
        return normalizer, weight @ reduced

    result = np.zeros_like(prior)
    if rank:
        optimum = minimize(objective, np.zeros(rank), jac=True, method="L-BFGS-B",
                           bounds=[(-60, 60)] * rank, options={"gtol": 1e-10, "ftol": 1e-14})
        log_weights = log_prior + reduced @ optimum.x
        result[active] = np.exp(log_weights - logsumexp(log_weights))
    else:
        result[active] = np.exp(log_prior)
    if np.max(np.abs(result[active] @ differences)) > tolerance:
        return None
    return result
