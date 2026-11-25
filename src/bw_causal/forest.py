"""Common honest design, cross-fitted score machinery, and intrinsic forest."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from .geometry import (
    BarycenterFit, CovarianceSample, bw_barycenter, bw_exp, project_psd, psd_sqrt,
    smat, solve_augmented_barycenter, svec,
    sym,
)


@dataclass(frozen=True)
class HonestSplit:
    anchor: np.ndarray
    design: np.ndarray
    evaluation: np.ndarray
    design_fold: np.ndarray
    evaluation_fold: np.ndarray

    @classmethod
    def make(cls, n, *, seed=0, n_folds=3, anchor_fraction=0.15, design_fraction=0.35):
        if n_folds < 2 or min(anchor_fraction, design_fraction) <= 0:
            raise ValueError("At least two folds and separate anchor and design samples are required")
        if anchor_fraction + design_fraction >= 1:
            raise ValueError("A nonempty evaluation sample is required")
        perm = np.random.default_rng(seed).permutation(n)
        a, d = int(n * anchor_fraction), int(n * design_fraction)
        if min(a, d, n - a - d) < n_folds:
            raise ValueError("Too few observations for the honest split")
        return cls(perm[:a], perm[a:a + d], perm[a + d:],
                   np.arange(d) % n_folds, np.arange(n - a - d) % n_folds)


@dataclass(frozen=True)
class ForestConfig:
    n_trees: int = 200
    min_leaf_size: int = 12
    max_depth: int = 5
    subsample: float = 0.7
    max_features: float = 1.0
    max_cutpoints: int = 24
    seed: int = 0


def _features(X, use_squares):
    return np.column_stack((X, X**2)) if use_squares else np.asarray(X)


def _design(training, evaluation):
    mean = training.mean(axis=0)
    sd = training.std(axis=0)
    sd[sd < 1e-12] = 1
    return (np.column_stack((np.ones(len(training)), (training - mean) / sd)),
            np.column_stack((np.ones(len(evaluation)), (evaluation - mean) / sd)))


def _logistic(training, treatment, evaluation, penalty):
    design, test = _design(training, evaluation)
    p = design.shape[1]
    regularizer = np.repeat(penalty, p)
    regularizer[0] = 0
    beta = np.zeros(p)
    for _ in range(80):
        eta = design @ beta
        probability = expit(eta)
        score = design.T @ (probability - treatment) + regularizer * beta
        information = (design.T * (probability * (1 - probability))) @ design + np.diag(regularizer)
        direction = np.linalg.solve(information, score)
        if np.max(np.abs(direction)) < 1e-9:
            break
        loss = np.sum(np.logaddexp(0, eta) - treatment * eta) + regularizer @ beta**2 / 2
        for backtrack in range(15):
            candidate = beta - direction * 2.0**(-backtrack)
            new_eta = design @ candidate
            new_loss = (np.sum(np.logaddexp(0, new_eta) - treatment * new_eta)
                        + regularizer @ candidate**2 / 2)
            if new_loss <= loss:
                beta = candidate
                break
        else:
            break
    return expit(test @ beta)


@dataclass(frozen=True)
class _SmootherBlock:
    validation: np.ndarray
    training: np.ndarray
    evaluation_basis: np.ndarray
    coefficient_map: np.ndarray


class CrossFittedScores:
    """Factorized constant-reproducing arm score regressions and shared propensity."""

    def __init__(self, X, a, folds, *, use_squares=True, ridge=1.0,
                 propensity_penalty=1.0, overlap_floor=0.05, propensity_override=None,
                 score_feature_indices=None, fit_on_all_rows=False):
        if min(ridge, propensity_penalty) <= 0 or not 0 < overlap_floor < 0.5:
            raise ValueError("Nuisance penalties and overlap bounds must be positive")
        self.a = np.asarray(a)
        self.X = np.asarray(X)
        score_covariates = self.X if score_feature_indices is None else self.X[:, score_feature_indices]
        self.features = _features(score_covariates, use_squares)
        self.e = np.empty(len(a))
        self.blocks = ([], [])
        propensity_features = self.X
        for fold in np.unique(folds):
            training = np.arange(len(a)) if fit_on_all_rows else np.flatnonzero(folds != fold)
            validation = np.flatnonzero(folds == fold)
            if len(training) == 0 or np.unique(a[training]).size != 2:
                raise ValueError(f"Cross-fitting fold {fold} has missing treatment support")
            if propensity_override is None:
                self.e[validation] = _logistic(propensity_features[training], a[training],
                                               propensity_features[validation], propensity_penalty)
            for arm in (0, 1):
                arm_training = training[a[training] == arm]
                train, test = _design(self.features[arm_training], self.features[validation])
                penalty = np.repeat(ridge, train.shape[1])
                penalty[0] = 0
                coefficients = np.linalg.solve(train.T @ train + np.diag(penalty), train.T)
                self.blocks[arm].append(_SmootherBlock(validation, arm_training, test, coefficients))
        if propensity_override is not None:
            self.e[:] = propensity_override
        if not np.all(np.isfinite(self.e)) or np.any((self.e <= 0) | (self.e >= 1)):
            raise ValueError("Propensity values must lie in (0, 1)")
        self.e = np.clip(self.e, overlap_floor, 1 - overlap_floor)

    def inverse_probability(self, arm):
        return (self.a == arm) / (self.e if arm == 1 else 1 - self.e)

    def regression(self, response, arm):
        response = np.asarray(response, dtype=float)
        flat_response = response.reshape(len(response), -1)
        fitted = np.zeros_like(flat_response)
        for block in self.blocks[arm]:
            fitted[block.validation] = (
                block.evaluation_basis @ (block.coefficient_map @ flat_response[block.training])
            )
        return fitted.reshape(response.shape)

    def scores(self, response, arm):
        response = np.asarray(response, dtype=float)
        fitted = self.regression(response, arm)
        q = self.inverse_probability(arm).reshape((len(response),) + (1,) * (response.ndim - 1))
        return fitted + q * (response - fitted)

    def equivalent_weights(self, alpha, arm):
        alpha = np.asarray(alpha)
        q = self.inverse_probability(arm)
        weights = alpha * q
        for block in self.blocks[arm]:
            validation = block.validation
            adjustments = alpha[..., validation] * (1 - q[validation])
            weights[..., block.training] += (
                adjustments @ block.evaluation_basis @ block.coefficient_map
            )
        return weights


@dataclass(frozen=True)
class PreparedCohort:
    X: np.ndarray
    a: np.ndarray
    Y: np.ndarray
    split: HonestSplit
    design_scores: CrossFittedScores
    evaluation_scores: CrossFittedScores

    @property
    def dimension(self):
        return self.Y.shape[-1]


def prepare_cohort(X, a, Y, *, split=None, seed=0, propensity_override=None, **nuisance_options):
    X, a, Y = np.asarray(X, dtype=float), np.asarray(a), np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 3 or not (len(X) == len(a) == len(Y)):
        raise ValueError("X, A, and covariance outcomes must align")
    if not np.all(np.isfinite(X)) or not np.isin(a, [0, 1]).all():
        raise ValueError("Covariates must be finite and treatment must be binary")
    if split is None:
        split = HonestSplit.make(len(X), seed=seed)
    combined = np.concatenate((split.anchor, split.design, split.evaluation))
    if not np.array_equal(np.sort(combined), np.arange(len(X))):
        raise ValueError("Anchor, design and estimation sets must be a disjoint partition")
    if len(split.design_fold) != len(split.design) or len(split.evaluation_fold) != len(split.evaluation):
        raise ValueError("Fold assignments must cover the design and evaluation samples")
    override = None if propensity_override is None else np.asarray(propensity_override)
    design = CrossFittedScores(X[split.design], a[split.design], split.design_fold,
                              propensity_override=None if override is None else override[split.design],
                              **nuisance_options)
    evaluation = CrossFittedScores(X[split.evaluation], a[split.evaluation], split.evaluation_fold,
                                  propensity_override=None if override is None else override[split.evaluation],
                                  **nuisance_options)
    return PreparedCohort(X, a, Y, split, design, evaluation)


class AnchorCoordinates:
    def __init__(self, base):
        self.base = np.asarray(base, dtype=float)
        self.eigenvalues, self.vectors = np.linalg.eigh(self.base)
        if self.eigenvalues[0] <= 0:
            raise ValueError("Layer A requires a strictly positive definite anchor")
        self.identity = np.array_equal(self.base, np.eye(len(base)))
        self.rows, self.cols = np.triu_indices(len(base))
        self.scale = np.sqrt(self.eigenvalues[self.rows] +
                             np.where(self.rows == self.cols, 0, self.eigenvalues[self.cols]))

    def encode(self, Y):
        p = len(self.base)
        if self.identity:
            return svec(psd_sqrt(Y) - np.eye(p))
        lift = CovarianceSample(Y).evaluate(self.base).maps() - np.eye(p)
        in_basis = self.vectors.T @ lift @ self.vectors
        return in_basis[:, self.rows, self.cols] * self.scale

    def decode(self, vector):
        p = len(self.base)
        if self.identity:
            factor = project_psd(smat(vector, p) + np.eye(p))
            return sym(factor @ factor)
        lift = np.empty((p, p))
        lift[self.rows, self.cols] = vector / self.scale
        lift[self.cols, self.rows] = lift[self.rows, self.cols]
        return bw_exp(self.base, self.vectors @ lift @ self.vectors.T, project=True)

    def decode_lift(self, contrast):
        p = len(self.base)
        lift = np.empty((p, p))
        lift[self.rows, self.cols] = contrast / self.scale
        lift[self.cols, self.rows] = lift[self.rows, self.cols]
        return self.vectors @ lift @ self.vectors.T @ psd_sqrt(self.base)


def _choose_split(X, scores, indices, features, config):
    best = None
    count = len(indices)
    for feature in features:
        order = indices[np.argsort(X[indices, feature], kind="stable")]
        values = X[order, feature]
        cuts = np.arange(config.min_leaf_size, count - config.min_leaf_size + 1)
        cuts = cuts[values[cuts - 1] < values[cuts]]
        if not len(cuts):
            continue
        if len(cuts) > config.max_cutpoints:
            cuts = cuts[np.linspace(0, len(cuts) - 1, config.max_cutpoints, dtype=int)]
        sums = np.cumsum(scores[order], axis=0)
        left = sums[cuts - 1] / cuts[:, None]
        right = (sums[-1] - sums[cuts - 1]) / (count - cuts)[:, None]
        criterion = np.sum((left - right)**2, axis=1) * cuts * (count - cuts) / count**2
        choice = np.argmax(criterion)
        if best is None or criterion[choice] > best[0]:
            cut = cuts[choice]
            threshold = (values[cut - 1] + values[cut]) / 2
            best = (criterion[choice], feature, threshold, order[:cut], order[cut:])
    return best


@dataclass(frozen=True)
class _Tree:
    nodes: tuple
    occupancy: np.ndarray

    def leaves(self, X):
        leaves = np.empty(len(X), dtype=int)

        def descend(node, indices):
            if not len(indices):
                return
            split = self.nodes[node]
            if split is None:
                leaves[indices] = node
            else:
                feature, threshold, left, right = split
                is_left = X[indices, feature] <= threshold
                descend(left, indices[is_left])
                descend(right, indices[~is_left])

        descend(0, np.arange(len(X)))
        return leaves


def fit_forest(X, scores, config, *, split_rule=None):
    """Learn partitions using the design set exclusively."""
    if split_rule is None:
        split_rule = _choose_split
    rng = np.random.default_rng(config.seed)
    n, d = X.shape
    if n == 0 or d == 0:
        raise ValueError("The independent design data are empty")
    if (config.n_trees < 1 or config.min_leaf_size < 1 or config.max_cutpoints < 1
            or not 0 < config.subsample <= 1 or not 0 < config.max_features <= 1):
        raise ValueError("Invalid forest configuration")
    sample_size = min(n, max(2 * config.min_leaf_size, round(n * config.subsample)))
    feature_count = max(1, min(d, round(d * config.max_features)))
    trees = []
    for _ in range(config.n_trees):
        nodes = []

        def grow(indices, depth):
            node = len(nodes)
            nodes.append(None)
            if len(indices) >= 2 * config.min_leaf_size and depth < config.max_depth:
                features = np.sort(rng.choice(d, feature_count, replace=False))
                best = split_rule(X, scores, indices, features, config)
                if best is not None and best[0] > 1e-24:
                    _, feature, threshold, left, right = best
                    nodes[node] = feature, threshold, grow(left, depth + 1), grow(right, depth + 1)
            return node

        grow(rng.choice(n, sample_size, replace=False), 0)
        provisional = _Tree(tuple(nodes), np.empty(0))
        counts = np.bincount(provisional.leaves(X), minlength=len(nodes))
        trees.append(_Tree(tuple(nodes), counts / n))
    return trees


def forest_kernel(trees, evaluation_X, queries):
    kernel = np.zeros((len(queries), len(evaluation_X)))
    for tree in trees:
        sample_leaves = tree.leaves(evaluation_X)
        query_leaves = tree.leaves(queries)
        kernel += (query_leaves[:, None] == sample_leaves) / tree.occupancy[query_leaves, None]
    return kernel / len(trees)


def normalize_kernel(kernel):
    mass = kernel.sum(axis=-1, keepdims=True)
    return np.divide(kernel, mass, out=np.zeros_like(kernel, dtype=float), where=mass > 0)


@dataclass
class MatrixPrediction:
    matrices: np.ndarray
    statuses: list
    diagnostics: list
    alpha: np.ndarray | None = None
    influences: np.ndarray | None = None
    layer_a: np.ndarray | None = None

    @property
    def accepted(self):
        return np.asarray([[status == "ok" for status in pair] for pair in self.statuses])


class MatrixDRForest:
    """Use common orthogonal contrast splitting in a specified matrix embedding."""

    def __init__(self, config=ForestConfig()):
        self.config = config

    def _fit(self, cohort, coordinates):
        self.cohort = cohort
        self.coordinates = coordinates
        split = cohort.split
        design_response = coordinates.encode(cohort.Y[split.design])
        gamma = [cohort.design_scores.scores(design_response, arm) for arm in (0, 1)]
        self.trees = fit_forest(cohort.X[split.design], gamma[1] - gamma[0], self.config)
        evaluation_response = coordinates.encode(cohort.Y[split.evaluation])
        self.arm_scores = [cohort.evaluation_scores.scores(evaluation_response, arm) for arm in (0, 1)]
        return self

    def alpha(self, queries=None, *, subgroup_masks=None):
        evaluation_X = self.cohort.evaluation_scores.X
        if subgroup_masks is not None:
            masks = np.asarray(subgroup_masks, dtype=float)
            if masks.ndim != 2 or masks.shape[1] != len(evaluation_X):
                raise ValueError("Subgroup masks must use evaluation observation order")
            if not np.all(np.isfinite(masks)) or np.any(masks < 0):
                raise ValueError("Subgroup kernels must be finite and nonnegative")
            return normalize_kernel(masks)
        return normalize_kernel(forest_kernel(self.trees, evaluation_X, np.atleast_2d(queries)))

    def predict(self, queries=None, *, subgroup_masks=None):
        alpha = self.alpha(queries, subgroup_masks=subgroup_masks)
        p = self.cohort.dimension
        matrices = np.full((len(alpha), 2, p, p), np.nan)
        statuses, diagnostics = [], []
        arm_means = [alpha @ arm_scores for arm_scores in self.arm_scores]
        for row, weights in enumerate(alpha):
            row_status, row_diagnostics = [], []
            for arm in (0, 1):
                local_support = (weights > 0) & (self.cohort.evaluation_scores.a == arm)
                status = "ok" if np.any(local_support) else "unsupported"
                row_status.append(status)
                row_diagnostics.append({"local_arm_count": int(np.sum(local_support))})
                if status == "ok":
                    matrices[row, arm] = self.coordinates.decode(arm_means[arm][row])
            statuses.append(row_status)
            diagnostics.append(row_diagnostics)
        return MatrixPrediction(matrices, statuses, diagnostics, alpha,
                                layer_a=arm_means[1] - arm_means[0])


def _fit_diagnostics(fit, local_support, weights, local_weights):
    return {
        "local_arm_count": int(np.sum(local_support)),
        "effective_local_sample_size": float(1 / (local_weights @ local_weights)),
        "min_signed_weight": float(np.min(weights)),
        "solver_iterations": fit.iterations,
        "solver_residual": float(fit.residual) if np.isfinite(fit.residual) else None,
        "solver_condition": fit.condition,
        "guard_rejections": fit.guard_rejections,
    }


class BWCausalForest(MatrixDRForest):
    def fit(self, cohort, *, anchor=None):
        if anchor is None:
            index = cohort.split.anchor
            controls = cohort.Y[index[cohort.a[index] == 0]]
            if not len(controls):
                raise ValueError("No controls on the independent anchor split")
            pilot = bw_barycenter(controls)
            if not pilot.accepted:
                raise ValueError("The independent control anchor barycenter failed")
            scale = max(np.trace(pilot.covariance) / cohort.dimension, 1e-12)
            anchor = pilot.covariance + 1e-6 * scale * np.eye(cohort.dimension)
        self._fit(cohort, AnchorCoordinates(anchor))
        a = cohort.evaluation_scores.a
        outcomes = cohort.Y[cohort.split.evaluation]
        self.arm_samples = [CovarianceSample(outcomes[a == arm]) for arm in (0, 1)]
        return self

    def predict_layer_a(self, queries=None, *, subgroup_masks=None):
        return super().predict(queries, subgroup_masks=subgroup_masks)

    def predict(self, queries=None, *, subgroup_masks=None, variant="root", with_influence=False):
        if variant not in {"root", "one_step", "ipw", "unaugmented", "entropy"}:
            raise ValueError(f"Unknown BW variant {variant}")
        if with_influence and variant != "root":
            raise ValueError("Influences are defined here for a converged augmented root")
        alpha = self.alpha(queries, subgroup_masks=subgroup_masks)
        cohort = self.cohort
        n, p = len(cohort.split.evaluation), cohort.dimension
        matrices = np.full((len(alpha), 2, p, p), np.nan)
        influence = np.full((len(alpha), 2, n, p, p), np.nan) if with_influence else None
        status = [["unsupported", "unsupported"] for _ in alpha]
        diagnostics = [[{}, {}] for _ in alpha]
        scores = cohort.evaluation_scores
        for arm in (0, 1):
            arm_mask = scores.a == arm
            sample = self.arm_samples[arm]
            signed = scores.equivalent_weights(alpha, arm)[:, arm_mask]
            ipw = alpha[:, arm_mask] * scores.inverse_probability(arm)[arm_mask]
            for row, weights in enumerate(signed):
                arm_weights = ipw[row]
                local_support = arm_weights > 0
                if not np.any(local_support):
                    continue
                if variant == "unaugmented":
                    fit = bw_barycenter(sample, alpha[row, arm_mask])
                elif variant == "entropy":
                    from .baselines import entropy_balanced_weights

                    calibrated = entropy_balanced_weights(arm_weights, scores.features[arm_mask],
                                                         alpha[row] @ scores.features)
                    if calibrated is None:
                        fit = BarycenterFit(None, "balance_failed", 0, np.inf)
                    else:
                        fit = bw_barycenter(sample, calibrated)
                else:
                    pilot_tol = 1e-6 if variant == "root" else 1e-9
                    pilot = bw_barycenter(sample, arm_weights, tolerance=pilot_tol, max_iter=500)
                    if variant == "ipw":
                        fit = pilot
                    elif not pilot.accepted:
                        fit = BarycenterFit(None, f"pilot_{pilot.status}", 0, np.inf)
                    else:
                        base = pilot.covariance
                        magnitude = max(np.trace(base) / p, np.finfo(float).tiny)
                        if np.linalg.eigvalsh(base)[0] <= 1e-13 * magnitude:
                            base = base + 1e-8 * magnitude * np.eye(p)
                        fit = solve_augmented_barycenter(sample, weights, base,
                                                        one_step=(variant == "one_step"))
                status[row][arm] = fit.status
                diagnostics[row][arm] = _fit_diagnostics(fit, local_support, weights, alpha[row])
                if not fit.accepted:
                    continue
                if with_influence:
                    raw = np.zeros((n, p, p))
                    raw[arm_mask] = np.eye(p) - sample.evaluate(fit.covariance).maps()
                    individual = scores.scores(raw, arm)
                    centered = individual - np.einsum("n,nij->ij", alpha[row], individual)
                    active_rows = alpha[row] > 0
                    solved = np.zeros_like(centered)
                    try:
                        if np.any(active_rows):
                            solved[active_rows] = fit.hessian.solve(centered[active_rows])
                        influence[row, arm] = -alpha[row, :, None, None] * solved
                    except np.linalg.LinAlgError:
                        status[row][arm] = "influence_failure"
                        continue
                matrices[row, arm] = fit.covariance
        layer_a = alpha @ (self.arm_scores[1] - self.arm_scores[0])
        return MatrixPrediction(matrices, status, diagnostics, alpha, influence, layer_a)
