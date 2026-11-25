"""Stable Bures–Wasserstein geometry and guarded Fréchet score estimation."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, cg, eigsh


def sym(value):
    value = np.asarray(value, dtype=float)
    return (value + value.swapaxes(-1, -2)) / 2


def _eigh_psd(value):
    values, vectors = np.linalg.eigh(sym(value))
    scale = np.max(np.abs(values), axis=-1, keepdims=True)
    tolerance = 32 * np.finfo(float).eps * value.shape[-1] * scale
    if np.any(values < -tolerance):
        raise ValueError("Expected positive semidefinite covariance matrices")
    return np.where(values > tolerance, values, 0.0), vectors


def psd_sqrt(value):
    values, vectors = _eigh_psd(np.asarray(value))
    return sym((vectors * np.sqrt(values)[..., None, :]) @ vectors.swapaxes(-1, -2))


def project_psd(value, floor=0.0):
    values, vectors = np.linalg.eigh(sym(value))
    return sym((vectors * np.maximum(values, floor)[..., None, :]) @ vectors.swapaxes(-1, -2))


def svec(matrix):
    """Frobenius-isometric upper-triangle vectorization."""
    rows, cols = np.triu_indices(matrix.shape[-1])
    return matrix[..., rows, cols] * np.where(rows == cols, 1.0, np.sqrt(2.0))


def smat(vector, dimension):
    rows, cols = np.triu_indices(dimension)
    shape = vector.shape[:-1] + (dimension, dimension)
    result = np.empty(shape, dtype=float)
    result[..., rows, cols] = vector / np.where(rows == cols, 1.0, np.sqrt(2.0))
    result[..., cols, rows] = result[..., rows, cols]
    return result


def bw_distance(left, right):
    """Procrustes distance, including on the PSD boundary and near equality."""
    left_root, right_root = psd_sqrt(left), psd_sqrt(right)
    u, _, vt = np.linalg.svd(right_root @ left_root)
    return np.linalg.norm(left_root - right_root @ u @ vt, axis=(-2, -1))


def _is_interior(covariance):
    values = np.linalg.eigvalsh(sym(covariance))
    return values[0] > 64 * np.finfo(float).eps * len(values) * values[-1]


def lyapunov_solve(covariance, matrix):
    eigenvalues, basis = np.linalg.eigh(sym(covariance))
    if eigenvalues[0] <= 0:
        raise ValueError("Lyapunov solve requires a positive definite base")
    denominators = eigenvalues[:, None] + eigenvalues
    return sym(basis @ ((basis.T @ matrix @ basis) / denominators) @ basis.T)


def bw_exp(base, lift, *, project=False, floor=0.0):
    """Use the symmetric lift U in (I + U) base (I + U).

    A singular multiplier reaches the PSD boundary. An indefinite one leaves
    the minimizing geodesic domain and requires an explicit projection.
    """
    multiplier = sym(lift) + np.eye(base.shape[-1])
    eigenvalues = np.linalg.eigvalsh(multiplier)
    if project:
        multiplier = project_psd(multiplier, floor)
    elif np.min(eigenvalues) < -1e-12:
        raise ValueError("BW exponential multiplier has a negative eigenvalue")
    return sym(multiplier @ base @ multiplier)


class CovarianceSample:
    """Precompute stable support factors, grouped by the retained rank."""

    def __init__(self, matrices):
        self.matrices = np.asarray(matrices, dtype=float)
        if self.matrices.ndim != 3 or len(self.matrices) == 0:
            raise ValueError("Expected at least one covariance")
        if self.matrices.shape[-2] != self.matrices.shape[-1]:
            raise ValueError("Expected square covariance matrices")
        if not np.allclose(self.matrices, self.matrices.swapaxes(-1, -2), rtol=1e-9, atol=1e-12):
            raise ValueError("Expected symmetric covariance matrices")
        values, vectors = _eigh_psd(self.matrices)
        ranks = np.count_nonzero(values, axis=-1)
        self.dimension = self.matrices.shape[-1]
        self.groups = []
        for rank in np.unique(ranks):
            if rank == 0:
                continue
            indices = np.flatnonzero(ranks == rank)
            basis = vectors[indices, :, -rank:]
            factors = basis * np.sqrt(values[indices, None, -rank:])
            self.groups.append((indices, factors))

    def evaluate(self, base):
        groups = []
        for indices, factors in self.groups:
            values, vectors = np.linalg.eigh(sym(factors.swapaxes(-1, -2) @ base @ factors))
            if np.any(values <= 0):
                raise np.linalg.LinAlgError("Transport evaluation requires an interior base")
            roots = np.sqrt(values)
            scaled = (factors @ vectors) / np.sqrt(roots[:, None, :])
            groups.append((indices, scaled, roots))
        return TransportEvaluation(self.dimension, len(self.matrices), groups)


@dataclass(frozen=True)
class TransportEvaluation:
    dimension: int
    count: int
    groups: list

    def maps(self):
        result = np.zeros((self.count, self.dimension, self.dimension))
        for indices, scaled, _ in self.groups:
            result[indices] = scaled @ scaled.swapaxes(-1, -2)
        return sym(result)

    def score(self, weights):
        weighted_map = np.zeros((self.dimension, self.dimension))
        for indices, scaled, _ in self.groups:
            weighted_map += np.einsum(
                "n,nir,njr->ij", weights[indices], scaled, scaled, optimize=True
            )
        return sym(np.eye(self.dimension) * np.sum(weights) - weighted_map)

    def hessian_action(self, weights, direction):
        result = np.zeros_like(direction)
        for indices, scaled, roots in self.groups:
            transpose = scaled.swapaxes(-1, -2)
            inner = (transpose @ direction @ scaled) / (roots[:, None, :] + roots[:, :, None])
            result += np.einsum("n,nij->ij", weights[indices], scaled @ inner @ transpose)
        return sym(result)

    def dense_hessian(self, weights):
        p = self.dimension
        dimension = p * (p + 1) // 2
        result = np.zeros((dimension, dimension))
        for indices, basis, roots in self.groups:
            rank_rows, rank_cols = np.triu_indices(roots.shape[-1])
            multiplicity = np.where(rank_rows == rank_cols, 1.0, 2.0)
            for start in range(0, len(indices), 16):
                part = slice(start, start + 16)
                left = basis[part][..., rank_rows].swapaxes(-1, -2)
                right = basis[part][..., rank_cols].swapaxes(-1, -2)
                outer = left[..., :, None] * right[..., None, :]
                directions = svec(sym(outer)).reshape(-1, dimension)
                denominators = roots[part][:, rank_rows] + roots[part][:, rank_cols]
                factors = weights[indices[part], None] * multiplicity / denominators
                result += directions.T @ (directions * factors.reshape(-1, 1))
        return sym(result)


def bw_transport(base, outcome):
    if not _is_interior(base):
        raise ValueError("BW transport requires a positive definite base")
    return CovarianceSample(np.asarray(outcome)[None]).evaluate(base).maps()[0]


def bw_log_psi(base, outcome):
    return (bw_transport(base, outcome) - np.eye(base.shape[0])) @ psd_sqrt(base)


def frechet_score(base, outcome):
    return np.eye(base.shape[0]) - bw_transport(base, outcome)


def frechet_hessian_action(base, outcome, direction):
    evaluation = CovarianceSample(np.asarray(outcome)[None]).evaluate(base)
    return evaluation.hessian_action(np.ones(1), direction)


@dataclass
class BarycenterFit:
    covariance: np.ndarray | None
    status: str
    iterations: int
    residual: float
    condition: float | None = None
    guard_rejections: int = 0
    hessian: object = None

    @property
    def accepted(self):
        return self.status == "ok"


def _positive_weights(sample, weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(sample.matrices),) or np.any(weights < 0):
        raise ValueError("Barycenter weights must be nonnegative and align with observations")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Barycenter weights must be finite")
    return weights


def bw_barycenter(outcomes, weights=None, *, tolerance=1e-9, max_iter=500):
    """Nonnegative Fréchet barycenter on the joint outcome support."""
    sample = outcomes if isinstance(outcomes, CovarianceSample) else CovarianceSample(outcomes)
    if weights is None:
        weights = np.ones(len(sample.matrices))
    weights = _positive_weights(sample, weights)
    if weights.sum() <= 0:
        return BarycenterFit(None, "unsupported", 0, np.inf)
    weights = weights / weights.sum()
    arithmetic = np.einsum("n,nij->ij", weights, sample.matrices)
    values, basis = _eigh_psd(arithmetic)
    positive = values > 0
    if not np.any(positive):
        return BarycenterFit(np.zeros_like(arithmetic), "ok", 0, 0.0)
    if not np.all(positive):
        support = basis[:, positive]
        reduced = bw_barycenter(support.T @ sample.matrices[weights > 0] @ support, weights[weights > 0],
                                tolerance=tolerance, max_iter=max_iter)
        lifted = None if reduced.covariance is None else sym(support @ reduced.covariance @ support.T)
        return BarycenterFit(lifted, reduced.status, reduced.iterations, reduced.residual)
    active = weights > 0
    if not np.all(active):
        sample = CovarianceSample(sample.matrices[active])
        weights = weights[active]
    base = arithmetic
    norm_scale = max(np.linalg.norm(base), np.finfo(float).tiny)
    residual = np.inf
    for iteration in range(1, max_iter + 1):
        try:
            score = sample.evaluate(base).score(weights)
        except np.linalg.LinAlgError:
            return BarycenterFit(base, "boundary", iteration, residual)
        multiplier = np.eye(sample.dimension) - score
        update = sym(multiplier @ base @ multiplier)
        residual = np.linalg.norm(update - base) / norm_scale
        if residual <= tolerance:
            return BarycenterFit(update, "ok", iteration, residual)
        base = update
    return BarycenterFit(base, "nonconverged", max_iter, residual)


class _Hessian:
    def __init__(self, evaluation, weights):
        self.p = evaluation.dimension
        dimension = self.p * (self.p + 1) // 2
        self._evaluation = evaluation
        self._weights = weights
        self.dense = None
        self.operator = None
        if self.p <= 20:
            self.dense = evaluation.dense_hessian(weights)
            estimates, self.basis = np.linalg.eigh(self.dense)
            self.eigenvalues = estimates
            self.minimum, self.maximum = estimates[0], estimates[-1]
        else:
            def product(vector):
                return svec(evaluation.hessian_action(weights, smat(vector, self.p)))

            self.operator = LinearOperator((dimension, dimension), matvec=product, dtype=float)
            initial = np.random.default_rng(253).normal(size=dimension)
            self.minimum = eigsh(self.operator, k=1, which="SA", tol=1e-5, maxiter=200,
                                 v0=initial, return_eigenvectors=False)[0]
            self.maximum = eigsh(self.operator, k=1, which="LA", tol=1e-5, maxiter=200,
                                 v0=initial, return_eigenvectors=False)[0]

    @property
    def usable(self):
        return self.minimum > 1e-10 * max(1, self.maximum)

    def _ensure_dense(self):
        if self.dense is None:
            self.dense = self._evaluation.dense_hessian(self._weights)
            self.eigenvalues, self.basis = np.linalg.eigh(self.dense)
            self.minimum, self.maximum = self.eigenvalues[0], self.eigenvalues[-1]
            if not self.usable:
                raise np.linalg.LinAlgError("Dense Hessian is not positive definite")

    def solve(self, right_hand_side):
        vector = svec(right_hand_side)
        flat = vector.reshape(-1, vector.shape[-1])
        if self.dense is None and len(flat) >= 20 and self.p <= 50:
            self._ensure_dense()
        if self.dense is not None:
            return smat(((vector @ self.basis) / self.eigenvalues) @ self.basis.T, self.p)
        result = np.empty_like(flat)
        for index, row in enumerate(flat):
            result[index], info = cg(self.operator, row, rtol=1e-9, atol=1e-12, maxiter=350)
            if info:
                raise np.linalg.LinAlgError("Matrix-free Hessian solve failed")
        return smat(result.reshape(vector.shape), self.p)


def solve_augmented_barycenter(outcomes, weights, pilot, *, tolerance=1e-8, max_iter=30, one_step=False):
    """Solve the actual signed augmented score, with local curvature and fold guards."""
    sample = outcomes if isinstance(outcomes, CovarianceSample) else CovarianceSample(outcomes)
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights)) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise ValueError("Signed augmentation weights must sum to one")
    if weights.shape != (len(sample.matrices),):
        raise ValueError("Signed augmentation weights must align with observations")
    if pilot is None or not _is_interior(pilot):
        return BarycenterFit(pilot, "unsupported_interior", 0, np.inf)
    active = weights != 0
    if not np.all(active):
        sample = CovarianceSample(sample.matrices[active])
        weights = weights[active]
    base = sym(pilot)
    guard_rejections = 0
    for iteration in range(max_iter + 1):
        evaluation = sample.evaluate(base)
        score = evaluation.score(weights)
        residual = np.linalg.norm(score)
        try:
            hessian = _Hessian(evaluation, weights)
        except (np.linalg.LinAlgError, ArpackNoConvergence):
            return BarycenterFit(base, "hessian_failure", iteration, residual,
                                 guard_rejections=guard_rejections)
        if not hessian.usable:
            return BarycenterFit(base, "nonpositive_hessian", iteration, residual,
                                 guard_rejections=guard_rejections)
        condition = hessian.maximum / hessian.minimum
        if residual <= tolerance or (one_step and iteration == 1):
            return BarycenterFit(base, "ok", iteration, residual, condition, guard_rejections, hessian)
        if iteration == max_iter:
            break
        try:
            direction = lyapunov_solve(base, -hessian.solve(score))
        except np.linalg.LinAlgError:
            return BarycenterFit(base, "hessian_failure", iteration, residual, condition, guard_rejections)
        minimum_direction = np.min(np.linalg.eigvalsh(direction))
        for backtrack in range(15):
            step = 2.0 ** (-backtrack)
            if 1 + step * minimum_direction <= 1e-6:
                guard_rejections += 1
                continue
            candidate = bw_exp(base, step * direction)
            if not np.all(np.isfinite(candidate)) or not _is_interior(candidate):
                guard_rejections += 1
                continue
            try:
                candidate_residual = np.linalg.norm(sample.evaluate(candidate).score(weights))
            except np.linalg.LinAlgError:
                guard_rejections += 1
                continue
            if candidate_residual <= (1 - 1e-4 * step) * residual:
                base = candidate
                break
        else:
            return BarycenterFit(base, "stalled", iteration, residual, condition, guard_rejections)
    return BarycenterFit(base, "nonconverged", max_iter, residual, condition, guard_rejections)


def bw_onestep_aipw_update(outcomes, weights, pilot):
    return solve_augmented_barycenter(outcomes, weights, pilot, one_step=True)


def projector(covariance, rank):
    if not 1 <= rank < covariance.shape[0]:
        raise ValueError("Requested spectral rank must lie between one and p−1")
    _, vectors = np.linalg.eigh(sym(covariance))
    subspace = vectors[:, -rank:]
    return sym(subspace @ subspace.T)


def spectral_projector_if(covariance, perturbation, rank):
    values, vectors = np.linalg.eigh(sym(covariance))
    if not 1 <= rank < len(values):
        raise ValueError("Requested spectral rank must lie between one and p−1")
    top, bottom = vectors[:, -rank:], vectors[:, :-rank]
    gaps = values[-rank:][None, :] - values[:-rank, None]
    if np.min(gaps) <= 0:
        raise ValueError("The rank boundary crosses an eigenvalue tie")
    cross = (bottom.T @ perturbation @ top) / gaps
    derivative = bottom @ cross @ top.T
    return derivative + derivative.swapaxes(-1, -2)
