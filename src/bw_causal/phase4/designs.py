"""Synthetic laws and BW references declared before the full sweep."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib

import numpy as np
from scipy.special import expit

from ..dgp import transport_noise
from ..geometry import bw_barycenter, bw_distance, psd_sqrt, sym


@dataclass(frozen=True)
class Data:
    X: np.ndarray
    treatment: np.ndarray
    observed: np.ndarray
    propensity: np.ndarray
    queries: np.ndarray
    truth: np.ndarray
    stratified: bool = False
    latent_truth: np.ndarray | None = None

    def subgroup_kernel(self, evaluation):
        if not self.stratified:
            return None
        return self.queries[:, 0, None] == self.X[evaluation, 0]


def rng_for(seed, purpose):
    namespace = int.from_bytes(hashlib.sha256(purpose.encode()).digest()[:4], "big")
    return np.random.default_rng(np.random.SeedSequence([seed, namespace]))


def _rotate(eigenvalues, theta, *, plane23=None):
    n, p = eigenvalues.shape
    basis = np.broadcast_to(np.eye(p), (n, p, p)).copy()
    cosine, sine = np.cos(theta), np.sin(theta)
    basis[:, 0, 0] = cosine
    basis[:, 1, 1] = cosine
    basis[:, 0, 1] = -sine
    basis[:, 1, 0] = sine
    if plane23 is not None:
        c2, s2 = np.cos(plane23), np.sin(plane23)
        basis[:, 2, 2] = c2
        basis[:, 3, 3] = c2
        basis[:, 2, 3] = -s2
        basis[:, 3, 2] = s2
    return sym((basis * eigenvalues[:, None, :]) @ basis.swapaxes(-1, -2))


def crossover_truth(X, task):
    x1, x2 = X[:, 0], X[:, 1]
    p, kappa, angle, floor = (task[key] for key in ("p", "condition", "angle", "floor"))
    scenario = task["scenario"]
    if floor == 0 and scenario != "structural":
        # A zero in the observed-rank branch does not force a singular BW target.
        floor = 0.02
    typical_bulk = 1.0
    # Place the leading spike (kappa) in coordinate 0 and a unit bulk mode in coordinate 1 so
    # (0, 1) rotation rotates the spike directly into the bulk with effective ratio kappa.
    second_spike = 1.5 if kappa == 2 else min(0.25 * kappa, 8.0)
    tail = np.linspace(1.35, typical_bulk, p - 4) if p > 4 else np.empty(0)
    spectrum = np.r_[kappa, typical_bulk, second_spike, tail, floor]
    regime = 0.5 * (1.0 + np.tanh(2.5 * x1))
    steps = []
    for arm in (0, 1):
        if scenario == "spiked":
            theta = 0.24 * np.sin(np.pi * x1) + np.deg2rad(angle) * arm * (0.2 + 0.8 * regime)
        elif scenario == "commuting":
            theta = np.zeros_like(x1)
        else:
            theta = np.deg2rad(angle) * (0.14 * np.sin(np.pi * x1) + arm * (0.35 + 0.65 * regime + 0.15 * x2))
        if scenario == "recovery":
            theta += 0.18 * np.sin(np.pi * x1 * x2)
        scale = np.exp(0.18 * np.sin(np.pi * x1) + 0.10 * x2 + arm * (0.08 + 0.28 * regime))
        eig = scale[:, None] * spectrum
        # The two design axes are separated from the fixed floor at positive settings.
        eig[:, -1] = floor
        steps.append(_rotate(eig, theta))
    return np.stack(steps, axis=1)


def _matrix_noise(centers, rng, task):
    mode = task["scenario"]
    if mode == "commuting":
        factors = 1 + rng.uniform(-0.28, 0.28, (len(centers), centers.shape[-1]))
        return centers * factors[:, :, None] * factors[:, None, :]
    if mode == "structural":
        result = np.zeros_like(centers)
        result[:, :-1, :-1] = transport_noise(centers[:, :-1, :-1], rng, amplitude=0.4)
        return result
    if task["floor"] == 0:
        return sym(transport_noise(centers, rng, singular=True))
    amplitude = 0.16 if mode == "recovery" else 0.48
    return sym(transport_noise(centers, rng, amplitude=amplitude))


def crossover(task, core, *, validation=False):
    scenario = task["scenario"]
    n = core["validation_n"] if validation else task["n"]
    rng = rng_for(task["seed"], "core_validation" if validation else "core_training")
    if scenario == "null":
        return _group_data(task, n, rng, null=True)
    X = rng.uniform(-1, 1, (n, 2))
    e = expit(0.12 + 0.90 * X[:, 0] - 0.60 * X[:, 1])
    a = rng.binomial(1, e)
    roots = crossover_truth(X, task)
    Y = _matrix_noise(roots[np.arange(n), a], rng, task)
    queries = np.asarray(core["test_queries"])
    return Data(X, a, Y, e, queries, crossover_truth(queries, task))


def _group_roots(p, groups, angle, gap, *, null=False):
    root = []
    for arm in (0, 1):
        applied = 0 if null else arm
        g = np.arange(groups) - (groups - 1) / 2
        scale = np.exp(0.07 * g + applied * (0.09 + 0.035 * g))
        # In each arm/group third-to-fourth gap is exactly the requested absolute gap.
        eig = scale[:, None] * np.broadcast_to(np.r_[7.0, 3.8, 1., np.ones(p - 3)], (groups, p))
        eig[:, 2] += gap
        # Rotate both (0, 1) and the rank-3 boundary (2, 3) so the rank-3 projector contrast is non-null.
        roots = _rotate(eig, 0.13 * g + applied * angle,
                        plane23=0.10 * g + applied * 0.65 * angle)
        root.append(roots)
    return np.stack(root, axis=1)


def _group_data(task, n, rng, *, null=False):
    groups = 5
    g = rng.integers(0, groups, n)
    x = rng.uniform(-1, 1, n)
    X = np.column_stack([g, x])
    logits = 0.64 * (g - 2) + 1.45 * (x**2 - 0.33) - 0.30 * x
    propensity = np.clip(expit(logits), 0.08, 0.92)
    a = rng.binomial(1, propensity)
    roots = _group_roots(task["p"], groups, 0.55, task.get("gap", 1.0), null=null)
    noise = 0.35 if not null else 0.26
    Y = sym(transport_noise(roots[g, a], rng, amplitude=noise))
    queries = np.column_stack([np.arange(groups), np.zeros(groups)])
    return Data(X, a, Y, propensity, queries, roots, stratified=True)


def coverage(task, *, validation=False, n_validation=72):
    rng = rng_for(task["seed"], "coverage_validation" if validation else "coverage_training")
    return _group_data(task, n_validation if validation else task["n"], rng)


def _checked_barycenter(Y):
    fit = bw_barycenter(Y, tolerance=3e-9, max_iter=350)
    if not fit.accepted:
        raise ValueError(f"Truth construction failed with status {fit.status}")
    return fit.covariance


def misspecification(task):
    n, p = task["n"], task["p"]
    rng = rng_for(task["seed"], "misspecification_training")
    g, z = rng.integers(0, 2, n), rng.integers(0, 3, n)
    z1, z2 = z == 1, z == 2
    X = np.column_stack([g, z1, z2, g * z1, g * z2])
    e = np.array([[0.22, 0.48, 0.77], [0.29, 0.54, 0.81]])[g, z]
    a = rng.binomial(1, e)
    rows_g, rows_z = np.indices((2, 3))
    rows_g, rows_z = rows_g.reshape(-1), rows_z.reshape(-1)
    outcomes = []
    for arm in (0, 1):
        magnitudes = np.exp(0.13 * rows_g + 0.39 * rows_z + 0.12 * arm * (1 + rows_g))
        eig = magnitudes[:, None] * np.broadcast_to(np.linspace(3.6, 0.35, p), (6, p))
        angle = 0.12 * rows_g + 0.29 * rows_z + 0.23 * arm
        outcomes.append(_rotate(eig, angle).reshape(2, 3, p, p))
    atoms = np.stack(outcomes, axis=0)
    Y = atoms[a, g, z]
    truth = np.array([[_checked_barycenter(atoms[arm, group]) for arm in (0, 1)] for group in (0, 1)])
    queries = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    return Data(X, a, Y, e, queries, truth, stratified=True)


def _sample_covariances(covariances, period, rng):
    if period is None:
        return covariances
    n, p, _ = covariances.shape
    factors = psd_sqrt(covariances)
    samples = factors @ rng.normal(size=(n, p, period))
    return sym(samples @ samples.swapaxes(-1, -2) / period)


@lru_cache(maxsize=16)
def measurement_reference(p, period, draws):
    latent = _group_roots(p, 3, 0.49, 0.9)
    result = np.empty_like(latent)
    discrepancy = np.zeros((3, 2))
    if period is None:
        return latent, discrepancy
    for group in range(3):
        for arm in (0, 1):
            rng = rng_for(717077, f"measurement_truth_{group}_{arm}")
            centers = np.broadcast_to(latent[group, arm], (draws, p, p))
            latent_draws = transport_noise(centers, rng, amplitude=0.37)
            observed = _sample_covariances(latent_draws, period, rng)
            result[group, arm] = _checked_barycenter(observed)
            left = _checked_barycenter(observed[:draws // 2])
            right = _checked_barycenter(observed[draws // 2:])
            discrepancy[group, arm] = bw_distance(left, right)
    return result, discrepancy


def finite_measurement(task, stress, core, *, validation=False):
    n = core["validation_n"] if validation else task["n"]
    p, period = task["p"], task["period"]
    rng = rng_for(task["seed"], "measurement_validation" if validation else "measurement_training")
    g = rng.integers(0, 3, n)
    x = rng.uniform(-1, 1, n)
    X = np.column_stack([g, x])
    e = expit(-0.4 + 0.43 * g + 0.66 * x)
    a = rng.binomial(1, e)
    latent_truth = _group_roots(p, 3, 0.49, 0.9)
    centers = transport_noise(latent_truth[g, a], rng, amplitude=0.37)
    Y = _sample_covariances(centers, period, rng)
    truth, _ = measurement_reference(p, period, stress["reference_draws_per_arm_stratum"])
    queries = np.column_stack([np.arange(3), np.zeros(3)])
    return Data(X, a, Y, e, queries, truth, stratified=True, latent_truth=latent_truth)
