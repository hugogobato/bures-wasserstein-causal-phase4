"""Known BW roots for the small prototype correctness ladder."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class PrototypeSample:
    X: np.ndarray
    treatment: np.ndarray
    observed: np.ndarray
    propensity: np.ndarray
    query: np.ndarray
    truth: np.ndarray


def transport_noise(covariances, rng, *, amplitude=0.15, singular=False):
    """Generate Y=G B G with a PSD random G and conditional mean E[G]=I."""
    n, p, _ = covariances.shape
    if singular:
        direction = rng.normal(size=(n, p))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        orthogonal = np.eye(p) - direction[:, :, None] * direction[:, None, :]
        perturbation = p / (p - 1) * orthogonal
    else:
        direction = rng.normal(size=(n, p, p))
        direction = (direction + direction.swapaxes(-1, -2)) / 2
        factor = np.maximum(np.linalg.norm(direction, axis=(-2, -1)), 1e-12)
        perturbation = np.eye(p) + amplitude * direction / factor[:, None, None]
    return perturbation @ covariances @ perturbation


def _true_covariances(X, p, kind):
    group = X[..., 0]
    target = np.empty((len(X), 2, p, p))
    for arm in (0, 1):
        if kind == "anisotropic":
            eig = np.r_[50.0, 8.0, 3.0, np.linspace(1.6, 1.0, p - 3)]
            theta = 0.32 * group + arm * np.deg2rad(45)
            scale = (1 + 0.15 * group + 0.12 * arm * group)
        else:
            eig = np.r_[8.0, 4.0, 2.0, np.linspace(1.1, 0.55, p - 3)]
            treatment_effect = arm if kind == "noiseless" else 0
            theta = 0.25 * group + 0.15 * treatment_effect * (1 + group)
            scale = 1 + 0.20 * group + 0.15 * treatment_effect * (1 + group)
        rotation = np.broadcast_to(np.eye(p), (len(X), p, p)).copy()
        rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(theta)
        rotation[:, 0, 1], rotation[:, 1, 0] = -np.sin(theta), np.sin(theta)
        target[:, arm] = (rotation * (scale[:, None] * eig)[:, None, :]) @ rotation.swapaxes(-1, -2)
    return target


def prototype_sample(kind, n, p, seed, *, singular=False, n_queries=2):
    """Use fixed-stratum roots with noncommuting within-arm variation for noisy outcomes."""
    if kind not in {"noiseless", "null", "anisotropic"} or p < 4 or n_queries < 2:
        raise ValueError("Invalid prototype experiment configuration")
    rng = np.random.default_rng(seed)
    group = rng.integers(2, size=n)
    if kind == "noiseless":
        X = group[:, None].astype(float)
        propensity = 0.33 + 0.34 * group
        query = np.array([[0.0], [1.0]])
    else:
        nuisance = rng.uniform(-1, 1, n)
        X = np.column_stack((group, nuisance))
        nonlinear = 0.9 * (2 * nuisance**2 - 0.65) if kind == "null" else 0.7 * nuisance
        propensity = expit(-0.7 + 1.45 * group + nonlinear)
        query = np.column_stack((np.arange(n_queries) % 2, np.linspace(-0.75, 0.75, n_queries)))
    treatment = rng.binomial(1, propensity)
    potential_roots = _true_covariances(X, p, kind)
    observed = potential_roots[np.arange(n), treatment]
    if kind != "noiseless":
        amplitude = 0.10 if kind == "null" else 0.48
        observed = transport_noise(observed, rng, amplitude=amplitude, singular=singular)
    return PrototypeSample(X, treatment, observed, propensity, query, _true_covariances(query, p, kind))
