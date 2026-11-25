import numpy as np
import pytest

from bw_causal.geometry import (
    CovarianceSample, bw_barycenter, bw_distance, bw_exp, bw_log_psi, bw_transport,
    frechet_hessian_action, frechet_score, projector, psd_sqrt, smat, spectral_projector_if,
    svec, sym,
)


def _covariances(rank):
    rng = np.random.default_rng(412)
    basis, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    base = (basis * np.geomspace(1.0, 1e-4, 5)) @ basis.T
    raw = rng.normal(size=(3, 5, rank))
    return base, raw @ raw.swapaxes(-1, -2)


@pytest.mark.parametrize("rank", [1, 4, 5])
def test_distance_lift_roundtrip_and_identity_anchor(rank):
    base, (first, second, third) = _covariances(rank)
    assert abs(bw_distance(first, second) - bw_distance(second, first)) < 1e-10
    assert bw_distance(first, third) <= bw_distance(first, second) + bw_distance(second, third) + 1e-10
    assert abs(np.linalg.norm(bw_log_psi(base, first)) - bw_distance(base, first)) < 1e-10
    lift = bw_transport(base, first) - np.eye(5)
    assert np.linalg.norm(bw_exp(base, lift) - first) < 1e-10
    assert np.linalg.norm(bw_log_psi(np.eye(5), first) - (psd_sqrt(first) - np.eye(5))) < 1e-12
    with pytest.raises(ValueError, match="negative eigenvalue"):
        bw_exp(base, -2 * np.eye(5))


@pytest.mark.parametrize("rank", [0, 1, 4, 5])
def test_exact_gradient_and_dense_and_matrix_free_hessian(rank):
    rng = np.random.default_rng(432)
    base = np.diag([3.0, 2.0, 1.4, 0.8, 0.5])
    raw = rng.normal(size=(5, rank))
    covariance = raw @ raw.T
    direction = sym(rng.normal(size=(5, 5)))
    h = 2e-6
    dloss = (bw_distance(base + h * direction, covariance)**2
             - bw_distance(base - h * direction, covariance)**2) / (2 * h)
    assert np.isclose(dloss, np.sum(frechet_score(base, covariance) * direction), atol=1e-8)
    exact = frechet_hessian_action(base, covariance, direction)
    numerical = (frechet_score(base + h * direction, covariance)
                 - frechet_score(base - h * direction, covariance)) / (2 * h)
    assert np.linalg.norm(exact - numerical) < 1e-8
    evaluation = CovarianceSample(np.repeat(covariance[None], 20, axis=0)).evaluate(base)
    weights = np.ones(20) / 20
    dense = evaluation.dense_hessian(weights)
    assert np.linalg.norm(smat(dense @ svec(direction), 5) - exact) < 1e-10


def test_commuting_barycenter_and_common_boundary():
    observations = np.stack([np.diag([0.0, 1.0, 9.0]), np.diag([0.0, 4.0, 16.0])])
    fit = bw_barycenter(observations, np.array([0.25, 0.75]))
    assert fit.accepted
    expected = np.diag([0, 1.75**2, 3.75**2])
    assert bw_distance(fit.covariance, expected) < 1e-10


def test_projector_derivative_with_repeated_internal_eigenvalue():
    rng = np.random.default_rng(6)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    base = (q * [9, 9, 3, 2, 1, 0.5]) @ q.T
    direction = sym(rng.normal(size=(6, 6)))
    eps = 1e-5
    fd = (projector(base + eps * direction, 2) - projector(base - eps * direction, 2)) / (2 * eps)
    analytic = spectral_projector_if(base, direction, 2)
    assert np.linalg.norm(fd - analytic) < 1e-10
    with pytest.raises(ValueError, match="eigenvalue tie"):
        spectral_projector_if(np.diag([2.0, 2.0, 1.0]), np.eye(3), 1)
