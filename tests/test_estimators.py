import numpy as np

from bw_causal import (
    BWCausalForest, BWFrechetGlobalTLearner, BWFrechetTLearner, BWRFWLFRTLearner,
    ElementwiseDRForest, ForestConfig, LogCholeskyDRForest, LogEuclideanDRForest,
    SqrtEuclideanDRForest, prepare_cohort, simultaneous_regions, spectral_targets,
)
from bw_causal.baselines import entropy_balanced_weights
from bw_causal.geometry import bw_barycenter, bw_distance, solve_augmented_barycenter


def _example():
    rng = np.random.default_rng(56)
    X = rng.integers(2, size=(240, 1)).astype(float)
    a = rng.binomial(1, 0.25 + 0.4 * X[:, 0])
    base = np.diag([5.0, 2.4, 1.4, 0.8])
    direction = np.eye(4)[[1, 0, 3, 2]]
    G = np.eye(4) + 0.06 * rng.choice([-1.0, 1.0], size=240)[:, None, None] * direction
    Y = G @ base @ G
    return X, a, Y


def test_signed_representation_and_identity_equivalence_on_shared_honest_split():
    X, a, Y = _example()
    cohort = prepare_cohort(X, a, Y, seed=13)
    evaluation = cohort.evaluation_scores
    response = Y[cohort.split.evaluation]
    alpha = np.random.default_rng(3).dirichlet(np.ones(len(evaluation.a)), size=3)
    for arm in [0, 1]:
        signed = evaluation.equivalent_weights(alpha, arm)
        assert np.allclose(signed.sum(axis=1), 1, rtol=0, atol=1e-12)
        assert np.all(signed[:, evaluation.a != arm] == 0)
        raw = evaluation.scores(response, arm)
        assert np.allclose(alpha @ raw.reshape(len(raw), -1), signed @ response.reshape(len(raw), -1),
                           rtol=0, atol=1e-12)
    config = ForestConfig(n_trees=10, min_leaf_size=6, seed=3)
    bw = BWCausalForest(config).fit(cohort, anchor=np.eye(4))
    sqrt = SqrtEuclideanDRForest(config).fit(cohort)
    queries = [[0.0], [1.0]]
    first, second = bw.predict_layer_a(queries), sqrt.predict(queries)
    np.testing.assert_array_equal(first.alpha, second.alpha)
    np.testing.assert_array_equal(first.matrices, second.matrices)
    assert first.accepted.all()
    changed = Y.copy()
    changed[cohort.split.evaluation] *= 2
    same = BWCausalForest(config).fit(prepare_cohort(X, a, changed, split=cohort.split), anchor=np.eye(4))
    np.testing.assert_array_equal(same.alpha(queries), first.alpha)


def test_all_estimators_validate_psd_and_repeatable_joint_regions():
    cohort = prepare_cohort(*_example(), seed=23)
    config = ForestConfig(n_trees=6, min_leaf_size=6, seed=8)
    estimators = [
        BWCausalForest(config), SqrtEuclideanDRForest(config), LogEuclideanDRForest(config),
        LogCholeskyDRForest(config), ElementwiseDRForest(config), BWFrechetTLearner(config),
        BWRFWLFRTLearner(config), BWFrechetGlobalTLearner(),
    ]
    for estimator in estimators:
        result = estimator.fit(cohort).predict([[0.0], [1.0]])
        assert result.accepted.all(), (type(estimator).__name__, result.statuses)
        assert np.min(np.linalg.eigvalsh(result.matrices)) > -1e-10

    bw = estimators[0]
    strata = cohort.evaluation_scores.X[:, 0] == np.array([[0], [1]])
    prediction = bw.predict(subgroup_masks=strata, with_influence=True)
    first = simultaneous_regions(prediction, n_multiplier=59, seed=3)
    second = simultaneous_regions(prediction, n_multiplier=59, seed=3)
    assert first.critical_value == second.critical_value
    assert first.covers(spectral_targets(prediction.matrices))
    assert not first.covers({})


def test_boundary_targets_and_incompatible_calibration_are_counted():
    common_null = np.repeat(np.diag([3.0, 1.0, 0.0])[None], 10, axis=0)
    weights = np.repeat(0.1, len(common_null))
    pilot = bw_barycenter(common_null, weights)
    fit = solve_augmented_barycenter(common_null, weights, pilot.covariance)
    assert not fit.accepted and fit.status == "unsupported_interior"
    assert entropy_balanced_weights(weights, np.arange(10)[:, None], np.array([15.0])) is None


def test_rotating_rank_one_outcomes_with_interior_target():
    base = np.array([[2.0, 0.3], [0.3, 1.2]])
    directions = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    multipliers = 2 * directions[..., None] * directions[:, None, :]
    observations = multipliers @ base @ multipliers
    weights = np.repeat(0.25, 4)
    pilot = bw_barycenter(observations, weights)
    assert pilot.accepted
    fit = solve_augmented_barycenter(observations, weights, pilot.covariance)
    assert fit.accepted and bw_distance(fit.covariance, base) < 1e-8


def test_dimensionless_kato_guard_ignores_distant_spike_noise():
    # A large noisy leading spike must not cause false abstention on a clean rank-3 boundary.
    rng = np.random.default_rng(77)
    n, p = 80, 6
    base0 = np.diag([20.0, 8.0, 1.8, 1.0, 1.0, 1.0])
    base1 = np.diag([22.0, 8.5, 2.0, 1.0, 1.0, 1.0])
    matrices = np.array([[base0, base1]])
    influences = np.zeros((1, 2, n, p, p))
    # Huge variance strictly along coordinate 0 (the distant spike), small noise on coordinates 2..5
    influences[0, 0, :, 0, 0] = rng.normal(0.0, 0.25, size=n)
    influences[0, 1, :, 0, 0] = rng.normal(0.0, 0.25, size=n)
    for arm in (0, 1):
        sub = rng.normal(0.0, 0.005, size=(n, 4, 4))
        sub = 0.5 * (sub + sub.swapaxes(-1, -2))
        influences[0, arm, :, 2:, 2:] = sub
        influences[0, arm] -= influences[0, arm].mean(axis=0, keepdims=True)
    from bw_causal.forest import MatrixPrediction

    pred = MatrixPrediction(matrices, [["ok", "ok"]], [[{}, {}]], influences=influences)
    regions = simultaneous_regions(pred, projector_rank=3, n_multiplier=99, seed=1, max_gap_ratio=1.0)
    assert (0, "eigenvalue_3") not in regions.abstentions
    assert (0, "projector_3") not in regions.abstentions
