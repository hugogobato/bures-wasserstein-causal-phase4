"""Matched method evaluation with explicit finite-sample failures."""

import hashlib
import resource
import signal
import time

import numpy as np
from scipy.linalg import expm_frechet

from .. import (
    BWCausalForest, BWFrechetGlobalTLearner, BWFrechetTLearner, BWRFWLFRTLearner,
    ElementwiseDRForest, ForestConfig, HonestSplit, LogCholeskyDRForest,
    LogEuclideanDRForest, SqrtEuclideanDRForest, prepare_cohort, simultaneous_regions,
    spectral_targets,
)
from ..geometry import bw_distance, bw_transport, projector, smat, sym
from ..forest import _logistic
from . import designs


BASE_METHODS = (
    "BW", "SqrtEuclidean", "LogEuclidean", "LogCholesky", "Elementwise",
    "BW_TLearner", "BW_GlobalTLearner", "BW_RFWLFR",
)
ABLATIONS = (
    "LayerA", "LayerAIdentity", "BW_RootIdentitySplit", "BW_OneStep",
    "BW_Unaugmented", "BW_IPW", "BW_Entropy",
)
FIXED_LOG = ("LogEuclidean_default", "LogCholesky_default")
CORE_METHODS = BASE_METHODS + ABLATIONS + FIXED_LOG + ("Oracle",)
DR_METHODS = ("BW", "LayerA", "BW_OneStep", "BW_TLearner", "BW_IPW", "BW_NoCrossFit")
SCALING_METHODS = BASE_METHODS
COVERAGE_METHODS = ("BW", "BW_Pointwise", "SqrtEuclidean", "LogEuclidean", "Elementwise")


def methods_for(task):
    kind = task["kind"]
    if kind == "scaling":
        return (task["method"],)
    if kind == "dr":
        return DR_METHODS
    if kind == "coverage":
        return COVERAGE_METHODS
    if task.get("scenario") == "null":
        return CORE_METHODS + ("BW_Pointwise",)
    return CORE_METHODS


def _oracle(data):
    return {
        "status": "oracle_reference", "attempted_arms": 2 * len(data.truth), "failed_arms": 0,
        "psd_violations": 0, "bw_pehe": 0., "worst_query_pehe": 0., "projector_rmse": 0.,
        "eigenvalue_mae": 0., "transport_rmse": 0., "bulk_ratio_median": 1.,
        "latent_pehe": None if data.latent_truth is None else _risk(data.truth, data.latent_truth)[0],
    }


def _risk(estimate, truth):
    row_errors = np.sum(bw_distance(estimate, truth) ** 2, axis=1)
    return float(np.sqrt(row_errors.mean())), float(np.sqrt(row_errors.max()))


def _metrics(fit, data, *, store_signed=False):
    accepted = fit.accepted & np.isfinite(fit.matrices).all(axis=(-1, -2))
    accepted_eigenvalues = np.linalg.eigvalsh(fit.matrices[accepted])
    negative = np.count_nonzero(np.min(accepted_eigenvalues, axis=-1) < -1e-10)
    result = {
        "status": "ok" if accepted.all() else "query_failure",
        "attempted_arms": int(accepted.size), "failed_arms": int(accepted.size - accepted.sum()),
        "psd_violations": int(negative), "query_statuses": fit.statuses,
        "bw_pehe": None, "worst_query_pehe": None, "projector_rmse": None,
        "eigenvalue_mae": None, "transport_rmse": None, "bulk_ratio_median": None,
        "latent_pehe": None,
    }
    query_mask = accepted.all(axis=1)
    if not query_mask.any():
        return result
    if not accepted.all():
        result["complete_query_only_pehe"] = _risk(
            fit.matrices[query_mask], data.truth[query_mask]
        )[0]
        return result
    estimate, truth = fit.matrices, data.truth
    result["bw_pehe"], result["worst_query_pehe"] = _risk(estimate, truth)
    if data.latent_truth is not None:
        result["latent_pehe"] = _risk(estimate, data.latent_truth)[0]
    eig_est, eig_true = np.linalg.eigvalsh(estimate), np.linalg.eigvalsh(truth)
    offset = np.diff(eig_est[..., -3:], axis=1) - np.diff(eig_true[..., -3:], axis=1)
    result["eigenvalue_mae"] = float(np.mean(np.abs(offset)))
    proj = lambda pair: projector(pair[1], 1) - projector(pair[0], 1)
    deltas = np.array([proj(candidate) - proj(target) for candidate, target in zip(estimate, truth)])
    result["projector_rmse"] = float(np.sqrt(np.mean(np.sum(deltas**2, axis=(1, 2)))))
    # Interior transport and positive bulk denominators only.
    positive = eig_true[..., 0] > 1e-12
    if np.all(positive):
        result["bulk_ratio_median"] = float(np.median(eig_est[..., :-2] / eig_true[..., :-2]))
    try:
        deltas = np.array([
            bw_transport(left[0], left[1]) - bw_transport(right[0], right[1])
            for left, right in zip(estimate, truth)
        ])
        transport = np.sqrt(np.mean(np.sum(deltas**2, axis=(1, 2))))
        if np.isfinite(transport):
            result["transport_rmse"] = float(transport)
    except (ValueError, np.linalg.LinAlgError):
        pass
    if store_signed:
        result["signed_contrast_errors"] = (np.diff(estimate - truth, axis=1)[:, 0]).tolist()
    return result


def _factual_validation_loss(model, validation, probability, cohort):
    if validation.stratified:
        masks = validation.queries[:, 0, None] == cohort.evaluation_scores.X[:, 0]
        prediction = model.predict(subgroup_masks=masks)
        rows = np.asarray(validation.X[:, 0], dtype=int)
    else:
        prediction = model.predict(validation.X)
        rows = np.arange(len(validation.X))
    if not np.all(prediction.accepted[rows, validation.treatment]):
        return np.inf
    estimated = prediction.matrices[rows, validation.treatment]
    observed_arm_probability = np.where(validation.treatment, probability, 1 - probability)
    losses = bw_distance(estimated, validation.observed) ** 2
    weights = 1 / observed_arm_probability
    return float(np.average(losses, weights=weights))


def _tune_log(constructor, candidates, cohort, config, validation, probability):
    models = []
    errors = {}
    for ridge in candidates:
        try:
            model = constructor(config, ridge=ridge).fit(cohort)
            loss = (_factual_validation_loss(model, validation, probability, cohort)
                    if validation is not None else 0.)
            models.append((loss, ridge, model))
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
            errors[str(ridge)] = str(error)
    if not models:
        raise ValueError(f"Every log-ridge candidate failed: {errors}")
    chosen = min(models, key=lambda row: row[:2])
    default = next((model for _, ridge, model in models if ridge == candidates[0]), None)
    return chosen[2], default, {
        "selected_ridge": chosen[1],
        "candidate_failures": errors,
        "valid_selection_loss": bool(np.isfinite(chosen[0])),
        "validation_losses": {str(ridge): loss if np.isfinite(loss) else None
                              for loss, ridge, _ in models},
    }


def _group_features(data):
    groups = int(data.queries[:, 0].max()) + 1
    return np.column_stack([data.X, data.X[:, 1] ** 2, *(data.X[:, 0] == g for g in range(1, groups))])


def _validation_propensity(validation, cohort):
    if validation is None:
        return None
    features = _group_features(validation) if validation.stratified else validation.X
    design = cohort.design_scores
    return np.clip(_logistic(design.X, design.a, features, 1.0), 0.05, 0.95)


def _prepare(task, data, core, *, no_crossfit=False):
    seed = task["seed"]
    split = HonestSplit.make(len(data.X), seed=13000 + seed, **core["split"])
    options = {"fit_on_all_rows": no_crossfit}
    prop = None
    if task["kind"] == "dr":
        nuisance = task["nuisance"]
        prop = data.propensity if nuisance in {"both", "e_only"} else np.full(len(data.X), 0.5)
        options |= {
            "score_feature_indices": None if nuisance in {"both", "m_only"} else [0],
            "use_squares": False, "ridge": 1e-8,
        }
    elif data.stratified:
        # The conditional score is constant within each declared group. Use saturated indicators
        # with minimal ridge for the outcome regressions, and include x^2 in X for propensity.
        X = _group_features(data)
        options["use_squares"] = False
        options["ridge"] = 1e-4
        options["score_feature_indices"] = list(range(3, X.shape[1]))
        cohort = prepare_cohort(X, data.treatment, data.observed, split=split, **options)
        return cohort
    return prepare_cohort(data.X, data.treatment, data.observed, split=split,
                          propensity_override=prop, **options)


def _prediction_inputs(data, cohort):
    if data.stratified:
        return {"subgroup_masks": data.subgroup_kernel(cohort.split.evaluation)}
    return {"queries": data.queries}


def _model(name, config):
    constructors = {
        "BW": BWCausalForest, "SqrtEuclidean": SqrtEuclideanDRForest,
        "LogEuclidean": LogEuclideanDRForest, "LogCholesky": LogCholeskyDRForest,
        "Elementwise": ElementwiseDRForest, "BW_TLearner": BWFrechetTLearner,
        "BW_RFWLFR": BWRFWLFRTLearner,
    }
    if name == "BW_GlobalTLearner":
        return BWFrechetGlobalTLearner()
    return constructors[name](config)


def _fit_predict(name, model, cohort, inputs, *, anchor=None):
    if name == "BW_TLearner":
        model.fit(cohort, anchor=anchor)
    else:
        model.fit(cohort)
    return model.predict(**inputs)


def _origin_status(error):
    return {"status": "exception", "error": f"{type(error).__name__}: {error}", "bw_pehe": None}


def _record(metrics, elapsed, *, fit_seconds=None):
    return metrics | {"wall_seconds": elapsed, "fit_seconds": fit_seconds,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def _baseline_derivatives(name, model, fit):
    """Plug-in embedding mean influences conditional on the fixed subgroup design."""
    p = model.cohort.dimension
    n = len(model.cohort.split.evaluation)
    fit.influences = np.full((len(fit.alpha), 2, n, p, p), np.nan)
    for group, weights in enumerate(fit.alpha):
        for arm, scores in enumerate(model.arm_scores):
            if not fit.accepted[group, arm]:
                continue
            mean = weights @ scores
            difference = (scores - mean) * weights[:, None]
            if name == "LogCholesky":
                factor = np.diag(np.exp(mean[-p:]))
                rows, cols = np.tril_indices(p, k=-1)
                factor[rows, cols] = mean[:-p]
                ridge = model.coordinates.ridge
                unguarded = factor @ factor.T - ridge * np.eye(p)
                tangents = np.eye(p) * (difference[:, -p:] * np.diag(factor))[:, None, :]
                tangents[:, rows, cols] = difference[:, :-p]
                influences = tangents @ factor.T + factor @ tangents.swapaxes(-1, -2)
            else:
                coordinate = smat(mean, p)
                tangents = smat(difference, p)
                if name == "Elementwise":
                    unguarded, influences = coordinate, tangents
                elif name == "SqrtEuclidean":
                    square_root = coordinate + np.eye(p)
                    unguarded = square_root
                    influences = square_root @ tangents + tangents @ square_root
                else:
                    values, vectors = np.linalg.eigh(coordinate)
                    ridge = model.coordinates.ridge
                    unguarded = (vectors * (np.exp(values) - ridge)) @ vectors.T
                    influences = np.array([
                        expm_frechet(coordinate, tangent, compute_expm=False) for tangent in tangents
                    ])
            if np.linalg.eigvalsh(unguarded)[0] <= 1e-9:
                fit.statuses[group][arm] = "nondifferentiable_psd_projection"
                continue
            fit.influences[group, arm] = sym(influences)
    return fit


def _coverage(fit, data, task, core, *, joint=True):
    rank = 3
    truth = spectral_targets(data.truth, projector_rank=rank)
    tied = task.get("gap") == 0
    regions = simultaneous_regions(fit, projector_rank=rank, n_multiplier=core["bootstrap_steps"],
                                   seed=16000 + task["seed"], joint=joint)
    available = []
    matrix_family = []
    spectral_family = []
    for (group, target), value in truth.items():
        if tied and target in {"eigenvalue_3", "projector_3"}:
            continue
        key = group, target
        present = key in regions.regions
        result = regions.regions[key].contains(value) if present else None
        radius = regions.regions[key].radius if present else None
        reason = regions.abstentions.get(key)
        row = {
            "stratum": group, "target": target, "covered": None if result is None else bool(result),
            "radius": radius, "abstention": reason,
        }
        available.append(row)
        (matrix_family if target == "matrix" else spectral_family).append(row)
    all_covered = lambda rows: bool(rows) and all(row["covered"] is True for row in rows)
    noncover = lambda rows: any(row["covered"] is False for row in rows)
    radii = [row["radius"] for row in available if row["radius"] is not None]
    return {
        "joint_covered": None if tied else all_covered(available),
        "spectral_joint_covered": None if tied else all_covered(spectral_family),
        "matrix_joint_covered": all_covered(matrix_family),
        "identifiable_joint_covered": all_covered(available),
        "known_null_rejected": noncover(available) if task.get("scenario") == "null" else None,
        "mean_twice_radius": float(2 * np.mean(radii)) if radii else None,
        "abstentions": sum(row["covered"] is None for row in available),
        "declared_family": len(truth), "identified_family": len(available),
        "unidentified": ["projector_3"] if tied else [],
        "nonregular": ["eigenvalue_3"] if tied else [],
        "details": available,
    }


def _truth_metadata(data):
    eigenvalues = np.linalg.eigvalsh(data.truth)
    minimum = float(np.min(eigenvalues[..., 0]))
    p = data.truth.shape[-1]
    probes = designs.rng_for(71347, f"truth_shard_check_{p}").normal(size=(7, p))
    probes /= np.linalg.norm(probes, axis=1)[:, None]
    checks = np.concatenate([eigenvalues, np.einsum("jp,...pq,jq->...j", probes, data.truth, probes)],
                            axis=-1)
    return {
        "truth_hash": hashlib.sha256(np.ascontiguousarray(data.truth).tobytes()).hexdigest(),
        "truth_probes": checks.tolist(),
        "true_min_eigenvalue": minimum,
        "truth_interior": minimum > 0.,
        "true_max_condition": float(np.max(eigenvalues[..., -1] / eigenvalues[..., 0]))
        if minimum > 0. else None,
        "min_observed_rank": int(np.min(np.linalg.matrix_rank(data.observed, hermitian=True))),
        "max_observed_rank": int(np.max(np.linalg.matrix_rank(data.observed, hermitian=True))),
    }


def _task_data(task, core, stress):
    if task["kind"] == "dr":
        return designs.misspecification(task), None
    if task["kind"] == "measurement":
        return (designs.finite_measurement(task, stress, core),
                designs.finite_measurement(task, stress, core, validation=True))
    if task["kind"] == "coverage":
        return (designs.coverage(task), designs.coverage(task, validation=True,
                n_validation=core["validation_n"]))
    data = designs.crossover(task, core)
    validation = designs.crossover(task, core, validation=True) if task["kind"] == "core" else None
    return data, validation


def _study_methods(task, core, cohort, data, validation):
    config = ForestConfig(seed=14000 + task["seed"], **core["forest"])
    inputs = _prediction_inputs(data, cohort)
    root = _model("BW", config)
    root_error = None
    start_root = time.perf_counter()
    try:
        root.fit(cohort)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        root_error = error
    root_fit = time.perf_counter() - start_root
    result = {}
    tuned = {}
    validation_probability = _validation_propensity(validation, cohort)
    root_inference = None
    if "Oracle" in methods_for(task):
        result["Oracle"] = _oracle(data)
    need_inference = task["kind"] == "coverage" or task.get("scenario") == "null"
    for name in methods_for(task):
        if name == "Oracle":
            continue
        start = time.perf_counter()
        try:
            if name in {"BW", "LayerA", "BW_OneStep", "BW_Unaugmented", "BW_IPW", "BW_Entropy",
                        "BW_Pointwise"} and root_error is not None:
                raise root_error
            if name == "BW":
                root_inference = root.predict(**inputs, with_influence=need_inference)
                metrics = _metrics(root_inference, data, store_signed=task["kind"] == "dr")
                if need_inference:
                    metrics["coverage"] = _coverage(root_inference, data, task, core)
                result[name] = _record(metrics, time.perf_counter() - start + root_fit,
                                       fit_seconds=root_fit)
                continue
            if name == "BW_Pointwise":
                if root_inference is None:
                    raise ValueError("Primary fit failed before pointwise comparison")
                metrics = _metrics(root_inference, data)
                metrics["coverage"] = _coverage(root_inference, data, task, core, joint=False)
            elif name == "LayerA":
                metrics = _metrics(root.predict_layer_a(**inputs), data,
                                   store_signed=task["kind"] == "dr")
            elif name == "LayerAIdentity":
                result[name] = dict(result["SqrtEuclidean"], exact_alias="SqrtEuclidean")
                continue
            elif name == "BW_RootIdentitySplit":
                model = BWCausalForest(config).fit(cohort, anchor=np.eye(cohort.dimension))
                metrics = _metrics(model.predict(**inputs), data)
            elif name == "BW_NoCrossFit":
                shared_rows = _prepare(task, data, core, no_crossfit=True)
                model = BWCausalForest(config).fit(shared_rows, anchor=root.coordinates.base)
                metrics = _metrics(model.predict(**inputs), data, store_signed=True)
            elif name in {"BW_OneStep", "BW_Unaugmented", "BW_IPW", "BW_Entropy"}:
                variant = {"BW_OneStep": "one_step", "BW_Unaugmented": "unaugmented",
                           "BW_IPW": "ipw", "BW_Entropy": "entropy"}[name]
                metrics = _metrics(root.predict(**inputs, variant=variant), data,
                                   store_signed=task["kind"] == "dr")
            elif name in {"LogEuclidean", "LogCholesky"}:
                constructor = LogEuclideanDRForest if name == "LogEuclidean" else LogCholeskyDRForest
                model, fixed, tuning = _tune_log(constructor, core["ridge_candidates"],
                                                cohort, config, validation, validation_probability)
                tuned[name] = fixed
                prediction = model.predict(**inputs)
                metrics = _metrics(prediction, data) | tuning
                if task["kind"] == "coverage":
                    metrics["coverage"] = _coverage(_baseline_derivatives(name, model, prediction),
                                                    data, task, core)
            elif name in FIXED_LOG:
                base_name = name.removesuffix("_default")
                fixed_model = tuned.get(base_name)
                if fixed_model is None:
                    raise ValueError("The prespecified default ridge fit failed")
                metrics = _metrics(fixed_model.predict(**inputs), data)
                metrics["selected_ridge"] = core["ridge_candidates"][0]
            else:
                model = _model(name, config)
                selected_inputs = inputs
                if name == "BW_GlobalTLearner" and data.stratified:
                    # Global local predictions are evaluated at the subgroup marker and
                    # centered nuisance values; this comparator does not target a subgroup mixture.
                    queries = np.column_stack([data.queries, data.queries[:, 1] ** 2,
                                               *(data.queries[:, 0] == g
                                                 for g in range(1, len(data.queries)))])
                    selected_inputs = {"queries": queries}
                prediction = _fit_predict(name, model, cohort, selected_inputs,
                                          anchor=root.coordinates.base if root_error is None else None)
                metrics = _metrics(prediction, data, store_signed=task["kind"] == "dr")
                if task["kind"] == "coverage":
                    metrics["coverage"] = _coverage(_baseline_derivatives(name, model, prediction),
                                                    data, task, core)
            result[name] = _record(metrics, time.perf_counter() - start)
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
            result[name] = _record(_origin_status(error), time.perf_counter() - start)
    return result


class ScalingTimeout(TimeoutError):
    """A planned scaling profile exceeded its measurement limit."""


def _scaling(task, core, scaling, data, cohort):
    def cutoff(signum, frame):
        raise ScalingTimeout("Scaling method exceeded the 30-minute wall limit")

    start = time.perf_counter()
    previous = signal.signal(signal.SIGALRM, cutoff)
    signal.setitimer(signal.ITIMER_REAL, scaling["per_method_timeout_sec"])
    name = task["method"]
    clock = {}
    try:
        config = ForestConfig(seed=14000 + task["seed"], **core["forest"])
        model = _model(name, config)
        model.fit(cohort)
        clock["fit_seconds"] = time.perf_counter() - start
        stage = time.perf_counter()
        prediction = model.predict(data.queries)
        clock["predict_seconds"] = time.perf_counter() - stage
        result = _metrics(prediction, data)
        if name in {"BW", "SqrtEuclidean", "LogEuclidean", "LogCholesky", "Elementwise"}:
            stage = time.perf_counter()
            # A single frozen kernel and the full matrix plus spectral family quantify inference cost.
            point = data.queries[:1]
            if name == "BW":
                with_influence = model.predict(point, with_influence=True)
            else:
                with_influence = _baseline_derivatives(name, model, model.predict(point))
            clock["influence_seconds"] = time.perf_counter() - stage
            stage = time.perf_counter()
            interval = simultaneous_regions(with_influence, n_multiplier=core["bootstrap_steps"])
            clock["bootstrap_seconds"] = time.perf_counter() - stage
            result["bootstrap_abstentions"] = len(interval.abstentions)
            result["bootstrap_region_count"] = len(interval.regions)
    except ScalingTimeout as error:
        result = _origin_status(error) | {"status": "timeout"}
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as error:
        result = _origin_status(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return {name: _record(result, time.perf_counter() - start) | clock}


def execute(task, core, stress, scaling):
    setup_start = time.perf_counter()
    data, validation = _task_data(task, core, stress)
    metadata = _truth_metadata(data)
    if task["kind"] == "measurement":
        _, discrepancies = designs.measurement_reference(
            task["p"], task["period"], stress["reference_draws_per_arm_stratum"]
        )
        metadata["measurement_reference_half_max_bw_gap"] = float(discrepancies.max())
    cohort = _prepare(task, data, core)
    metadata["setup_seconds"] = time.perf_counter() - setup_start
    methods = (_scaling(task, core, scaling, data, cohort) if task["kind"] == "scaling"
               else _study_methods(task, core, cohort, data, validation))
    return {"methods": methods, "metadata": metadata}
