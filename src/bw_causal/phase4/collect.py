"""Validate Phase 4 archives and assemble tables, scorecards, and figures."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .protocol import (
    atomic_json, check_checkpoint, check_numeric_versions, configurations, make_manifest,
)
from .simulation import CORE_METHODS


METRICS = ("bw_pehe", "worst_query_pehe", "projector_rmse", "eigenvalue_mae", "transport_rmse",
           "bulk_ratio_median", "latent_pehe")
SUCCESS = {"ok", "oracle_reference"}


def read_archives(archives, manifest):
    records, archives_metadata = {}, []
    for path in archives:
        with ZipFile(path) as archive:
            seen = set()
            for name in archive.namelist():
                if name in seen:
                    raise ValueError(f"Duplicate member {name} in {path}")
                seen.add(name)
            info = json.loads(archive.read("run_manifest.json"))
            shard = info.get("shard")
            if (shard not in manifest["shards"] or info.get("protocol_hash") != manifest["protocol_hash"]
                    or info.get("fixture") != manifest["fixture"]
                    or info.get("job_ids") != manifest["shards"][shard]["jobs"]):
                raise ValueError(f"Incompatible shard/protocol in {path}")
            expected = set(info["job_ids"])
            count = 0
            for name in seen:
                if not (name.startswith("checkpoints/") and name.endswith(".json")):
                    continue
                key = name.removeprefix("checkpoints/").removesuffix(".json")
                if key not in expected:
                    raise ValueError(f"Unassigned checkpoint {name} in {path}")
                record = check_checkpoint(json.loads(archive.read(name)), manifest, key)
                if key in records and record != records[key]:
                    raise ValueError(f"Conflicting duplicate checkpoint {key}")
                records[key] = record
                count += 1
            archives_metadata.append({"path": str(path), "shard": shard,
                                      "checkpoint_count": count, "expected": len(expected),
                                      "numpy": info.get("numpy"), "scipy": info.get("scipy"),
                                      "python": info.get("python"), "platform": info.get("platform")})
    return records, archives_metadata


def _cell(task):
    return {key: value for key, value in task.items() if key not in {"seed", "method"}}


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, mean=None, mcse=None, lo95=None, hi95=None,
                    median=None, q25=None, q75=None, worst=None)
    mean = float(values.mean())
    error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return dict(n=len(values), mean=mean, mcse=error,
                lo95=None if error is None else mean - 1.96 * error,
                hi95=None if error is None else mean + 1.96 * error,
                median=float(np.median(values)), q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)), worst=float(values.max()))


def _wilson(success, n):
    if not n:
        return None, None, None
    prop = success / n
    z = 1.96
    center = (prop + z**2 / (2 * n)) / (1 + z**2 / n)
    distance = z * math.sqrt(prop * (1 - prop) / n + z**2 / (4 * n * n)) / (1 + z**2 / n)
    return prop, center - distance, center + distance


def _same_truth(records):
    truths = {}
    for record in records:
        metadata = record["metadata"]
        if "truth_hash" not in metadata:
            continue
        task = record["task"]
        key = json.dumps({k: v for k, v in _cell(task).items() if k != "nuisance"}, sort_keys=True)
        value = np.asarray(metadata["truth_probes"])
        if key in truths and (truths[key].shape != value.shape
                              or not np.allclose(truths[key], value, rtol=1e-9, atol=1e-8)):
            raise ValueError(f"Ground-truth fingerprint mismatch across replications at {key}")
        truths[key] = value
    return len(truths)


def _unroll(records):
    scalar, nested, by_cell = [], [], defaultdict(list)
    for record in records:
        task, metadata = record["task"], record["metadata"]
        for method, values in record["methods"].items():
            item = {"job_id": record["job_id"], **task, "method": method,
                    **{key: val for key, val in metadata.items()
                       if key != "exception" and (val is None or isinstance(val, (float, int, bool, str)))}}
            item.update({key: val for key, val in values.items()
                         if val is None or isinstance(val, (float, int, bool, str))})
            if "exception" in metadata:
                item["task_exception"] = metadata["exception"]
            coverage = values.get("coverage", {})
            for key, val in coverage.items():
                if val is None or isinstance(val, (float, int, bool, str)):
                    item[f"coverage_{key}"] = val
            scalar.append(item)
            nested.extend({"job_id": item["job_id"], **task, "method": method, **detail}
                          for detail in coverage.get("details", []))
            by_cell[(json.dumps(_cell(task), sort_keys=True), method)].append((record, values))
    return scalar, nested, by_cell


def _cell_summary(by_cell):
    metrics_rows, coverage_rows, dr_rows, scaling_rows = [], [], [], []
    for (key, method), values in sorted(by_cell.items()):
        info = json.loads(key) | {"method": method}
        attempted = len(values)
        success = sum(item["status"] in SUCCESS for _, item in values)
        summary = {"replicates_attempted": attempted, "replicates_completed": success,
                   "replicates_with_failure": attempted - success,
                   "attempted_arm_fits": sum(item.get("attempted_arms", 0) for _, item in values),
                   "failed_arm_fits": sum(item.get("failed_arms", 0) for _, item in values),
                   "psd_violations": sum(item.get("psd_violations", 0) for _, item in values)}
        for metric in METRICS:
            observations = [item[metric] for _, item in values if item.get(metric) is not None]
            metrics_rows.append(info | summary | {"metric": metric} | _stats(observations))
        kind = info["kind"]
        if kind == "coverage" or info.get("scenario") == "null":
            coverages = [item.get("coverage", {}) for _, item in values]
            attrs = ("joint_covered", "spectral_joint_covered", "matrix_joint_covered",
                     "identifiable_joint_covered")
            for attr in attrs:
                if info.get("gap") == 0 and attr in {"joint_covered", "spectral_joint_covered"}:
                    n = 0
                else:
                    n = attempted
                count = sum(item.get(attr) is True for item in coverages)
                rate, lo, hi = _wilson(count, n)
                intervals = [item["mean_twice_radius"] for item in coverages
                             if item.get("mean_twice_radius") is not None]
                reported = sum(bool(item) for item in coverages)
                declared = next((item.get("identified_family") for item in coverages if item), 25)
                abstentions = sum(item.get("abstentions", 0) for item in coverages)
                coverage_rows.append(info | summary | {
                    "family": attr, "joint_successes": count, "coverage_denominator": n,
                    "empirical_coverage": rate, "wilson_lo95": lo, "wilson_hi95": hi,
                    "replicates_missing_band_job": attempted - reported,
                    "abstentions": abstentions + (attempted - reported) * declared,
                    "known_null_rejections": sum(item.get("known_null_rejected") is True
                                                 for item in coverages),
                    "mean_twice_radius": float(np.mean(intervals)) if intervals else None,
                })
        if kind == "dr":
            errors = np.asarray([item["signed_contrast_errors"] for _, item in values
                                 if "signed_contrast_errors" in item])
            bias, rmse = None, None
            if len(errors):
                bias = float(np.sqrt(np.mean(np.sum(errors.mean(axis=0)**2, axis=(-2, -1)))))
                rmse = float(np.sqrt(np.mean(np.sum(errors**2, axis=(-2, -1)))))
            dr_rows.append(info | summary | {"bias_frobenius": bias, "contrast_rmse": rmse})
        if kind == "scaling":
            breakdown = {}
            for time_key in ("wall_seconds", "fit_seconds", "predict_seconds", "influence_seconds",
                             "bootstrap_seconds", "peak_rss_kib"):
                times = [item[time_key] for _, item in values if item.get(time_key) is not None]
                breakdown.update({f"{time_key}_{stat}": value for stat, value in _stats(times).items()})
            timeout = sum(item["status"] == "timeout" for _, item in values)
            scaling_rows.append(info | summary | breakdown | {"timeouts": timeout})
    return metrics_rows, coverage_rows, dr_rows, scaling_rows


def _paired(records, method, region, gates):
    by_seed = defaultdict(list)
    for record in records:
        by_seed[record["task"]["seed"]].append(record)
    cell_counts = {len(block) for block in by_seed.values()}
    failed = 0
    means = []
    for _, cases in sorted(by_seed.items()):
        bw = [case["methods"]["BW"].get("bw_pehe") for case in cases]
        other = [case["methods"][method].get("bw_pehe") for case in cases]
        if any(item is None for item in bw + other) or len(cell_counts) != 1:
            failed += 1
        else:
            means.append((np.mean(bw), np.mean(other)))
    array = np.asarray(means).reshape(-1, 2)
    left, right = map(_stats, array.T)
    diff = _stats(array[:, 1] - array[:, 0])
    gain = 1 - left["mean"] / right["mean"] if right["mean"] else None
    nonoverlap = bool(left["hi95"] is not None and left["hi95"] < right["lo95"])
    threshold = gates["sqrt_gain"] if method == "SqrtEuclidean" else gates["log_and_elementwise_gain"]
    available = failed == 0 and len(means) == 50 and len(cell_counts) == 1
    return {"region": region, "comparator": method, "cells": max(cell_counts, default=0),
            "seed_block_count": len(by_seed), "paired_seed_blocks": len(means), "failed_seed_blocks": failed,
            "gain": gain, "paired_difference": diff["mean"], "paired_ci_low": diff["lo95"],
            "paired_ci_high": diff["hi95"], "bw_mean": left["mean"], "bw_mcse": left["mcse"],
            "bw_ci_low": left["lo95"], "bw_ci_high": left["hi95"], "comparator_mean": right["mean"],
            "comparator_ci_low": right["lo95"], "comparator_ci_high": right["hi95"], "gain_threshold": threshold,
            "nonoverlap": nonoverlap, "complete_confirmatory_pairing": available,
            "practical_screen_pass": available and gain is not None and gain >= threshold and nonoverlap,
            "bw_mcse_screen_pass": bool(available and left["mean"] and left["mcse"] /
                                         left["mean"] < gates["relative_mcse"])}


def _gate_pairing(records, gates):
    allowed = [rec for rec in records if rec["task"].get("scenario") in {"crossover", "spiked"}
               and rec["task"]["kind"] == "core"]
    def registry(row):
        return ((row["condition"] >= gates["register_min_condition"]
                 and row["angle"] >= gates["min_rotation_deg"])
                or row["floor"] <= gates["register_boundary_floor"])
    def mechanism(row):
        return (row["condition"] >= gates["mechanism_min_condition"]
                and row["angle"] >= gates["min_rotation_deg"])
    filters = {
        "G3_register": registry,
        "G3_register_realistic": lambda row: registry(row) and row["condition"] <= gates["max_realistic_condition"],
        "section8_mechanism": mechanism,
        "section8_mechanism_realistic": lambda row: mechanism(row) and row["condition"] <= gates["max_realistic_condition"],
        "pivot_boundary": lambda row: row["floor"] <= gates["pivot_boundary_floor"],
        "zero_observed_rank": lambda row: row["floor"] == 0,
    }
    rows = []
    for region, accept in filters.items():
        subset = [rec for rec in allowed if accept(rec["task"])]
        for method in ("SqrtEuclidean", "LogEuclidean", "Elementwise"):
            rows.append(_paired(subset, method, region, gates))
    for cell in sorted({json.dumps(_cell(item["task"]), sort_keys=True) for item in allowed}):
        subset = [rec for rec in allowed if json.dumps(_cell(rec["task"]), sort_keys=True) == cell]
        for method in CORE_METHODS:
            if method not in {"BW", "Oracle"}:
                rows.append(_paired(subset, method, cell, gates))
    return rows


def _rate_slopes(dr):
    by_method = defaultdict(list)
    for row in dr:
        if row["contrast_rmse"] is not None and row["contrast_rmse"] > 0:
            by_method[row["method"], row["nuisance"]].append(row)
    rows = []
    for (method, nuisance), subset in sorted(by_method.items()):
        ordered = sorted(subset, key=lambda row: row["n"])
        slope = (np.polyfit(np.log([row["n"] for row in ordered]),
                            np.log([row["contrast_rmse"] for row in ordered]), 1)[0]
                 if len(ordered) > 1 else None)
        rows.append({"method": method, "nuisance": nuisance, "n_points": len(ordered),
                     "log_rmse_vs_log_n_slope": slope})
    return rows


def _scaling_slopes(rows):
    by_method = defaultdict(list)
    for row in rows:
        if row["replicates_completed"] == row["replicates_attempted"] and row["wall_seconds_mean"]:
            by_method[row["method"]].append(row)
    records = []
    for method, subset in sorted(by_method.items()):
        design = np.log([[row["n"], row["p"]] for row in subset])
        design = np.column_stack((np.ones(len(design)), design))
        if len(design) < 3 or np.linalg.matrix_rank(design) < 3:
            continue
        intercept, n_exponent, p_exponent = np.linalg.lstsq(
            design, np.log([row["wall_seconds_mean"] for row in subset]), rcond=None,
        )[0]
        records.append({"method": method, "points": len(design), "log_intercept": intercept,
                        "n_exponent": n_exponent, "p_exponent": p_exponent})
    return records


def _write_gate_memo(output, manifest, merge, paired, coverage, scaling, method_rows):
    lines = ["# Gate G3 evidence sheet", "",
             "**Status** Scientific review pending.", "",
             f"Protocol `{manifest['protocol_hash']}` and {merge['present']} of {merge['expected']} jobs validated.",
             f"Mode: {'fixture, never usable for the gate' if manifest['fixture'] else 'confirmatory'}.",
             "", "## Practical margin screen", "",
             "| Region | Comparator | Complete seed blocks | Gain | Nonoverlap | Screen |",
             "|---|---|---:|---:|---|---|"]
    for row in paired:
        if row["region"].startswith("{"):
            continue
        percentage = "NA" if row["gain"] is None else f"{row['gain']:.1%}"
        lines.append(f"| {row['region']} | {row['comparator']} | {row['paired_seed_blocks']} "
                     f"| {percentage} | {row['nonoverlap']} | {row['practical_screen_pass']} |")
    lines += ["", "## Coverage and numerical stress", ""]
    for row in coverage:
        if row["method"] != "BW" or row["family"] != "joint_covered":
            continue
        rate = row["empirical_coverage"]
        if row["kind"] == "coverage" and (row["n"] < 500 or row["gap"] < .5):
            continue
        result = "NA" if rate is None else f"{rate:.1%}"
        lines.append(f"- {row['kind']} {row.get('scenario', '')}, n={row['n']}, "
                     f"gap={row.get('gap', 'null fixed')}: {result} ({row['joint_successes']}/"
                     f"{row['coverage_denominator']}), {row['abstentions']} target abstentions.")
    boundary = [row for row in method_rows if row["method"] == "BW" and row.get("floor") == 0
                and row.get("scenario") in {"spiked", "crossover"}]
    lines += ["", f"Boundary rank-deficient diagnostics: {sum(row['status'] != 'ok' for row in boundary)} "
              f"failed primary runs among {len(boundary)} observed-rank boundary runs.",
              "", "## Scaling and interpretation", ""]
    for row in scaling:
        if row["method"] == "BW" and (row["p"], row["n"]) in {(20, 500), (30, 1000)}:
            lines.append(f"- p={row['p']}, n={row['n']}: mean total seconds "
                         f"{row['wall_seconds_mean']}, peak KiB {row['peak_rss_kib_worst']}, "
                         f"timeouts {row['timeouts']}.")
    lines += ["", "Inspect the paired cell table, direct benchmark failures, the same-neighborhood Layer A "
              "ablation, realistic finite-T comparisons, and the DR rate table before assigning a verdict. "
              "A full-family miss or abstention enters the coverage denominator. Two or more completed "
              "replicates are required for a numerical Monte Carlo interval. User-facing conclusions must "
              "exclude fixture output.", ""]
    (output / "gate3_evidence.md").write_text("\n".join(lines))


def _plots(output, metrics, coverage, scaling):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, p in zip(axes, (5, 15)):
        for method, marker in (("BW", "o"), ("SqrtEuclidean", "s"), ("LogEuclidean", "^")):
            subset = [row for row in metrics if row.get("scenario") == "crossover"
                      and row["p"] == p and row["metric"] == "bw_pehe" and row["method"] == method
                      and row["mean"] is not None and row.get("floor") == .1]
            for angle in (0, 30, 60):
                part = sorted((row for row in subset if row["angle"] == angle), key=lambda row: row["condition"])
                if part:
                    axis.plot([row["condition"] for row in part], [row["mean"] for row in part],
                              marker=marker, label=f"{method}, {angle}°")
        axis.set(xlabel="Spike-to-reference-bulk ratio", title=f"Dimension {p}", xscale="log")
    axes[0].set_ylabel("Mean BW PEHE")
    if axes[1].lines:
        axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "crossover_pehe.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 4))
    for n in sorted({row["n"] for row in coverage}):
        part = sorted((row for row in coverage if row["kind"] == "coverage" and row["method"] == "BW"
                       and row["family"] == "joint_covered" and row["n"] == n
                       and row["empirical_coverage"] is not None), key=lambda row: row["gap"])
        if part:
            axis.plot([row["gap"] for row in part], [row["empirical_coverage"] for row in part],
                      "o-", label=f"n={n}")
    axis.axhspan(.92, .98, alpha=.15, color="gray")
    axis.set(xlabel="Third-to-fourth eigengap", ylabel="Full-family coverage", ylim=(0, 1.02))
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "spectral_coverage.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 4))
    for method in ("BW", "SqrtEuclidean", "LogEuclidean"):
        part = sorted((row for row in metrics if row["kind"] == "measurement"
                       and row["method"] == method and row["metric"] == "bw_pehe"
                       and row["mean"] is not None and row["period"] is not None),
                      key=lambda row: row["period"])
        if part:
            axis.plot([row["period"] for row in part], [row["mean"] for row in part],
                      "o-", label=method)
    axis.set(xlabel="Observations per covariance report", ylabel="Mean observed-target BW PEHE")
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "boundary_stability.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.5, 4))
    for method in ("BW", "SqrtEuclidean", "LogEuclidean"):
        part = sorted((row for row in scaling if row["method"] == method and row["n"] == 1000
                       and row["wall_seconds_median"] is not None), key=lambda row: row["p"])
        if part:
            axis.plot([row["p"] for row in part], [row["wall_seconds_median"] for row in part],
                      "o-", label=method)
    axis.set(xlabel="Matrix dimension at n=1000", ylabel="Median measured seconds", yscale="log")
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "scaling.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    manifest = make_manifest(args.root, fixture=args.fixture)
    records, sources = read_archives(args.archives, manifest)
    if not args.fixture:
        for archive in sources:
            check_numeric_versions(archive.get("numpy"), archive.get("scipy"))
    missing = sorted(set(manifest["tasks"]) - set(records))
    data = [records[key] for key in sorted(records)]
    truth_cells = _same_truth(data)
    summary = {"protocol_hash": manifest["protocol_hash"], "fixture": args.fixture,
               "present": len(data), "expected": len(manifest["tasks"]), "missing_job_ids": missing,
               "complete": not missing, "truth_cells_checked": truth_cells, "archives": sources}
    print(f"Validated {len(data)}/{len(manifest['tasks'])} jobs across {len(sources)} archives")
    if missing and not args.allow_incomplete:
        raise ValueError(f"{len(missing)} jobs missing, including {', '.join(missing[:4])}")
    output = args.output
    atomic_json(output / "merge_manifest.json", summary)
    rows, coverage_details, cells = _unroll(data)
    metrics, coverage, dr, scaling = _cell_summary(cells)
    core, _, _ = configurations(args.root, args.fixture)
    paired = _gate_pairing(data, core["gates"])
    _write_csv(output / "core_r50.csv", [row for row in rows if row["kind"] == "core"])
    _write_csv(output / "stress_r50_cov100.csv", [row for row in rows if row["kind"] not in {"core", "scaling"}])
    _write_csv(output / "scaling_profile.csv", [row for row in rows if row["kind"] == "scaling"])
    _write_csv(output / "per_cell_metrics.csv", metrics)
    _write_csv(output / "coverage_targets.csv", coverage_details)
    _write_csv(output / "coverage_by_family.csv", coverage)
    _write_csv(output / "dr_errors.csv", dr)
    _write_csv(output / "dr_rate_slopes.csv", _rate_slopes(dr))
    _write_csv(output / "scaling_by_cell.csv", scaling)
    _write_csv(output / "scaling_exponents.csv", _scaling_slopes(scaling))
    _write_csv(output / "gate3_pairing.csv", paired)
    _write_gate_memo(output, manifest, summary, paired, coverage, scaling, rows)
    if not args.no_plots:
        _plots(output, metrics, coverage, scaling)
    return 0
