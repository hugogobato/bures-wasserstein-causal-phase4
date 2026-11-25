"""Frozen configuration, explicit job IDs, sharding, and checkpoint validation."""

import hashlib
from itertools import product
import json
import os
from pathlib import Path
import tempfile

from .simulation import SCALING_METHODS, methods_for


SCHEMA_VERSION = 1
STUDY = "bw-phase4-v2"
GROUPS = {"core": ("core",), "stress": ("dr", "measurement", "coverage"), "scaling": ("scaling",)}


def canonical(item):
    return json.dumps(item, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def digest(item):
    return hashlib.sha256(canonical(item)).hexdigest()


def atomic_json(path, item):
    data = canonical(item) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def load_json(path):
    return json.loads(Path(path).read_text())


def protocol_sources(root):
    paths = [*root.glob("src/bw_causal/**/*.py"), *root.glob("configs/phase4_*.yaml"),
             *root.glob("scripts/*phase4*.py"), root / "pyproject.toml", root / "requirements-colab.txt"]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _version_tuple(version):
    parts = []
    for piece in str(version).split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


def check_numeric_versions(numpy_version, scipy_version):
    np_ver = _version_tuple(numpy_version)
    sp_ver = _version_tuple(scipy_version)
    if not ((2, 0) <= np_ver < (3, 0)):
        raise ValueError(f"Unsupported NumPy version {numpy_version}; require >=2,<3")
    if not ((1, 13) <= sp_ver < (2, 0)):
        raise ValueError(f"Unsupported SciPy version {scipy_version}; require >=1.13,<2")


def configurations(root, fixture):
    core, stress, scaling = [
        load_json(root / "configs" / f"phase4_{name}.yaml")
        for name in ("core", "stress", "scaling")
    ]
    if fixture:
        core.update(seeds=1, n=160, dimensions=[5], condition_ratios=[50],
                    rotation_deg=[30], eig_floors=[0.1, 0], validation_n=32, bootstrap_steps=29)
        core["forest"].update(n_trees=2, min_leaf_size=6, max_depth=2)
        core["regimes"] = [row for row in core["regimes"] if row["scenario"] in {"null", "structural"}]
        stress.update(seeds=1, coverage_seeds=1, sample_sizes=[160], measurement_periods=[8, None],
                      measurement_n=160, reference_draws_per_arm_stratum=48, coverage_eigengaps=[0, 0.5])
        scaling.update(seeds=1, dimensions=[5], sample_sizes=[160], additional_gate_point=None,
                       per_method_timeout_sec=60)
    return core, stress, scaling


def task_id(task):
    return f"{task['kind']}-{digest(task)[:20]}"


def jobs(core, stress, scaling):
    grid = [
        {"scenario": "crossover", "p": p, "condition": k, "angle": theta, "floor": floor}
        for p, k, theta, floor in product(core["dimensions"], core["condition_ratios"],
                                         core["rotation_deg"], core["eig_floors"])
    ]
    items = [
        dict(regime, kind="core", n=core["n"], seed=seed)
        for regime, seed in product(grid + core["regimes"], range(core["seeds"]))
    ]
    items.extend({"kind": "dr", "p": 5, "n": n, "nuisance": nuisance, "seed": seed}
                 for n, nuisance, seed in product(stress["sample_sizes"], stress["misspec_nuisances"],
                                                  range(stress["seeds"])))
    items.extend({"kind": "measurement", "p": stress["measurement_p"], "n": stress["measurement_n"],
                  "period": period, "seed": seed}
                 for period, seed in product(stress["measurement_periods"], range(stress["seeds"])))
    items.extend({"kind": "coverage", "p": stress["coverage_p"], "n": n,
                  "gap": gap, "seed": seed}
                 for n, gap, seed in product(stress["sample_sizes"], stress["coverage_eigengaps"],
                                              range(stress["coverage_seeds"])))
    scale_grid = list(product(scaling["dimensions"], scaling["sample_sizes"]))
    if scaling["additional_gate_point"] is not None:
        scale_grid.append(tuple(scaling["additional_gate_point"]))
    items.extend({"kind": "scaling", "scenario": "crossover", "p": p, "n": n, "seed": seed,
                  "condition": 50, "angle": 30, "floor": 0.1, "method": method}
                 for (p, n), seed, method in product(scale_grid, range(scaling["seeds"]), SCALING_METHODS))
    keyed = {task_id(task): task for task in items}
    if len(keyed) != len(items):
        raise ValueError("Duplicate simulation tasks")
    return keyed


def predict_task_seconds(task):
    kind, p, n = (task[k] for k in ("kind", "p", "n"))
    if kind == "dr":
        return 3.0 * n / 800
    if kind == "coverage":
        return 4.5 * n / 800
    if kind == "measurement":
        return 110.0 if task.get("period") in {8, 15} else 45.0
    if kind == "core":
        base = {5: 20., 6: 25., 15: 65.}[p]
        return base * (2.5 if task.get("floor") == 0 else 1.0)
    method = task["method"]
    relative_n = n / 800
    if method == "BW":
        return min(1800., 5 + 18 * (p / 15)**3.5 * relative_n**1.2)
    if method == "BW_RFWLFR":
        return min(1800., 10 + 30 * relative_n**1.2 * (p / 15)**3)
    if method == "BW_GlobalTLearner":
        return min(1800., 4 + 12 * relative_n * (p / 15)**4)
    return 3 + 10 * relative_n * (p / 50)**3


def make_manifest(root, *, fixture=False):
    core, stress, scaling = configurations(root, fixture)
    items = jobs(core, stress, scaling)
    sources = protocol_sources(root)
    protocol_hash = digest({"study": STUDY, "configs": [core, stress, scaling], "sources": sources})
    counts = {"core": 1, "stress": 1, "scaling": 1} if fixture else {
        "core": 12, "stress": 3, "scaling": 3,
    }
    shards = {}
    for group, kinds in GROUPS.items():
        partitions = [[] for _ in range(counts[group])]
        costs = [0.] * counts[group]
        ordered = sorted(
            (key for key, task in items.items() if task["kind"] in kinds),
            key=lambda key: (-predict_task_seconds(items[key]), key),
        )
        for key in ordered:
            part = min(range(len(costs)), key=lambda index: (costs[index], index))
            costs[part] += predict_task_seconds(items[key])
            partitions[part].append(key)
        for index, keys in enumerate(partitions):
            # Group finite-T periods to hit the deterministic reference cache per process.
            keys.sort(key=lambda key: (items[key]["kind"], str(items[key].get("period")), key))
            name = f"{group}_{index + 1:02}"
            workers = 1 if group == "scaling" else 2
            shards[name] = {"group": group, "workers": workers, "jobs": keys,
                            "estimated_worker_hours": costs[index] / (3600 * workers)}
    return {"schema_version": SCHEMA_VERSION, "study": STUDY, "fixture": fixture,
            "source_files": sources, "protocol_hash": protocol_hash,
            "tasks": items, "shards": shards}


def check_manifest(manifest, root):
    expected = make_manifest(root, fixture=manifest["fixture"])
    if manifest != expected:
        raise ValueError("Manifest does not match the frozen source and configuration")
    return manifest


def checkpoint(protocol, key, task, result):
    body = {"schema_version": SCHEMA_VERSION, "protocol_hash": protocol, "job_id": key,
            "task": task, **result}
    return dict(body, record_sha256=digest(body))


def check_checkpoint(record, manifest, key):
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    task = manifest["tasks"][key]
    if (record.get("record_sha256") != digest(body) or record.get("job_id") != key
            or record.get("schema_version") != SCHEMA_VERSION
            or record.get("protocol_hash") != manifest["protocol_hash"] or record.get("task") != task):
        raise ValueError(f"Checkpoint identity/integrity mismatch for {key}")
    if not isinstance(record.get("metadata"), dict):
        raise ValueError(f"Checkpoint missing ground-truth metadata for {key}")
    expected = set(methods_for(task))
    if set(record.get("methods", {})) != expected:
        raise ValueError(f"Method schema mismatch for {key}")
    for method, values in record["methods"].items():
        status = values.get("status")
        if status not in {"ok", "oracle_reference", "query_failure", "exception", "timeout", "task_failure"}:
            raise ValueError(f"Invalid status {status} for {key}/{method}")
        if status in {"ok", "oracle_reference"} and values.get("bw_pehe") is None:
            raise ValueError(f"Completed method has no PEHE at {key}/{method}")
    return record
