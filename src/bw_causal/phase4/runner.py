"""Resumable batch execution with independent single-threaded workers."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy

from .protocol import (
    atomic_json, check_checkpoint, check_manifest, check_numeric_versions, checkpoint,
    configurations, load_json, make_manifest,
)
from .simulation import execute, methods_for


def _run_task(protocol_hash, key, task, configs):
    start = time.perf_counter()
    try:
        result = execute(task, *configs)
        result["metadata"]["job_seconds"] = time.perf_counter() - start
        return checkpoint(protocol_hash, key, task, result)
    except Exception as error:
        return checkpoint(protocol_hash, key, task, {
            "methods": {
                method: {"status": "task_failure", "bw_pehe": None, "error": str(error)}
                for method in methods_for(task)
            },
            "metadata": {"job_seconds": time.perf_counter() - start,
                         "exception": traceback.format_exc()},
        })


def _checkpoint_file(root, key):
    return root / "checkpoints" / f"{key}.json"


def _argument_parser(group):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--shard", required=group is None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2))
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--wall-limit-seconds", type=int, default=0,
                        help="Stop launching new jobs after this many seconds; zero disables the limit")
    return parser


def main(group=None):
    args = _argument_parser(group).parse_args()
    if args.wall_limit_seconds < 0:
        raise ValueError("The wall limit cannot be negative")
    root = args.root.resolve()
    manifest = (
        check_manifest(load_json(args.manifest), root) if args.manifest is not None
        else make_manifest(root, fixture=args.fixture)
    )
    if args.fixture and not manifest["fixture"]:
        raise ValueError("A confirmatory manifest cannot be used in fixture mode")
    if not manifest["fixture"]:
        check_numeric_versions(np.__version__, scipy.__version__)
    selected = manifest["shards"]
    if args.shard is not None:
        selected = {args.shard: selected[args.shard]}
    if group is not None:
        selected = {name: shard for name, shard in selected.items() if shard["group"] == group}
    if not selected:
        raise ValueError("The selected shard is outside this study group")
    configs = configurations(root, manifest["fixture"])
    for name, shard in selected.items():
        workers = args.workers or shard["workers"]
        if shard["group"] == "scaling" and workers != 1:
            raise ValueError("Scaling records use one worker without competing study tasks")
        output = args.output if len(selected) == 1 else args.output / name
        run_shard(manifest, name, output, configs, workers, args.wall_limit_seconds)
    return 0


def run_shard(manifest, name, output, configs, workers, wall_limit_seconds=0):
    shard = manifest["shards"][name]
    keys = shard["jobs"]
    run_details = {
        "schema_version": manifest["schema_version"], "study": manifest["study"],
        "fixture": manifest["fixture"], "protocol_hash": manifest["protocol_hash"],
        "shard": name, "job_ids": keys,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "logical_cores": os.cpu_count(),
        "workers": workers,
    }
    existing = output / "run_manifest.json"
    run_environment = {key: run_details[key] for key in ("python", "numpy", "scipy", "platform",
                                                         "logical_cores", "workers")}
    previous_environments = []
    if existing.exists():
        previous = load_json(existing)
        reference_keys = ("schema_version", "study", "fixture", "protocol_hash", "shard", "job_ids")
        if any(run_details[key] != previous.get(key) for key in reference_keys):
            raise ValueError("Existing output manifest is incompatible with this shard")
        previous_environments = previous.get("environment_history", [])
    if run_environment not in previous_environments:
        previous_environments.append(run_environment)
    run_details["environment_history"] = previous_environments
    atomic_json(existing, run_details)
    pending = []
    paths = output / "checkpoints"
    unexpected = {file.stem for file in paths.glob("*.json")} - set(keys)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint files in this output directory: {sorted(unexpected)[:5]}")
    for key in keys:
        path = _checkpoint_file(output, key)
        if path.exists():
            check_checkpoint(load_json(path), manifest, key)
        else:
            pending.append(key)
    print(f"{name}: {len(keys)} jobs, {len(keys) - len(pending)} resuming from checkpoints, "
          f"{workers} workers", flush=True)
    if not pending:
        atomic_json(output / "run_completion.json", {
            "shard": name, "protocol_hash": manifest["protocol_hash"], "total": len(keys),
            "new_checkpoints": 0, "new_jobs_with_failures": 0, "still_missing": 0,
        })
        return
    failures = 0
    completed = 0
    deadline = time.monotonic() + wall_limit_seconds if wall_limit_seconds else np.inf
    # A new process per scaling record makes the lifetime peak RSS specific to one method.
    isolate = shard["group"] == "scaling"
    if isolate and workers != 1:
        raise ValueError("Scaling records require a single worker")
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1 if isolate else None) as pool:
        attempts = {}
        todo = iter(pending)
        while True:
            while len(attempts) < workers and time.monotonic() < deadline:
                key = next(todo, None)
                if key is None:
                    break
                future = pool.submit(_run_task, manifest["protocol_hash"], key,
                                     manifest["tasks"][key], configs)
                attempts[future] = key
            if not attempts:
                break
            for future in wait(attempts, return_when=FIRST_COMPLETED)[0]:
                key = attempts.pop(future)
                record = future.result()
                check_checkpoint(record, manifest, key)
                atomic_json(_checkpoint_file(output, key), record)
                completed += 1
                methods = record["methods"]
                failed = [f"{method}:{row['status']}" for method, row in methods.items()
                          if row["status"] not in {"ok", "oracle_reference"}]
                failures += bool(failed)
                detail = f" [{', '.join(failed)}]" if failed else ""
                elapsed = record["metadata"]["job_seconds"]
                print(f"{name} {completed}/{len(pending)} {key} {elapsed:.2f}s{detail}", flush=True)
    atomic_json(output / "run_completion.json", {
        "shard": name, "protocol_hash": manifest["protocol_hash"], "total": len(keys),
        "new_checkpoints": completed, "new_jobs_with_failures": failures,
        "still_missing": len(pending) - completed,
    })
    if completed < len(pending):
        print(f"{name}: wall limit reached. Resume {len(pending) - completed} remaining jobs "
              "from this shard's ZIP.", flush=True)
