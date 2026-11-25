# Bures–Wasserstein causal forests

This repository snapshot runs the preregistered Phase 4 simulations for covariance-valued causal effects. It includes the fitted augmented-score root, matched DR embedding baselines, BW T-learners, ablations, experiment assignments, and the result collector.

## Run on Colab

1. Use the 18 prepared notebooks in the project’s `colab/` folder. Each includes a frozen copy of this repository. Open each in Colab with a CPU runtime and select **Run all**. Leave the settings at their defaults for the full experiment.
2. Allow each shard to finish and download its single ZIP. Core and stress shards use two single-threaded workers, and scaling shards use one. Planning estimates in the notebook index vary by CPU. The code stops starting new jobs after 8½ hours and bundles all completed checkpoints.
3. If a notebook returns fewer jobs than assigned, upload its ZIP to a fresh session and set that same notebook’s `RESUME_ZIP` to the uploaded path. **Run all** resumes the missing jobs. Preserve both ZIPs. Scientific failures and method timeouts are completed records and stay in the result set.
4. Place the downloaded ZIPs together and use the collector below. A setup-failure ZIP might contain only the initiating traceback and logs. Inspect those files and rerun that shard after resolving the setup error.

You can upload this directory to GitHub as the repository root. The notebooks can use that upload through `GITHUB_REPO` and its full commit SHA in `GITHUB_REF`. Keep the source/configuration identical to the packaged release. An edited configuration has a different protocol and cannot supply results for this run. You can also provide the supplied source ZIP using `SOURCE_ARCHIVE`. No dataset download or API credentials are required.

## Verify and collect the returned ZIPs

Use a POSIX environment with Python 3.12 and run from the uploaded or extracted repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-summary.txt
PYTHONPATH=src python scripts/summarize_phase4.py /path/to/downloads/bw_phase4_*.zip \
  --output results/phase4_sim
```

The collector validates shard assignment, protocol hashes, record checksums, method rows, matching truth fingerprints, compatible numerical library versions, and complete assignment of every job. It accepts byte-identical checkpoints copied from partial ZIPs into resumed ZIPs. Any conflicting repeat, wrong protocol, or missing job aborts the default collection. Use `--allow-incomplete` only to inspect progress. Keep the raw ZIPs and all results.

The output includes raw core, stress, and scaling CSVs, per-cell and paired metrics, coverage by declared family and target, rate and scaling diagnostics, PDF figures, and an evidence sheet for Gate G3 review. Review the evidence together with the [frozen protocol](docs/phase4_protocol.md) before making a scientific decision.

## Run a shard locally

Use the same dependencies and choose one `--shard` from `phase4_manifest.json`. The scripts enforce one BLAS thread per worker and limit scaling to one worker.

```bash
PYTHONPATH=src python scripts/run_phase4_shard.py \
  --manifest phase4_manifest.json --shard core_01 --workers 1 \
  --output results/local_core_01 --wall-limit-seconds 30600
```

ZIP that output’s contents from its root when mixing local and Colab results. For a pipeline check use `--fixture --shard core_01` with a new output directory and omit the confirmatory manifest. Fixture checkpoints have their own signature and cannot enter the confirmatory collector.

See [Gate G3 status](docs/gate3_simulation_memo.md) and the [Phase 2 estimator and inference scope](docs/phase2_enabling_spec.md).
