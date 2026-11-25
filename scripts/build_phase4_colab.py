#!/usr/bin/env python3
"""Build the uploadable source snapshot and self-contained Phase 4 notebooks."""

import argparse
import base64
import io
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import nbformat as nbf

from bw_causal.phase4.protocol import atomic_json, make_manifest


NOTEBOOK_CONFIG = '''# @title Phase 4 shard settings
SHARD_NAME = {shard!r}
WORKSPACE = ""          # Leave blank to use a shard-specific directory.
GITHUB_REPO = ""        # Optional HTTPS repository URL.
GITHUB_REF = ""         # Required immutable Git revision when supplying a GitHub URL.
SOURCE_ARCHIVE = ""     # Optional alternative to the embedded source ZIP.
RESUME_ZIP = ""         # Path to a partial ZIP from this exact shard.
WALL_LIMIT_SECONDS = 30600
SKIP_SETUP_INSTALL = False
RUN_FIXTURE = False     # Testing only. Fixture results are rejected by the full collector.
'''

NOTEBOOK_RUN = '''# @title Run the shard and save one results ZIP
import base64
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
workspace = Path(WORKSPACE or f"phase4_{SHARD_NAME}").resolve()
result_root = workspace / "output"
log_root = result_root / "logs"
log_root.mkdir(parents=True, exist_ok=True)
source_root = workspace / "source"
runtime_log = log_root / f"run_{run_tag}.log"
output_file = str(workspace / f"bw_phase4_{SHARD_NAME}_{run_tag}.zip")


def unpack(archive, destination, *, resume=False):
    with ZipFile(archive) as bundle:
        if len(set(bundle.namelist())) != len(bundle.namelist()):
            raise ValueError("Archive contains duplicate path names")
        for member in bundle.infolist():
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            if member.is_dir():
                continue
            if resume and not (member.filename == "run_manifest.json"
                               or member.filename.startswith(("checkpoints/", "logs/"))):
                continue
            target = destination.joinpath(*relative.parts)
            payload = bundle.read(member)
            allow_overwrite = resume and (member.filename == "run_manifest.json"
                                          or member.filename.startswith("logs/"))
            if target.exists() and not allow_overwrite and target.read_bytes() != payload:
                raise ValueError(f"Conflicting file in ZIP or existing workspace: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)


def checked_command(args, *, root=None, env=None):
    with runtime_log.open("a", encoding="utf-8") as log:
        with subprocess.Popen(args, cwd=root, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, args)


def setup():
    if GITHUB_REPO and SOURCE_ARCHIVE:
        raise ValueError("Choose one optional external source")
    if GITHUB_REPO:
        if not GITHUB_REPO.startswith("https://") or not GITHUB_REF:
            raise ValueError("Set an HTTPS repository URL and a Git commit revision")
        if not (source_root / ".git").is_dir():
            checked_command(["git", "clone", "--", GITHUB_REPO, str(source_root)])
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", GITHUB_REF + "^{commit}"],
            cwd=source_root, text=True,
        ).strip()
        if GITHUB_REF.lower() != revision:
            raise ValueError("GITHUB_REF must be the full immutable commit SHA")
        checked_command(["git", "checkout", "--detach", revision], root=source_root)
    else:
        archive = SOURCE_ARCHIVE or (workspace / "embedded_source.zip")
        if not SOURCE_ARCHIVE:
            snapshot = base64.b64decode(SOURCE_B64, validate=True)
            if hashlib.sha256(snapshot).hexdigest() != SOURCE_SHA256:
                raise ValueError("Embedded ZIP integrity check failed")
            Path(archive).write_bytes(snapshot)
        unpack(archive, source_root)
    if RESUME_ZIP:
        unpack(RESUME_ZIP, result_root, resume=True)
    executable = sys.executable
    if not SKIP_SETUP_INSTALL:
        need_install = False
        try:
            import numpy as _np, scipy as _sp
            np_ver = tuple(int("".join(c for c in p if c.isdigit()) or 0) for p in _np.__version__.split(".")[:2])
            sp_ver = tuple(int("".join(c for c in p if c.isdigit()) or 0) for p in _sp.__version__.split(".")[:2])
            need_install = not ((2, 0) <= np_ver < (3, 0) and (1, 13) <= sp_ver < (2, 0))
        except Exception:
            need_install = True
        if need_install:
            checked_command([executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                             "-r", str(source_root / "requirements-colab.txt")])
    return executable


def package():
    with ZipFile(output_file, "w", ZIP_DEFLATED, strict_timestamps=False) as bundle:
        for path in sorted(result_root.rglob("*")):
            if path.is_file() and (path.parent != result_root / "checkpoints"
                                   or path.name.endswith(".json")):
                bundle.write(path, path.relative_to(result_root))


failure = None
try:
    python = setup()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = "1"
    command = [python, str(source_root / "scripts/run_phase4_shard.py"), "--shard", SHARD_NAME,
               "--root", str(source_root), "--output", str(result_root),
               "--wall-limit-seconds", str(WALL_LIMIT_SECONDS)]
    command += ["--fixture"] if RUN_FIXTURE else [
        "--manifest", str(source_root / "phase4_manifest.json")
    ]
    checked_command(command, root=source_root, env=environment)
except BaseException as error:
    failure = error
    (log_root / f"error_{run_tag}.txt").write_text(traceback.format_exc())

archive_ok = False
try:
    package()
    archive_ok = True
except Exception:
    if failure is None:
        raise
    print("ZIP packaging also failed:")
    traceback.print_exc()

if archive_ok:
    print("Saved archive:", output_file, flush=True)
    try:
        from google.colab import files
        files.download(output_file)
        print("Downloaded:", output_file)
    except Exception as e:
        print("(Not on Colab / download skipped):", e)
if failure is not None:
    raise failure.with_traceback(failure.__traceback__)
'''


def make_source(root, destination, manifest):
    methods = [path for path in manifest["source_files"] if path != "scripts/build_phase4_colab.py"]
    files = [*methods, "scripts/build_phase4_colab.py", "README_PHASE4.md",
             "docs/phase4_protocol.md", "docs/gate3_simulation_memo.md",
             "docs/phase2_enabling_spec.md", "docs/phase3_prototype_memo.md",
             "tests/test_bw_geometry.py", "tests/test_estimators.py", "requirements-summary.txt"]
    destination.mkdir(parents=True, exist_ok=True)
    for relative in sorted(set(files)):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    shutil.copyfile(root / "README_PHASE4.md", destination / "README.md")
    atomic_json(destination / "phase4_manifest.json", manifest)
    snapshot = io.BytesIO()
    members = set(files) | {"README.md", "phase4_manifest.json"}
    with ZipFile(snapshot, "w", ZIP_DEFLATED, strict_timestamps=False) as archive:
        for name in sorted(members):
            archive.write(destination / name, name)
    return snapshot.getvalue()


def notebook(shard_name, config, encoded, checksum):
    estimate = config["estimated_worker_hours"]
    summary = (
        f"# Bures–Wasserstein causal forests · {shard_name}\n\n"
        f"Run all cells on a CPU runtime. This shard contains **{len(config['jobs'])} jobs** and "
        f"uses {config['workers']} independent single-threaded worker(s). The planning estimate "
        f"is **{estimate:.1f} hours** and varies with CPU and method convergence.\n\n"
        "The notebook already contains the frozen source and needs no GitHub URL. If a previous "
        "run returned a partial ZIP, upload it to the session, set `RESUME_ZIP`, and run this "
        "same shard again. The final cell saves and downloads one uniquely named ZIP.\n\n"
        f"Source ZIP SHA-256 `{checksum}`."
    )
    embedded = f"# @title Frozen source snapshot\nSOURCE_SHA256 = {checksum!r}\nSOURCE_B64 = {encoded!r}"
    return nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell(summary),
        nbf.v4.new_code_cell(NOTEBOOK_CONFIG.format(shard=shard_name)),
        nbf.v4.new_code_cell(embedded),
        nbf.v4.new_code_cell(NOTEBOOK_RUN),
    ], metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": f"bwc_phase4_{shard_name}.ipynb", "provenance": []},
    })


def main():
    import hashlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = make_manifest(root)
    snapshot = make_source(root, root / "colab_repo", manifest)
    encoded = base64.b64encode(snapshot).decode("ascii")
    checksum = hashlib.sha256(snapshot).hexdigest()
    colab = root / "colab"
    colab.mkdir(parents=True, exist_ok=True)
    for stale in colab.glob("bwc_phase4_*.ipynb"):
        stale.unlink()
    (colab / "bw_phase4_source.zip").write_bytes(snapshot)
    shard_count = len(manifest["shards"])
    lines = ["# Colab shards", "",
             f"Run the **{shard_count} notebooks** in this directory on CPU runtimes. Each notebook already "
             "contains the source snapshot. The optional source ZIP is also suitable for extracting "
             "to a GitHub repository. See [the run instructions](../colab_repo/README.md).", "",
             "| Notebook | Jobs | Workers | Estimated runtime |", "|---|---:|---:|---:|"]
    written = []
    for name, config in manifest["shards"].items():
        name_on_disk = f"bwc_phase4_{name}.ipynb"
        definition = notebook(name, config, encoded, checksum)
        nbf.validate(definition)
        nbf.write(definition, colab / name_on_disk)
        written.append(name_on_disk)
        lines.append(f"| [{name}]({name_on_disk}) | {len(config['jobs'])} | "
                     f"{config['workers']} | {config['estimated_worker_hours']:.1f} h |")
    (colab / "README.md").write_text("\n".join(lines) + "\n")
    with ZipFile(colab / "bw_phase4_notebooks.zip", "w", ZIP_DEFLATED, strict_timestamps=False) as zf:
        for name_on_disk in sorted(written):
            zf.write(colab / name_on_disk, name_on_disk)
        zf.write(colab / "README.md", "README.md")


if __name__ == "__main__":
    main()
