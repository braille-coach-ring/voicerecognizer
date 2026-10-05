"""Transfer this worktree's exact inputs to a private Colab T4 comparison session."""

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from script.compare_speaker_independent import audit_splits
from script.evaluate_speaker_independent import hash_file
from voicerecognizer.config import DEFAULT_SPEAKER_SPLIT_DIR, PROJECT_ROOT


def wsl_path(path: Path) -> str:
    path = Path(path).resolve()
    return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]


def colab(
    session: str,
    arguments: list[str],
    code: str | None = None,
    check: bool = True,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    command = ["wsl", "-u", "root", "--", "/root/.local/bin/colab", "--auth=oauth2", *arguments]
    if session:
        command += ["-s", session]
    result = subprocess.run(
        command, input=code, text=True, encoding="utf-8", capture_output=True, timeout=timeout
    )
    if "not found." in result.stdout and "[colab] Session" in result.stdout:
        result.returncode = 1
    for output in (result.stdout, result.stderr):
        safe_output = re.sub(
            r"colab-runtime-proxy-token=[^\s&)]+",
            "colab-runtime-proxy-token=[redacted]",
            output,
        )
        encoding = sys.stdout.encoding or "utf-8"
        safe_output = safe_output.encode(encoding, errors="replace").decode(encoding)
        print(safe_output, end="", flush=True)
    if check:
        result.check_returncode()
    return result


def package_inputs(initial_model: Path, staging: Path, splits: Path | None = None) -> Path:
    splits = splits or DEFAULT_SPEAKER_SPLIT_DIR
    summary = audit_splits(splits)
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "inputs.tar.gz"
    files = [p for base in ("src", "script") for p in (PROJECT_ROOT / base).rglob("*.py")]
    files += [splits / f"{name}.csv" for name in ("train", "val", "test")]
    files += [PROJECT_ROOT / path for value in summary.values() for path in value["audio"]]
    captured = {
        "splits": summary,
        "source_files": {
            p.relative_to(PROJECT_ROOT).as_posix(): hash_file(p) for p in files if p.suffix == ".py"
        },
        "source_head": subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={PROJECT_ROOT.as_posix()}",
                "-C",
                str(PROJECT_ROOT),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "model_source": f"read-only snapshot of {initial_model}; historical data provenance unverified",
        "initial_model_sha256": hash_file(initial_model / "model.safetensors"),
    }
    capture = staging / "bundle_manifest.json"
    capture.write_text(json.dumps(captured, ensure_ascii=False, indent=2), encoding="utf-8")
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for path in sorted(set(files)):
            tar.add(path, arcname=path.relative_to(PROJECT_ROOT).as_posix())
        for name in ("model.safetensors", "config.json", "labels.json", "preprocessor_config.json"):
            tar.add(initial_model / name, arcname=f"experiments/initial/{name}")
        tar.add(capture, arcname="experiments/bundle_manifest.json")
    print(f"Packed {archive.stat().st_size / 1024**2:.1f} MiB", flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--session", default="vr-si-" + time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPEAKER_SPLIT_DIR)
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=("fresh", "warm_start", "phoneme_multi"),
        default=["fresh", "warm_start", "phoneme_multi"],
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Also retrieve checkpoint tensors; JSON is always retrieved",
    )
    args = parser.parse_args()
    staging = (PROJECT_ROOT / "experiments" / args.session).resolve()
    staging.relative_to((PROJECT_ROOT / "experiments").resolve())
    if staging.exists():
        raise FileExistsError("Use a new session name; saved experiments are never overwritten")
    args.split_dir = args.split_dir.resolve()
    args.split_dir.relative_to(PROJECT_ROOT.resolve())
    archive = package_inputs(args.initial_model.resolve(), staging, args.split_dir)
    remote_root = "/content/voicerecognizer-review"
    dependencies = [
        f"{name}=={importlib.metadata.version(name)}"
        for name in ("transformers", "numpy", "scikit-learn", "soundfile", "librosa", "accelerate")
    ]
    dependencies += [
        "sounddevice",
        "matplotlib",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "tqdm",
        "python-dotenv",
    ]
    colab(args.session, ["new", "--gpu", "T4"])
    try:
        chunks: list[Path] = []
        with archive.open("rb") as stream:
            while block := stream.read(16 * 1024 * 1024):
                path = staging / f"input-{len(chunks):04d}.part"
                path.write_bytes(block)
                chunks.append(path)
        for i, path in enumerate(chunks):
            for attempt in range(3):
                result = colab(
                    args.session, ["upload", wsl_path(path), f"/content/{path.name}"], check=False
                )
                if result.returncode == 0:
                    break
                if attempt == 2:
                    result.check_returncode()
                time.sleep(2)
            print(f"Uploaded part {i + 1}/{len(chunks)}", flush=True)
        setup = f"""
import os, subprocess, sys, tarfile, pathlib, hashlib, json
project = pathlib.Path({remote_root!r})
project.mkdir()
with pathlib.Path('/content/inputs.tar.gz').open('wb') as joined:
    for part in sorted(pathlib.Path('/content').glob('input-*.part')):
        with part.open('rb') as stream:
            import shutil
            shutil.copyfileobj(stream, joined)
with tarfile.open('/content/inputs.tar.gz') as bundle:
    bundle.extractall(project, filter='data')
captured = json.loads((project / 'experiments/bundle_manifest.json').read_text())
for name, digest in captured['source_files'].items():
    assert hashlib.sha256((project / name).read_bytes()).hexdigest() == digest, name
assert hashlib.sha256((project / 'experiments/initial/model.safetensors').read_bytes()).hexdigest() == captured['initial_model_sha256']
subprocess.run(['apt-get', 'update', '-qq'], check=True)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'libportaudio2'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *{dependencies!r}], check=True)
os.chdir(project)
environment = os.environ.copy()
environment['PYTHONPATH'] = str(project / 'src') + ':' + str(project)
environment['PYTHONDONTWRITEBYTECODE'] = '1'
environment['VOICERECOGNIZER_CACHE_DIR'] = str(project / 'experiments/cache')
environment['HF_HOME'] = str(project / 'experiments/hf-cache')
environment['NUMBA_CACHE_DIR'] = str(project / 'experiments/numba-cache')
environment['MPLCONFIGDIR'] = str(project / 'experiments/matplotlib-cache')
command = [
    sys.executable, '-u', '-m', 'script.compare_speaker_independent',
    '--initial-model', str(project / 'experiments/initial'),
    '--output-dir', str(project / 'experiments/comparison'),
    '--epochs', {str(args.epochs)!r},
    '--learning-rate', {str(args.learning_rate)!r},
    '--patience', {str(args.patience)!r},
    '--split-dir', str(project / {args.split_dir.relative_to(PROJECT_ROOT).as_posix()!r}),
    '--runs', *{args.runs!r},
]
worker = (
    "import subprocess, pathlib; "
    + "log=open(" + repr(str(project / 'experiments/comparison.log')) + ", 'w'); "
    + "code=subprocess.call(" + repr(command) + ", stdout=log, stderr=subprocess.STDOUT); "
    + "pathlib.Path(" + repr(str(project / 'experiments/comparison.exit')) + ").write_text(str(code))"
)
process = subprocess.Popen([sys.executable, '-u', '-c', worker], env=environment,
                           start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('Started comparison worker:', process.pid)
"""
        colab(args.session, ["exec", "--timeout", "900"], code=setup, timeout=1000)
        deadline = time.monotonic() + 14400
        consecutive_errors = 0
        while True:
            snapshot = f"""
import pathlib, tarfile, json
project = pathlib.Path({remote_root!r})
root = project / 'experiments/comparison'
manifest = root / 'manifest.json'
# A manifest is written atomically; absent means the worker is still importing.
state = json.loads(manifest.read_text()) if manifest.exists() else {{'runs': {{}}}}
exit_file = project / 'experiments/comparison.exit'
status = {{'exit_code': int(exit_file.read_text()) if exit_file.exists() else None,
           'runs': {{name: value['status'] for name, value in state['runs'].items()}}}}
(project / 'experiments/progress.json').write_text(json.dumps(status))
log = project / 'experiments/comparison.log'
if log.exists():
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 12000))
        tail = stream.read().decode('utf-8', errors='replace')
    (project / 'experiments/progress.log').write_text(tail)
with tarfile.open('/content/progress.tar.gz', 'w:gz') as bundle:
    for path in root.rglob('*.json'):
        bundle.add(path, arcname=path.relative_to(project).as_posix())
    for name in ('progress.json', 'progress.log'):
        path = project / 'experiments' / name
        if path.exists():
            bundle.add(path, arcname=path.relative_to(project).as_posix())
print('COMPARISON_STATUS=' + json.dumps(status))
"""
            try:
                result = colab(
                    args.session, ["exec", "--timeout", "120"], code=snapshot, timeout=300, check=False
                )
                statuses = [
                    line for line in result.stdout.splitlines() if line.startswith("COMPARISON_STATUS=")
                ]
                if not statuses:
                    raise RuntimeError("Comparison status missing")
                status = json.loads(statuses[-1].split("=", 1)[1])
                colab(
                    args.session,
                    ["download", "/content/progress.tar.gz", wsl_path(staging / "progress.tar.gz")],
                    check=False,
                )
                progress_archive = staging / "progress.tar.gz"
                if progress_archive.exists():
                    try:
                        with tarfile.open(progress_archive) as bundle:
                            bundle.extractall(staging / "progress", filter="data")
                    except Exception:
                        pass
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                print(f"[Warning] Polling snapshot failed ({consecutive_errors}/5): {exc}")
                if consecutive_errors >= 5:
                    raise
                time.sleep(30)
                continue

            if status["exit_code"] is not None:
                if status["exit_code"]:
                    raise RuntimeError(
                        f"Comparison exited with status {status['exit_code']}; partial results retained"
                    )
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Comparison exceeded four hours; partial results retained")
            time.sleep(60)

    finally:
        collect = f"""
import pathlib, tarfile
project = pathlib.Path({remote_root!r})
output = pathlib.Path('/content/results.tar.gz')
with tarfile.open(output, 'w:gz', compresslevel=1) as bundle:
    for path in (project / 'experiments').rglob('*'):
        if path.is_file() and (path.suffix == '.json' or path.name == 'comparison.log'):
            if 'hf-cache' not in path.parts:
                bundle.add(path, arcname=path.relative_to(project).as_posix())
    if {args.download_models!r}:
        for path in (project / 'experiments/comparison').glob('*/checkpoint/model.safetensors'):
            bundle.add(path, arcname=path.relative_to(project).as_posix())
with output.open('rb') as stream:
    part_count = 0
    while block := stream.read(2 * 1024 * 1024):
        pathlib.Path(f'/content/result-{{part_count:04d}}.part').write_bytes(block)
        part_count += 1
pathlib.Path('/content/result-count.txt').write_text(str(part_count))
print('Results archive bytes:', output.stat().st_size, 'parts:', part_count)
"""
        try:
            colab(args.session, ["exec", "--timeout", "600"], code=collect, check=False)
            colab(
                args.session,
                ["download", "/content/result-count.txt", wsl_path(staging / "result-count.txt")],
            )
            part_count = int((staging / "result-count.txt").read_text())
            with (staging / "results.tar.gz").open("wb") as joined:
                for i in range(part_count):
                    path = staging / f"result-{i:04d}.part"
                    for attempt in range(3):
                        result = colab(
                            args.session,
                            ["download", f"/content/{path.name}", wsl_path(path)],
                            check=False,
                        )
                        if result.returncode == 0:
                            break
                        if attempt == 2:
                            result.check_returncode()
                        time.sleep(2)
                    joined.write(path.read_bytes())
            with tarfile.open(staging / "results.tar.gz") as bundle:
                bundle.extractall(staging / "results", filter="data")
        finally:
            colab(args.session, ["stop"], check=False)


if __name__ == "__main__":
    main()
