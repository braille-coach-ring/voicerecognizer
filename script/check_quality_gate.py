#!/usr/bin/env python3
"""
Strict Quality Gate Runner with Baseline Freezing for Ruff, Mypy, and Pytest.

Usage:
  python script/check_quality_gate.py          # Run full quality gate (pytest -> ruff -> mypy)
  python script/check_quality_gate.py --sync   # Freeze / sync current baseline for Ruff and Mypy
  python script/check_quality_gate.py --step pytest|ruff|mypy
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
RUFF_BASELINE_PATH = ROOT_DIR / ".ruff-baseline.json"
MYPY_BASELINE_PATH = ROOT_DIR / "mypy-baseline.txt"

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def print_status(msg: str, status: str = "INFO", color: str = COLOR_BLUE) -> None:
    print(f"{color}[{status}]{COLOR_RESET} {msg}")


def run_pytest() -> int:
    print_status("Running pytest test suite...", "PYTEST")
    cmd = [sys.executable, "-m", "pytest"]
    res = subprocess.run(cmd, cwd=ROOT_DIR)
    if res.returncode == 0:
        print_status("Pytest passed cleanly!", "SUCCESS", COLOR_GREEN)
    else:
        print_status("Pytest test failure detected!", "FAILURE", COLOR_RED)
    return res.returncode


def get_current_ruff_errors() -> list[dict]:
    cmd = [sys.executable, "-m", "ruff", "check", "--output-format", "json", "."]
    res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8")
    if not res.stdout.strip():
        return []
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError as e:
        print_status(f"Failed to parse ruff output: {e}", "ERROR", COLOR_RED)
        return []


def sync_ruff_baseline() -> None:
    print_status("Syncing Ruff baseline...", "RUFF-SYNC")
    errors = get_current_ruff_errors()
    # Normalize path separators for cross-platform compatibility
    normalized_errors = []
    for err in errors:
        rel_path = (
            Path(err["filename"]).relative_to(ROOT_DIR).as_posix()
            if Path(err["filename"]).is_absolute()
            else err["filename"].replace("\\", "/")
        )
        normalized_errors.append(
            {
                "filename": rel_path,
                "code": err.get("code"),
                "line": err.get("location", {}).get("row"),
                "column": err.get("location", {}).get("column"),
                "message": err.get("message"),
            }
        )

    with open(RUFF_BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized_errors, f, indent=2, ensure_ascii=False)
    print_status(
        f"Saved {len(normalized_errors)} frozen Ruff errors to {RUFF_BASELINE_PATH.name}",
        "SUCCESS",
        COLOR_GREEN,
    )


def check_ruff_baseline() -> int:
    print_status("Checking Ruff against baseline...", "RUFF")
    if not RUFF_BASELINE_PATH.exists():
        print_status(
            f"Baseline file {RUFF_BASELINE_PATH.name} not found. Run with --sync to create baseline.",
            "WARNING",
            COLOR_YELLOW,
        )
        sync_ruff_baseline()
        return 0

    with open(RUFF_BASELINE_PATH, encoding="utf-8") as f:
        baseline_errors = json.load(f)

    # Build lookup set of baseline error keys
    baseline_set = set()
    for err in baseline_errors:
        key = (err["filename"], err["code"], err["line"], err["column"])
        baseline_set.add(key)

    current_errors = get_current_ruff_errors()
    new_errors = []

    for err in current_errors:
        rel_path = (
            Path(err["filename"]).relative_to(ROOT_DIR).as_posix()
            if Path(err["filename"]).is_absolute()
            else err["filename"].replace("\\", "/")
        )
        key = (
            rel_path,
            err.get("code"),
            err.get("location", {}).get("row"),
            err.get("location", {}).get("column"),
        )
        if key not in baseline_set:
            new_errors.append(
                (rel_path, err.get("code"), err.get("location", {}).get("row"), err.get("message"))
            )

    if not new_errors:
        print_status(
            f"Ruff baseline check passed! Total frozen baseline errors: {len(baseline_errors)}, New errors: 0",
            "SUCCESS",
            COLOR_GREEN,
        )
        return 0
    else:
        print_status(
            f"Ruff baseline check FAILED! Found {len(new_errors)} NEW lint errors:",
            "FAILURE",
            COLOR_RED,
        )
        for path, code, line, msg in new_errors:
            print(f"  {path}:{line} [{code}] {msg}")
        return 1


def sync_mypy_baseline() -> None:
    print_status("Syncing Mypy baseline...", "MYPY-SYNC")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    mypy_cmd = [sys.executable, "-m", "mypy", "."]
    mypy_res = subprocess.run(
        mypy_cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8", env=env
    )
    normalized_stdout = mypy_res.stdout.replace("\\", "/")

    sync_cmd = [sys.executable, "-m", "mypy_baseline", "sync"]
    sync_res = subprocess.run(
        sync_cmd,
        cwd=ROOT_DIR,
        input=normalized_stdout,
        text=True,
        capture_output=True,
        encoding="utf-8",
        env=env,
    )

    if sync_res.returncode == 0:
        print_status(f"Mypy baseline saved to {MYPY_BASELINE_PATH.name}", "SUCCESS", COLOR_GREEN)
    else:
        print_status(f"Failed to sync Mypy baseline: {sync_res.stderr}", "FAILURE", COLOR_RED)


def check_mypy_baseline() -> int:
    print_status("Checking Mypy against baseline...", "MYPY")
    if not MYPY_BASELINE_PATH.exists():
        print_status(
            f"Baseline file {MYPY_BASELINE_PATH.name} not found. Run with --sync to create baseline.",
            "WARNING",
            COLOR_YELLOW,
        )
        sync_mypy_baseline()
        return 0

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    mypy_cmd = [sys.executable, "-m", "mypy", "."]
    mypy_res = subprocess.run(
        mypy_cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8", env=env
    )
    normalized_stdout = mypy_res.stdout.replace("\\", "/")

    filter_cmd = [sys.executable, "-m", "mypy_baseline", "filter"]
    filter_res = subprocess.run(
        filter_cmd,
        cwd=ROOT_DIR,
        input=normalized_stdout,
        text=True,
        capture_output=True,
        encoding="utf-8",
        env=env,
    )

    if filter_res.stdout.strip():
        print(filter_res.stdout.strip())
    if filter_res.stderr.strip():
        print(filter_res.stderr.strip(), file=sys.stderr)

    if filter_res.returncode == 0:
        print_status(
            "Mypy baseline check passed! No new type errors introduced.", "SUCCESS", COLOR_GREEN
        )
        return 0
    else:
        print_status("Mypy baseline check FAILED! New type errors detected.", "FAILURE", COLOR_RED)
        return filter_res.returncode


def get_current_mypy_errors() -> list[dict[str, Any]]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    mypy_cmd = [sys.executable, "-m", "mypy", "."]
    res = subprocess.run(
        mypy_cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8", env=env
    )
    lines = res.stdout.replace("\\", "/").splitlines()
    errors = []
    for line in lines:
        line_str = line.strip()
        if ": error:" in line_str:
            parts = line_str.split(": error:", 1)
            file_loc = parts[0].strip()
            msg = parts[1].strip()
            code = "mypy"
            if "[" in msg and msg.endswith("]"):
                msg_body, code_part = msg.rsplit("[", 1)
                msg = msg_body.strip()
                code = code_part[:-1].strip()
            file_path = file_loc.split(":")[0] if ":" in file_loc else file_loc
            line_num = file_loc.split(":")[1] if ":" in file_loc else "1"
            errors.append(
                {
                    "filename": file_path,
                    "line": line_num,
                    "code": code,
                    "message": msg,
                    "tool": "mypy",
                }
            )
    return errors


def report_cli() -> None:
    print_status("Generating Incremental Refactoring & Baseline Error Report...", "STATS", COLOR_BLUE)
    ruff_errs = get_current_ruff_errors()
    mypy_errs = get_current_mypy_errors()

    print(f"\n{COLOR_BLUE}=== ACTIVE BASELINE ERROR SUMMARY ==={COLOR_RESET}")
    print(f"  Ruff Lint Errors:  {len(ruff_errs)}")
    print(f"  Mypy Type Errors:  {len(mypy_errs)}")
    print(f"  Total Baseline:    {len(ruff_errs) + len(mypy_errs)} remaining issues\n")

    file_counts: dict[str, int] = {}
    for err in ruff_errs:
        fn = (
            Path(err["filename"]).relative_to(ROOT_DIR).as_posix()
            if Path(err["filename"]).is_absolute()
            else err["filename"].replace("\\", "/")
        )
        file_counts[fn] = file_counts.get(fn, 0) + 1
    for err in mypy_errs:
        fn = err["filename"]
        file_counts[fn] = file_counts.get(fn, 0) + 1

    print(f"{COLOR_YELLOW}Top Files with Remaining Baseline Errors:{COLOR_RESET}")
    sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
    for fn, count in sorted_files[:15]:
        print(f"  {count:>4} errors  ->  {fn}")

    print(
        f"\n{COLOR_GREEN}Tip: Run `uv run errors` to list line numbers for step-by-step refactoring!{COLOR_RESET}\n"
    )


def errors_cli() -> None:
    parser = argparse.ArgumentParser(description="List active baseline errors line-by-line")
    parser.add_argument("--file", help="Filter errors by specific filename substring")
    args = parser.parse_args()

    print_status("Listing Active Baseline Errors for Incremental Refactoring...", "ERRORS", COLOR_BLUE)
    ruff_errs = get_current_ruff_errors()
    mypy_errs = get_current_mypy_errors()

    print(f"\n{COLOR_YELLOW}--- Ruff Lint Errors ({len(ruff_errs)}) ---{COLOR_RESET}")
    for err in ruff_errs:
        fn = (
            Path(err["filename"]).relative_to(ROOT_DIR).as_posix()
            if Path(err["filename"]).is_absolute()
            else err["filename"].replace("\\", "/")
        )
        if args.file and args.file not in fn:
            continue
        line = err.get("location", {}).get("row", 1)
        code = err.get("code", "")
        msg = err.get("message", "")
        print(f"  {fn}:{line} [{code}] {msg}")

    print(f"\n{COLOR_YELLOW}--- Mypy Type Errors ({len(mypy_errs)}) ---{COLOR_RESET}")
    for err in mypy_errs:
        fn = err["filename"]
        if args.file and args.file not in fn:
            continue
        print(f"  {fn}:{err['line']} [{err['code']}] {err['message']}")

    print()


def fmt_cli() -> None:
    print_status("Running Ruff Formatter...", "FMT")
    res = subprocess.run([sys.executable, "-m", "ruff", "format", "."], cwd=ROOT_DIR)
    sys.exit(res.returncode)


def lint_cli() -> None:
    sys.exit(check_ruff_baseline())


def typecheck_cli() -> None:
    sys.exit(check_mypy_baseline())


def test_cli() -> None:
    sys.exit(run_pytest())


def sync_baseline_cli() -> None:
    print_status("Synchronizing quality gate baselines...", "SYNC")
    sync_ruff_baseline()
    sync_mypy_baseline()
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Quality Gate Runner with Baseline Freezing"
    )
    parser.add_argument(
        "--sync", action="store_true", help="Sync / update baseline files for Ruff and Mypy"
    )
    parser.add_argument("--step", choices=["pytest", "ruff", "mypy"], help="Run specific step only")
    args = parser.parse_args()

    if args.sync:
        sync_baseline_cli()

    exit_codes = []

    if args.step:
        if args.step == "pytest":
            exit_codes.append(run_pytest())
        elif args.step == "ruff":
            exit_codes.append(check_ruff_baseline())
        elif args.step == "mypy":
            exit_codes.append(check_mypy_baseline())
    else:
        print_status("Starting complete Quality Gate check...", "QUALITY-GATE")
        exit_codes.append(run_pytest())
        exit_codes.append(check_ruff_baseline())
        exit_codes.append(check_mypy_baseline())

    final_code = max(exit_codes) if exit_codes else 0
    if final_code == 0:
        print_status("ALL QUALITY GATE CHECKS PASSED SUCCESSFULLY!", "PASSED", COLOR_GREEN)
    else:
        print_status(
            "QUALITY GATE CHECKS FAILED! Please fix the errors listed above.", "FAILED", COLOR_RED
        )

    sys.exit(final_code)


if __name__ == "__main__":
    main()
