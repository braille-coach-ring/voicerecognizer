#!/usr/bin/env python3
"""
Strict Quality Gate Runner with Baseline Freezing for Ruff, Pyrefly Strict, and Pytest.

Usage:
  python script/check_quality_gate.py          # Run full quality gate (noqa -> pytest -> ruff -> pyrefly)
  python script/check_quality_gate.py --sync   # Freeze / sync current baseline for Ruff and Pyrefly
  python script/check_quality_gate.py --step pytest|ruff|pyrefly
  uv run stats                                 # Show active baseline error counts & statistics
  uv run errors [--file filename]              # List active baseline errors line-by-line
"""

import argparse
import json
import os
import subprocess
import sys
import unicodedata
from operator import itemgetter
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
RUFF_BASELINE_PATH = ROOT_DIR / ".ruff-baseline.json"
PYREFLY_BASELINE_PATH = ROOT_DIR / "pyrefly-baseline.json"

COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"


def get_display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in text)


def pad_display(text: str, target_width: int, align: str = "left") -> str:
    dw = get_display_width(text)
    pad_len = max(0, target_width - dw)
    if align == "right":
        return " " * pad_len + text
    return text + " " * pad_len


def print_status(msg: str, status: str = "INFO", color: str = COLOR_BLUE) -> None:
    print(f"{color}[{status}]{COLOR_RESET} {msg}")


def get_base_ref() -> str | None:
    gh_base = os.environ.get("GITHUB_BASE_REF")
    if gh_base:
        for ref in [f"origin/{gh_base}", gh_base]:
            res = subprocess.run(
                ["git", "rev-parse", "--verify", ref],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                return ref

    custom_ref = os.environ.get("QUALITY_GATE_BASE_BRANCH")
    if custom_ref:
        return custom_ref

    for main_branch in ["origin/main", "main", "origin/master", "master"]:
        res = subprocess.run(
            ["git", "merge-base", "HEAD", main_branch],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()

    candidates = ["origin/main", "main", "origin/master", "master"]
    for cand in candidates:
        res = subprocess.run(
            ["git", "rev-parse", "--verify", cand],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            return cand
    return None


def get_base_file_content(rel_path: str) -> str | None:
    base_ref = get_base_ref()
    if not base_ref:
        return None

    clean_path = rel_path.replace("\\", "/")
    cmd = ["git", "show", f"{base_ref}:{clean_path}"]
    res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8")
    if res.returncode == 0 and res.stdout.strip():
        return res.stdout
    return None


def check_no_noqa_prohibited() -> int:
    print_status("Checking for prohibited '# noqa' comments...", "NOQA-CHECK")
    noqa_found: list[str] = []
    for py_file in ROOT_DIR.rglob("*.py"):
        if py_file == Path(__file__).resolve():
            continue
        parts_set = set(py_file.parts)
        if (
            ".venv" in parts_set
            or ".git" in parts_set
            or "build" in parts_set
            or ".mypy_cache" in parts_set
        ):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            for idx, line in enumerate(content.splitlines(), 1):
                if "# noqa" in line or "#noqa" in line:
                    rel_path = py_file.relative_to(ROOT_DIR)
                    noqa_found.append(f"{rel_path}:{idx}: {line.strip()}")
        except Exception:
            pass

    if noqa_found:
        print_status(
            f"FAILURE: Found {len(noqa_found)} prohibited '# noqa' comments!",
            "FAILURE",
            COLOR_RED,
        )
        for item in noqa_found:
            print(f"  {item}")
        return 1
    else:
        print_status("No '# noqa' comments found. All clean!", "SUCCESS", COLOR_GREEN)
        return 0


def run_pytest() -> int:
    print_status("Running pytest test suite...", "PYTEST")
    cmd = [sys.executable, "-m", "pytest"]
    res = subprocess.run(cmd, cwd=ROOT_DIR)
    if res.returncode == 0:
        print_status("Pytest passed cleanly!", "SUCCESS", COLOR_GREEN)
        return 0
    else:
        print_status("Pytest failed with errors!", "FAILURE", COLOR_RED)
        return res.returncode


def get_current_ruff_errors() -> list[dict[str, Any]]:
    cmd = [sys.executable, "-m", "ruff", "check", ".", "--output-format", "json"]
    res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8")
    if not res.stdout.strip():
        return []
    try:
        return json.loads(res.stdout)
    except Exception:
        return []


def sync_ruff_baseline() -> None:
    print_status("Syncing Ruff baseline...", "RUFF-SYNC")
    current_errors = get_current_ruff_errors()
    simplified = []
    for err in current_errors:
        rel_path = (
            Path(err["filename"]).relative_to(ROOT_DIR).as_posix()
            if Path(err["filename"]).is_absolute()
            else err["filename"].replace("\\", "/")
        )
        simplified.append(
            {
                "filename": rel_path,
                "code": err.get("code"),
                "line": err.get("location", {}).get("row"),
                "column": err.get("location", {}).get("column"),
                "message": err.get("message"),
            }
        )
    RUFF_BASELINE_PATH.write_text(
        json.dumps(simplified, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_status(
        f"Saved {len(simplified)} frozen Ruff errors to {RUFF_BASELINE_PATH.name}",
        "SUCCESS",
        COLOR_GREEN,
    )


def check_ruff_baseline() -> int:
    print_status("Checking Ruff against baseline...", "RUFF")
    baseline_errors = []
    if RUFF_BASELINE_PATH.exists():
        try:
            baseline_errors = json.loads(RUFF_BASELINE_PATH.read_text(encoding="utf-8"))
        except Exception:
            baseline_errors = []
    else:
        print_status(
            f"Baseline file {RUFF_BASELINE_PATH.name} not found. Run with --sync to create baseline.",
            "WARNING",
            COLOR_YELLOW,
        )
        sync_ruff_baseline()
        return 0

    base_ref = get_base_ref()
    base_content = get_base_file_content(".ruff-baseline.json") if base_ref else None
    base_errors_count = len(baseline_errors)
    if base_content:
        try:
            base_json = json.loads(base_content)
            base_errors_count = len(base_json)
            print_status(
                f"Comparing against base branch ref [{base_ref}] for Ruff baseline...",
                "RUFF-BASE",
            )
        except Exception:
            pass

    baseline_set = set()
    for err in baseline_errors:
        key = (err["filename"], err["code"], err["line"], err["column"])
        baseline_set.add(key)

    current_errors = get_current_ruff_errors()
    current_set = set()
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
        current_set.add(key)
        if key not in baseline_set:
            new_errors.append(
                (rel_path, err.get("code"), err.get("location", {}).get("row"), err.get("message"))
            )

    fixed_count = max(0, base_errors_count - len(current_set))

    if not new_errors:
        fixed_msg = f", Fixed (Resolved): {fixed_count}" if fixed_count > 0 else ""
        print_status(
            f"Ruff baseline check passed! Active baseline errors: {len(current_set)} (Base: {base_errors_count}{fixed_msg}), New errors: 0",
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


def get_pyrefly_baseline_errors(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("errors", [])
    except Exception:
        return []


def sync_pyrefly_baseline() -> None:
    print_status("Syncing Pyrefly Strict baseline...", "PYREFLY-SYNC")
    cmd = ["pyrefly", "check", f"--baseline={PYREFLY_BASELINE_PATH.name}", "--update-baseline"]
    res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8")
    if res.returncode == 0 or PYREFLY_BASELINE_PATH.exists():
        errs = get_pyrefly_baseline_errors(PYREFLY_BASELINE_PATH)
        print_status(
            f"Pyrefly baseline saved to {PYREFLY_BASELINE_PATH.name} ({len(errs)} frozen errors)",
            "SUCCESS",
            COLOR_GREEN,
        )
    else:
        print_status(f"Failed to sync Pyrefly baseline: {res.stderr}", "FAILURE", COLOR_RED)


def print_pyrefly_category_table(
    base_errs: list[dict[str, Any]], current_errs: list[dict[str, Any]]
) -> None:
    base_counts: dict[str, int] = {}
    current_counts: dict[str, int] = {}

    for err in base_errs:
        name = err.get("name", "unknown")
        base_counts[name] = base_counts.get(name, 0) + 1

    for err in current_errs:
        name = err.get("name", "unknown")
        current_counts[name] = current_counts.get(name, 0) + 1

    def get_sort_key(k: str) -> int:
        return current_counts.get(k, 0) + base_counts.get(k, 0)

    all_names = sorted(
        set(base_counts.keys()) | set(current_counts.keys()),
        key=get_sort_key,
        reverse=True,
    )

    border_line = "+--------------------------------+--------+--------+--------+--------+"
    header_line = "| Pyrefly Error Category         |   Base |  Fixed | Active |    New |"

    print(f"\n{COLOR_BLUE}Pyrefly Category Breakdown Table{COLOR_RESET}")
    print(border_line)
    print(header_line)
    print(border_line)

    total_base = sum(base_counts.values())
    total_curr = sum(current_counts.values())

    for name in all_names[:15]:
        b = base_counts.get(name, 0)
        c = current_counts.get(name, 0)
        fixed = max(0, b - c)
        name_str = (name[:28] + "..") if len(name) > 30 else name
        fixed_str = f"-{fixed}" if fixed > 0 else "0"
        print(f"| {name_str:<30} | {b:>6} | {fixed_str:>6} | {c:>6} | {0:>6} |")

    print(border_line)
    total_fixed = max(0, total_base - total_curr)
    tf_str = f"-{total_fixed}" if total_fixed > 0 else "0"

    print(
        f"| Total                          | {total_base:>6} | {tf_str:>6} | {total_curr:>6} | {0:>6} |"
    )
    print(border_line + "\n")


def check_pyrefly_baseline() -> int:
    print_status("Checking Pyrefly (Strict Mode) against baseline...", "PYREFLY")
    if not PYREFLY_BASELINE_PATH.exists():
        print_status(
            f"Baseline file {PYREFLY_BASELINE_PATH.name} not found. Syncing baseline...",
            "WARNING",
            COLOR_YELLOW,
        )
        sync_pyrefly_baseline()
        return 0

    current_baseline_errs = get_pyrefly_baseline_errors(PYREFLY_BASELINE_PATH)
    current_count = len(current_baseline_errs)

    base_ref = get_base_ref()
    base_content = get_base_file_content("pyrefly-baseline.json") if base_ref else None
    base_errs = current_baseline_errs
    base_count = current_count
    if base_content:
        try:
            base_data = json.loads(base_content)
            base_errs = base_data.get("errors", [])
            base_count = len(base_errs)
            print_status(
                f"Comparing against base branch ref [{base_ref}] for Pyrefly baseline...",
                "PYREFLY-BASE",
            )
        except Exception:
            pass

    fixed_count = max(0, base_count - current_count)

    cmd = ["pyrefly", "check", f"--baseline={PYREFLY_BASELINE_PATH.name}"]
    res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8")
    if res.stdout.strip():
        print(res.stdout.strip())
    if res.stderr.strip():
        print(res.stderr.strip(), file=sys.stderr)

    if res.returncode == 0:
        print_pyrefly_category_table(base_errs, current_baseline_errs)
        fixed_msg = f", Fixed (Resolved): {fixed_count}" if fixed_count > 0 else ""
        print_status(
            f"Pyrefly Strict baseline check passed! Active baseline errors: {current_count} (Base: {base_count}{fixed_msg}), New errors: 0",
            "SUCCESS",
            COLOR_GREEN,
        )
        return 0
    else:
        print_status("Pyrefly Strict check FAILED! New type errors detected.", "FAILURE", COLOR_RED)
        return res.returncode


def report_cli() -> None:
    print_status("Generating Quality Gate & Baseline Error Report...", "STATS", COLOR_BLUE)
    ruff_errs = get_current_ruff_errors()
    pyrefly_errs = get_pyrefly_baseline_errors(PYREFLY_BASELINE_PATH)

    print(f"\n{COLOR_BLUE}=== ACTIVE BASELINE ERROR SUMMARY ==={COLOR_RESET}")
    print(f"  Ruff Lint Errors:     {len(ruff_errs):>4} remaining")
    print(f"  Pyrefly Type Errors: {len(pyrefly_errs):>4} remaining (Baseline)")
    print(f"  Total Baseline:       {len(ruff_errs) + len(pyrefly_errs):>4} remaining issues\n")

    code_counts: dict[str, int] = {}
    file_counts: dict[str, int] = {}
    for err in pyrefly_errs:
        code = err.get("name", "unknown")
        fn = err.get("path", "unknown")
        code_counts[code] = code_counts.get(code, 0) + 1
        file_counts[fn] = file_counts.get(fn, 0) + 1

    print(f"{COLOR_YELLOW}Pyrefly Errors by Category:{COLOR_RESET}")
    for code, count in sorted(code_counts.items(), key=itemgetter(1), reverse=True)[:10]:
        print(f"  {count:>4} errors  ->  [{code}]")

    print(f"\n{COLOR_YELLOW}Top Files with Remaining Baseline Errors:{COLOR_RESET}")
    for fn, count in sorted(file_counts.items(), key=itemgetter(1), reverse=True)[:15]:
        print(f"  {count:>4} errors  ->  {fn}")

    print(
        f"\n{COLOR_GREEN}Tip: Run `uv run errors` to list line numbers for step-by-step refactoring!{COLOR_RESET}\n"
    )


def errors_cli() -> None:
    parser = argparse.ArgumentParser(description="List active baseline errors line-by-line")
    parser.add_argument("--file", help="Filter errors by specific filename substring")
    args = parser.parse_args()

    print_status("Listing Active Baseline Errors...", "ERRORS", COLOR_BLUE)
    ruff_errs = get_current_ruff_errors()
    pyrefly_errs = get_pyrefly_baseline_errors(PYREFLY_BASELINE_PATH)

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

    print(f"\n{COLOR_YELLOW}--- Pyrefly Type Errors ({len(pyrefly_errs)}) ---{COLOR_RESET}")
    for err in pyrefly_errs:
        fn = err.get("path", "")
        if args.file and args.file not in fn:
            continue
        line = err.get("line", 1)
        code = err.get("name", "error")
        msg = err.get("concise_description", "")
        print(f"  {fn}:{line} [{code}] {msg}")

    print()


def fmt_cli() -> None:
    print_status("Running Ruff Formatter...", "FMT")
    res = subprocess.run([sys.executable, "-m", "ruff", "format", "."], cwd=ROOT_DIR)
    sys.exit(res.returncode)


def lint_cli() -> None:
    sys.exit(check_ruff_baseline())


def typecheck_cli() -> None:
    sys.exit(check_pyrefly_baseline())


def test_cli() -> None:
    sys.exit(run_pytest())


def sync_baseline_cli() -> None:
    print_status("Synchronizing quality gate baselines...", "SYNC")
    sync_ruff_baseline()
    sync_pyrefly_baseline()
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Quality Gate Runner with Baseline Freezing"
    )
    parser.add_argument(
        "--sync", action="store_true", help="Sync / update baseline files for Ruff and Pyrefly"
    )
    parser.add_argument(
        "--step", choices=["pytest", "ruff", "pyrefly"], help="Run specific step only"
    )
    args = parser.parse_args()

    if args.sync:
        sync_baseline_cli()

    exit_codes = []

    if args.step:
        if args.step == "pytest":
            exit_codes.append(run_pytest())
        elif args.step == "ruff":
            exit_codes.append(check_ruff_baseline())
        elif args.step == "pyrefly":
            exit_codes.append(check_pyrefly_baseline())
    else:
        print_status("Starting complete Quality Gate check...", "QUALITY-GATE")
        exit_codes.append(check_no_noqa_prohibited())
        exit_codes.append(run_pytest())
        exit_codes.append(check_ruff_baseline())
        exit_codes.append(check_pyrefly_baseline())

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
