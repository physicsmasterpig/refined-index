#!/usr/bin/env python3
"""
scripts/run_assigned.py — Distributed computation coordinator.

Reads work_manifest.json to find this machine's assigned tasks,
runs them in order (skipping already-cached results), then commits
and pushes completed cache files to GitHub so the other machine
can pull them.

Usage (normally called by run.sh):
    python3 scripts/run_assigned.py [--dry-run] [--no-push]

Flags:
    --dry-run   Show what would be done, compute nothing.
    --no-push   Run computation but skip git commit/push.
    --task N    Only run task number N (0-indexed) from this machine's list.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent          # v0.4/scripts/
REPO_ROOT   = SCRIPT_DIR.parent.parent                  # ultimate/
V04_DIR     = SCRIPT_DIR.parent                         # v0.4/
CACHE_DIR   = V04_DIR / "cache"                         # v0.4/cache/
MANIFEST    = SCRIPT_DIR / "work_manifest.json"
LOG_DIR     = V04_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str, char: str = "=") -> None:
    line = char * 60
    print(f"\n{line}\n{msg}\n{line}")


def run(cmd: list[str], env: dict | None = None, check: bool = True) -> int:
    """Run a subprocess, streaming output to stdout."""
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=merged_env)
    if check and result.returncode != 0:
        print(f"[run_assigned] ERROR: command failed with code {result.returncode}")
        print(f"  {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result.returncode


def git(*args, check: bool = True) -> int:
    return run(["git", "-C", str(REPO_ROOT), *args], check=check)


def python3(script: str, *args, extra_env: dict | None = None) -> int:
    """Run a python script under v0.4/scripts/ with MANIFOLD_INDEX_CACHE_DIR set."""
    env = {
        "MANIFOLD_INDEX_CACHE_DIR": str(CACHE_DIR),
        **(extra_env or {}),
    }
    return run([sys.executable, str(SCRIPT_DIR / script), *args], env=env)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_pull() -> None:
    banner("git pull --rebase", "-")
    rc = git("pull", "--rebase", "--autostash", check=False)
    if rc != 0:
        print("[run_assigned] WARNING: pull failed (offline? conflict?). Continuing anyway.")


def git_push_cache(message: str, dry_run: bool = False) -> None:
    """Stage all new/modified cache files and push."""
    banner(f"Committing & pushing: {message}", "-")

    # Stage only the cache directory (binary blobs tracked by LFS)
    rc = git("add", str(CACHE_DIR.relative_to(REPO_ROOT)), check=False)
    if rc != 0:
        print("[run_assigned] Nothing new to stage.")
        return

    # Check if there's anything to commit
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--cached", "--quiet"],
        capture_output=True,
    )
    if status.returncode == 0:
        print("[run_assigned] Cache unchanged — nothing to commit.")
        return

    commit_msg = f"[{socket.gethostname()}] {message}  ({timestamp()})"
    if dry_run:
        print(f"[DRY RUN] would commit: {commit_msg}")
        return

    git("commit", "-m", commit_msg)

    # Retry push up to 3 times (handles simultaneous pushes from the other machine)
    for attempt in range(1, 4):
        rc = git("push", check=False)
        if rc == 0:
            print("[run_assigned] Push succeeded.")
            return
        print(f"[run_assigned] Push failed (attempt {attempt}/3). Pulling and retrying...")
        git("pull", "--rebase", "--autostash", check=False)
        time.sleep(3)
    print("[run_assigned] WARNING: push failed 3 times. Continuing — push manually later.")


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_kernels(task: dict, census: str, dry_run: bool, no_push: bool) -> None:
    qq      = task["qq"]
    q_min   = task.get("q_min", 0)
    q_max   = task.get("q_max", 5)
    workers = task.get("workers", 8)

    log_file = LOG_DIR / f"kernels_qq{qq}_Q{q_min}to{q_max}.log"
    banner(f"KERNELS  qq={qq}  Q={q_min}–{q_max}  workers={workers}")

    args = [
        "rebuild_kernels.py",
        "--qq", str(qq),
        "--q-min", str(q_min),
        "--q-max", str(q_max),
        "--workers", str(workers),
        "--census", census,
        "--no-iref",   # iref handled as separate tasks
        "--no-nc",
    ]
    if dry_run:
        args.append("--dry-run")

    # Tee output to log file
    env = {"MANIFOLD_INDEX_CACHE_DIR": str(CACHE_DIR)}
    merged = {**os.environ, **env}
    with open(log_file, "a") as lf:
        lf.write(f"\n\n{'='*60}\nRUN: {timestamp()}\n{'='*60}\n")
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / args[0]), *args[1:]],
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"[run_assigned] ERROR: kernels task failed (code {proc.returncode})")
        sys.exit(proc.returncode)

    if not no_push:
        git_push_cache(f"kernels qq={qq} Q={q_min}–{q_max}", dry_run=dry_run)


def run_iref(task: dict, census: str, dry_run: bool, no_push: bool) -> None:
    qq = task["qq"]
    log_file = LOG_DIR / f"iref_qq{qq}.log"
    banner(f"IREF CACHE  qq={qq}")

    args = [
        "build_iref_cache.py",
        "--qq", str(qq),
        "--census", census,
        "--skip-existing",
    ]
    if dry_run:
        args.append("--dry-run")

    env = {"MANIFOLD_INDEX_CACHE_DIR": str(CACHE_DIR)}
    merged = {**os.environ, **env}
    with open(log_file, "a") as lf:
        lf.write(f"\n\n{'='*60}\nRUN: {timestamp()}\n{'='*60}\n")
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / args[0]), *args[1:]],
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"[run_assigned] ERROR: iref task failed (code {proc.returncode})")
        sys.exit(proc.returncode)

    if not no_push:
        git_push_cache(f"iref cache qq={qq}", dry_run=dry_run)


def run_nc(task: dict, census: str, dry_run: bool, no_push: bool) -> None:
    qq = task.get("qq", 20)
    log_file = LOG_DIR / f"nc_qq{qq}.log"
    banner(f"NC CYCLE CACHE  qq={qq}")

    args = [
        "build_nc_cache.py",
        "--qq", str(qq),
        "--census", census,
        "--skip-existing",
    ]
    if dry_run:
        args.append("--dry-run")

    env = {"MANIFOLD_INDEX_CACHE_DIR": str(CACHE_DIR)}
    merged = {**os.environ, **env}
    with open(log_file, "a") as lf:
        lf.write(f"\n\n{'='*60}\nRUN: {timestamp()}\n{'='*60}\n")
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / args[0]), *args[1:]],
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"[run_assigned] ERROR: nc task failed (code {proc.returncode})")
        sys.exit(proc.returncode)

    if not no_push:
        git_push_cache(f"nc cycle cache qq={qq}", dry_run=dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run this machine's assigned computation tasks.")
    parser.add_argument("--dry-run",  action="store_true", help="Show plan, do not compute.")
    parser.add_argument("--no-push",  action="store_true", help="Skip git commit/push after each task.")
    parser.add_argument("--task",     type=int, default=None, metavar="N",
                        help="Run only task N (0-indexed). Default: run all.")
    args = parser.parse_args()

    # ── Load manifest ──
    manifest = json.loads(MANIFEST.read_text())
    census   = manifest.get("census", "m003-m412")
    machines = manifest.get("machines", {})
    hostname = socket.gethostname()

    if hostname not in machines:
        print(f"[run_assigned] ERROR: hostname '{hostname}' not found in work_manifest.json")
        print(f"  Known machines: {list(machines.keys())}")
        print(f"  Add an entry for this machine in v0.4/scripts/work_manifest.json")
        sys.exit(1)

    machine  = machines[hostname]
    tasks    = machine.get("tasks", [])
    name     = machine.get("name", hostname)

    banner(f"Machine : {name}  ({hostname})\nTasks   : {len(tasks)}\nCensus  : {census}\nCache   : {CACHE_DIR}")

    if not tasks:
        print("[run_assigned] No tasks assigned to this machine. Edit work_manifest.json.")
        return

    # ── Pull latest (get other machine's finished kernels) ──
    if not args.dry_run:
        git_pull()

    # ── Run tasks ──
    task_list = [tasks[args.task]] if args.task is not None else tasks

    for i, task in enumerate(task_list):
        t = task["type"]
        idx = args.task if args.task is not None else i
        print(f"\n>>> Task [{idx+1}/{len(tasks)}]: {t}  {task}")

        if t == "kernels":
            run_kernels(task, census, args.dry_run, args.no_push)
        elif t == "iref":
            run_iref(task, census, args.dry_run, args.no_push)
        elif t == "nc":
            run_nc(task, census, args.dry_run, args.no_push)
        else:
            print(f"[run_assigned] Unknown task type '{t}' — skipping.")

    banner("ALL TASKS COMPLETE")


if __name__ == "__main__":
    main()
