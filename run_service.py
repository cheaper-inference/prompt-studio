"""Run web and the durable queue worker in the existing Railway service/volume."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from sentry_init import init_sentry


def job_workers() -> int:
    """Number of durable queue worker processes, from GP_JOB_WORKERS (1–8, default 1)."""
    try:
        return min(8, max(1, int(os.environ.get("GP_JOB_WORKERS", "1"))))
    except ValueError:
        return 1


def job_lanes(n) -> list:
    """Each worker's lane, from GP_JOB_LANES (e.g. "short,any"). By default, with 2+ workers the first
    one only takes short MiniMe jobs (new outputs, re-judging, scoring), so they never wait behind a
    learning run; the others take everything."""
    raw = [x.strip() for x in os.environ.get("GP_JOB_LANES", "").split(",") if x.strip()]
    if raw:
        return [(raw[i] if i < len(raw) else "any") for i in range(n)]
    return (["short"] + ["any"] * (n - 1)) if n >= 2 else ["any"]


def main():
    init_sentry()
    root = Path(__file__).resolve().parent / "webapp_django"
    os.chdir(root)
    subprocess.run([sys.executable, "manage.py", "migrate", "--noinput"], check=True)
    children = []
    stopping = False
    def stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                child.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        # Durable queue workers. claim() is an atomic compare-and-set, so several workers
        # never take the same job. GP_JOB_WORKERS=N runs N jobs at a time (default 1).
        for lane in job_lanes(job_workers()):
            children.append(subprocess.Popen([sys.executable, "manage.py", "run_jobs"],
                                             env={**os.environ, "GP_JOB_LANE": lane}))
        children.append(subprocess.Popen([sys.executable, "-m", "gunicorn", "optimizer_site.wsgi:application",
            "--bind", "0.0.0.0:" + os.environ.get("PORT", "8000"), "--workers", "1", "--threads", "8",
            "--timeout", "120", "--graceful-timeout", "25", "--access-logfile", "-", "--error-logfile", "-"]))
        while not stopping and all(child.poll() is None for child in children):
            time.sleep(0.5)
        failed = not stopping
    finally:
        stop()
        deadline = time.monotonic() + 25
        for child in children:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
