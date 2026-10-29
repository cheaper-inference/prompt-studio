"""Several queue workers (GP_JOB_WORKERS > 1) must never run the same job twice."""
from __future__ import annotations
import os
import pathlib
from unittest.mock import patch
from cryptography.fernet import Fernet
from django.test import SimpleTestCase, TransactionTestCase

KEY = Fernet.generate_key().decode()


# Separate processes on a real database file, as in production: Django's in-memory test
# database uses table-level shared-cache locks that production never sees.
_WORKER = """
import os, sys, django, time
sys.path.insert(0, os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "optimizer_site.settings")
django.setup()
from optimizer import durable
def barrier(name):
    # File barrier: wait until every worker has reached this point.
    d, n = os.environ["BARRIER_DIR"], int(os.environ["WORKERS"])
    open(os.path.join(d, f"{name}-{os.getpid()}"), "w").close()
    deadline = time.time() + 60
    while sum(f.startswith(name + "-") for f in os.listdir(d)) < n:
        if time.time() > deadline:
            sys.exit(f"barrier {name} timed out")
        time.sleep(0.005)
mine = []
barrier("ready")                 # all workers start claiming together
row = durable.claim()            # each takes one job (jobs > workers) ...
if row is not None:
    mine.append(row.id)
barrier("first")                 # ... before anyone continues, so work is always shared
while True:
    row = durable.claim()
    if row is None:
        break
    mine.append(row.id)
print(",".join(mine))
"""


class ParallelWorkerTests(SimpleTestCase):
    def test_concurrent_worker_processes_claim_each_job_exactly_once(self):
        import subprocess, sys, tempfile
        root = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "DB_PATH": str(pathlib.Path(tmp) / "db.sqlite3"), "PROMPT_STUDIO_CREDENTIAL_KEY": KEY}
            run = lambda code: subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                                              capture_output=True, text=True, timeout=120)
            migrated = subprocess.run([sys.executable, "manage.py", "migrate", "--noinput"], cwd=root, env=env,
                                      capture_output=True, text=True, timeout=120)
            self.assertEqual(migrated.returncode, 0, migrated.stderr[-2000:])
            seeded = run(_WORKER.split("from optimizer import durable")[0] + (
                "from optimizer import durable\n"
                "from django.test.utils import override_settings\n"
                f"with override_settings(PROMPT_STUDIO_CREDENTIAL_KEY={KEY!r}):\n"
                "    print(','.join(durable.enqueue('optimize', {'_owner': f'user{i % 3}', 'mode': 'zero_shot', 'n': i}).id for i in range(24)))\n"))
            self.assertEqual(seeded.returncode, 0, seeded.stderr[-2000:])
            jobs = seeded.stdout.strip().split(",")
            barrier = pathlib.Path(tmp) / "barrier"; barrier.mkdir()
            env = {**env, "BARRIER_DIR": str(barrier), "WORKERS": "4"}
            workers = [subprocess.Popen([sys.executable, "-c", _WORKER], cwd=root, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
            outputs = [w.communicate(timeout=120) for w in workers]
            for w, (_, err) in zip(workers, outputs):
                self.assertEqual(w.returncode, 0, err[-2000:])
            claimed = [job for out, _ in outputs for job in out.strip().split(",") if job]
            self.assertEqual(sorted(claimed), sorted(jobs))       # every job claimed, none twice
            self.assertTrue(all(out.strip() for out, _ in outputs))   # every worker got work
            mode = run(_WORKER.split("from optimizer import durable")[0] +
                       "from django.db import connection\nc = connection.cursor(); c.execute('PRAGMA journal_mode'); print(c.fetchone()[0])\n")
            self.assertEqual(mode.stdout.strip().lower(), "wal")


_SERIAL_WORKER = _WORKER.split("mine = []")[0] + """
# Widen the race the serial check closes: each worker visits the queued jobs in its own random order
# (in queue order the compare-and-set on the shared first row would serialize them anyway), and pauses
# between finding no running job with the key and claiming. Only the check-and-claim transaction then
# keeps a second worker from starting another job with the same key during that pause.
from django.db.models.query import QuerySet
_order_by = QuerySet.order_by
QuerySet.order_by = lambda self, *f: _order_by(self, "?") if self.model.__name__ == "JobState" and f[:1] == ("created_at",) else _order_by(self, *f)
_busy = durable._serial_busy
def slow_busy(*a):
    out = _busy(*a)
    time.sleep(0.3)
    return out
durable._serial_busy = slow_busy
barrier("ready")                 # every worker claims at the same moment
row = durable.claim()
print(f"{row.id}:{row.kind}" if row else "")
"""


class SerialClaimTests(SimpleTestCase):
    def test_concurrent_workers_start_only_one_job_per_serial_key(self):
        """Separate processes on a real SQLite file race to claim six rejudges of one task: the serial
        check and the claim are one transaction, so exactly one starts (the rest wait for it)."""
        import subprocess, sys, tempfile
        root = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "DB_PATH": str(pathlib.Path(tmp) / "db.sqlite3"), "PROMPT_STUDIO_CREDENTIAL_KEY": KEY}
            run = lambda code: subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                                              capture_output=True, text=True, timeout=120)
            migrated = subprocess.run([sys.executable, "manage.py", "migrate", "--noinput"], cwd=root, env=env,
                                      capture_output=True, text=True, timeout=120)
            self.assertEqual(migrated.returncode, 0, migrated.stderr[-2000:])
            seeded = run(_WORKER.split("from optimizer import durable")[0] + (
                "from optimizer import durable\n"
                "from django.test.utils import override_settings\n"
                f"with override_settings(PROMPT_STUDIO_CREDENTIAL_KEY={KEY!r}):\n"
                "    for i in range(6):\n"
                "        durable.enqueue('minime_learn', {'_owner': 'u', 'task_id': 't1', 'rejudge_only': True, 'n': i})\n"))
            self.assertEqual(seeded.returncode, 0, seeded.stderr[-2000:])
            barrier = pathlib.Path(tmp) / "barrier"; barrier.mkdir()
            env = {**env, "BARRIER_DIR": str(barrier), "WORKERS": "4"}
            workers = [subprocess.Popen([sys.executable, "-c", _SERIAL_WORKER], cwd=root, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
            outputs = [w.communicate(timeout=120) for w in workers]
            for w, (_, err) in zip(workers, outputs):
                self.assertEqual(w.returncode, 0, err[-2000:])
            started = [out.strip() for out, _ in outputs if out.strip()]
            self.assertEqual(len(started), 1, started)                    # one worker got it; the others nothing
            running = run(_WORKER.split("from optimizer import durable")[0] +
                          "from optimizer.models import JobState\nprint(JobState.objects.filter(status='running').count())\n")
            self.assertEqual(running.stdout.strip(), "1")


class JobWorkerCountTests(SimpleTestCase):
    def test_worker_count_comes_from_env_and_is_bounded(self):
        import importlib.util, pathlib
        path = pathlib.Path(__file__).resolve().parents[2] / "run_service.py"
        spec = importlib.util.spec_from_file_location("run_service", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for value, expected in {None: 1, "2": 2, "0": 1, "50": 8, "two": 1}.items():
            env = {k: v for k, v in os.environ.items() if k != "GP_JOB_WORKERS"}
            if value is not None:
                env["GP_JOB_WORKERS"] = value
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(module.job_workers(), expected, value)
