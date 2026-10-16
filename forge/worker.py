"""Background job runner: launches one training subprocess at a time.

Runs as a thread inside the API server. Training lives in a separate process so
a crash or OOM never takes down the server, and cancelling is just a kill.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

from . import db
from .config import ROOT, RUNS_DIR


class Worker:
    def __init__(self, poll_s: float = 2.0):
        self.poll_s = poll_s
        self.proc: subprocess.Popen | None = None
        self.job_id: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="forge-worker")

    def start(self):
        # Jobs left "running" by a previous server process can't be resumed.
        for j in db.list_jobs():
            if j["status"] == "running":
                db.update_job(j["id"], status="failed", error="server restarted during training",
                              finished_at=time.time())
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def cancel(self, job_id: str) -> bool:
        job = db.get_job(job_id)
        if job is None or job["status"] not in ("queued", "running"):
            return False
        # Mark first so the worker doesn't record the killed process as a failure.
        db.update_job(job_id, status="cancelled", finished_at=time.time())
        if job["status"] == "running" and self.job_id == job_id and self.proc:
            os.killpg(self.proc.pid, signal.SIGTERM)
        return True

    def _loop(self):
        while not self._stop.is_set():
            job = db.next_queued_job()
            if job is None:
                time.sleep(self.poll_s)
                continue
            self._run(job)

    def _run(self, job):
        run_dir = RUNS_DIR / job["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        self.job_id = job["id"]
        with open(run_dir / "stdout.log", "w") as out:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "forge.trainer", job["id"]],
                stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                env={**os.environ, "FORGE_HOME": str(ROOT)})
            db.update_job(job["id"], status="running", pid=self.proc.pid, started_at=time.time())
            code = self.proc.wait()
        # The trainer records completed/failed itself; cover crashes it couldn't report.
        if db.get_job(job["id"])["status"] == "running":
            db.update_job(job["id"], status="failed", finished_at=time.time(),
                          error=f"trainer exited with code {code} (see stdout.log; OOM?)")
        self.proc, self.job_id = None, None
