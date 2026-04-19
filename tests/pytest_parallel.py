"""
Minimal parallel pytest plugin.

Flags:
  --parallel          run tests in parallel across nproc-1 workers
  --workers N         run with exactly N workers (or 'auto' for nproc-1)
  --fork              use fork-based workers (default on Linux, opt-in on macOS)

Drop this file next to conftest.py and add to conftest.py:
    pytest_plugins = ["pytest_parallel"]
"""

import json
import os
import queue
import subprocess
import sys
import threading
from multiprocessing import cpu_count

import pytest
from _pytest.reports import TestReport
from _pytest.runner import runtestprotocol as _runtestprotocol

_WORKER_ENV = "_PYTEST_PARALLEL_WORKER"
_SENTINEL = "\x00"


def pytest_addoption(parser):
    parser.addoption(
        "--parallel",
        action="store_true",
        default=False,
        help="Run tests in parallel across nproc-1 workers",
    )
    parser.addoption(
        "--workers",
        dest="numworkers",
        metavar="N",
        default=None,
        help="Number of parallel workers (integer or 'auto' for nproc-1)",
    )
    parser.addoption(
        "--fork",
        action="store_true",
        default=False,
        help="Use fork-based workers (default on Linux, opt-in on macOS)",
    )


def pytest_configure(config):
    if os.environ.get(_WORKER_ENV):
        config.pluginmanager.register(_SubprocessWorker(), "parallel_worker")
        return

    parallel = config.getoption("parallel", default=False)
    raw = config.getoption("numworkers", default=None)
    use_fork = config.getoption("fork", default=False) or sys.platform == "linux"

    if parallel and raw is None:
        raw = "auto"
    if raw is None:
        return

    n = max(1, cpu_count() - 1) if raw == "auto" else int(raw)
    config.pluginmanager.register(_Controller(n, use_fork), "parallel_controller")


class _SubprocessWorker:
    """Subprocess worker: streams serialized TestReports to stdout."""

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = yield
        data = item.config.hook.pytest_report_to_serializable(config=item.config, report=report)
        sys.stdout.write(_SENTINEL + json.dumps(data) + "\n")
        sys.stdout.flush()
        return report


class _Controller:
    def __init__(self, n: int, use_fork: bool):
        self.n = n
        self.use_fork = use_fork

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session):
        n = min(self.n, len(session.items))
        if not n:
            return True
        if self.use_fork:
            self._run_fork(session, n)
        else:
            self._run_subprocess(session, n)
        return True

    def _run_fork(self, session, n):
        q: queue.Queue[dict | None] = queue.Queue()
        workers: list[tuple[int, list]] = []

        for i in range(n):
            bucket = session.items[i::n]
            r_fd, w_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r_fd)
                _fork_worker(bucket, w_fd)  # never returns
            os.close(w_fd)
            workers.append((pid, bucket))
            threading.Thread(target=_read_pipe, args=(os.fdopen(r_fd), q), daemon=True).start()

        hook = session.config.hook
        reported: set[str] = set()
        pending = n
        while pending:
            data = q.get()
            if data is None:
                pending -= 1
            else:
                report = hook.pytest_report_from_serializable(config=session.config, data=data)
                if report.when == "call":
                    reported.add(report.nodeid)
                hook.pytest_runtest_logreport(report=report)

        for pid, bucket in workers:
            _, wstatus = os.waitpid(pid, 0)
            exit_code = os.waitstatus_to_exitcode(wstatus)
            if exit_code != 0:
                for item in bucket:
                    if item.nodeid not in reported:
                        hook.pytest_runtest_logreport(report=_crash_report(item, exit_code))

    def _run_subprocess(self, session, n):
        q: queue.Queue[dict | None] = queue.Queue()
        workers: list[tuple[subprocess.Popen, list]] = []

        for i in range(n):
            bucket = session.items[i::n]
            env = {**os.environ, _WORKER_ENV: "1"}
            cmd = [sys.executable, "-m", "pytest", "-p", "no:terminal", *[item.nodeid for item in bucket]]
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
            workers.append((proc, bucket))
            threading.Thread(target=_read_subprocess, args=(proc, q), daemon=True).start()

        hook = session.config.hook
        reported: set[str] = set()
        pending = n
        while pending:
            data = q.get()
            if data is None:
                pending -= 1
            else:
                report = hook.pytest_report_from_serializable(config=session.config, data=data)
                if report.when == "call":
                    reported.add(report.nodeid)
                hook.pytest_runtest_logreport(report=report)

        for proc, bucket in workers:
            proc.wait()
            if proc.returncode > 1:
                for item in bucket:
                    if item.nodeid not in reported:
                        hook.pytest_runtest_logreport(report=_crash_report(item, proc.returncode))


def _crash_report(item, exit_code: int) -> TestReport:
    return TestReport(
        nodeid=item.nodeid,
        location=item.location,
        keywords=dict(item.keywords),
        outcome="failed",
        longrepr=f"Worker process crashed with exit code {exit_code}",
        when="call",
        sections=[],
        duration=0.0,
        start=0.0,
        stop=0.0,
    )


def _fork_worker(items, write_fd):
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    out = os.fdopen(write_fd, "w")
    for i, item in enumerate(items):
        nextitem = items[i + 1] if i + 1 < len(items) else None
        for report in _runtestprotocol(item, nextitem=nextitem, log=False):
            data = item.config.hook.pytest_report_to_serializable(config=item.config, report=report)
            out.write(json.dumps(data) + "\n")
    out.flush()
    out.close()
    os._exit(0)


def _read_pipe(rf, q: queue.Queue) -> None:
    for line in rf:
        line = line.rstrip()
        if line:
            q.put(json.loads(line))
    q.put(None)


def _read_subprocess(proc: subprocess.Popen, q: queue.Queue) -> None:
    for line in proc.stdout:
        if line.startswith(_SENTINEL):
            q.put(json.loads(line[1:]))
    q.put(None)
