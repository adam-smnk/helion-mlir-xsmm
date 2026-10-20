"""Pytest configuration for the MLIR backend suite."""

from __future__ import annotations

import ctypes
import importlib
import multiprocessing
import multiprocessing.forkserver
import os
import pathlib
import sys
import tempfile

import pytest

# Imported once by the fork server, so an isolated test only pays for its own work.
_PRELOAD = [
    "torch",
    "helion",
    "helion.language",
    "mlir.ir",
    "lighthouse.pipeline.driver",
    "lighthouse.execution.runner",
    "torch_mlir.extras.fx_importer",
    "helion_mlir_backend",
    "helion_mlir_backend._compiler.execution",
    "helion_mlir_backend._compiler.mlir.driver",
    "tests.harness",
]
_context: multiprocessing.context.ForkServerContext | None = None

if (_workers := os.environ.get("PYTEST_XDIST_WORKER_COUNT")) is not None:
    # xdist workers share the cores; per-process OpenMP/torch pools would oversubscribe.
    os.environ.setdefault(
        "OMP_NUM_THREADS", str(max(2, (os.cpu_count() or 2) // int(_workers)))
    )
# Test kernels run fixed configs; autotuning tests re-enable the search explicitly.
os.environ["HELION_DISALLOW_AUTOTUNING"] = "1"


def _isolated_context() -> multiprocessing.context.ForkServerContext:
    global _context
    if _context is None:
        _context = multiprocessing.get_context("forkserver")
        _context.set_forkserver_preload(_PRELOAD)
    return _context


def _run_isolated(
    module: str, name: str, kwargs: dict, env: dict[str, str], log: str
) -> None:
    """Child side: run one test function with the parent's environment."""
    _disable_core_dumps()
    fd = os.open(log, os.O_WRONLY)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.environ.clear()
    os.environ.update(env)
    getattr(importlib.import_module(module), name)(**kwargs)
    sys.stdout.flush()
    sys.stderr.flush()


def _disable_core_dumps() -> None:
    """A crashing probe would otherwise dump the whole process (seconds per crash)."""
    if sys.platform.startswith("linux"):
        pr_set_dumpable = 4
        ctypes.CDLL(None).prctl(pr_set_dumpable, 0, 0, 0, 0)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="re-record tests/golden/*.txt IR signatures instead of comparing",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "isolated: run the test body in a forked child process so a native crash "
        "only fails that test",
    )
    config.addinivalue_line("markers", "slow: long-running test")


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """``-n auto`` (the default) parallelizes whole-suite runs only.

    Runs that name test files or node ids, or select with ``-k``, stay in-process:
    starting workers costs more than such runs take. Pass ``-n <count>`` to force.
    """
    targeted = bool(config.option.keyword) or any(
        not (config.invocation_params.dir / arg.split("::")[0]).is_dir()
        for arg in config.args
    )
    return 0 if targeted else None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Isolated tests last, so the fork server finishes importing meanwhile."""
    items.sort(key=lambda item: item.get_closest_marker("isolated") is not None)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Start the fork server early so its imports overlap the in-process tests."""
    if any(item.get_closest_marker("isolated") for item in session.items):
        _isolated_context()
        multiprocessing.forkserver.ensure_running()


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    if pyfuncitem.get_closest_marker("isolated") is None:
        return None
    if pyfuncitem.cls is not None:
        raise pytest.UsageError(
            "@pytest.mark.isolated supports module-level tests only"
        )
    kwargs = {
        name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames
    }
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as handle:
        log = handle.name
    try:
        process = _isolated_context().Process(
            target=_run_isolated,
            args=(
                pyfuncitem.module.__name__,
                pyfuncitem.originalname,
                kwargs,
                dict(os.environ),
                log,
            ),
        )
        process.start()
        process.join()
        output = pathlib.Path(log).read_text()
    finally:
        os.unlink(log)
    if process.exitcode != 0:
        tail = "\n".join(output.splitlines()[-25:])
        how = (
            f"crashed with signal {-process.exitcode}"
            if process.exitcode < 0
            else f"failed (exit code {process.exitcode})"
        )
        pytest.fail(f"isolated test {how}:\n{tail}", pytrace=False)
    return True
