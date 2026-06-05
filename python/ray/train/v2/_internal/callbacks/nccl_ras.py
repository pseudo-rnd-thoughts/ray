"""NCCL RAS-based hang detection for Ray Train v2.

NCCL ships a Reliability/Availability/Serviceability (RAS) subsystem (NCCL
>= 2.24) that runs a monitoring thread inside *every* NCCL process (one per
GPU/rank). Those threads form a peer mesh and track job health: dead/
unresponsive ranks and collective op-count mismatches between ranks. The
``ncclras`` client connects to *any single* rank's RAS socket (default
``localhost:28028``) and returns the **whole-job** aggregated view.

The controller is not part of the NCCL job, so it has no RAS thread and cannot
reach the localhost-bound RAS socket directly. This callback therefore runs on
the controller, and on each (throttled) poll asks a worker to shell out to
``ncclras`` against its local socket and return the parsed status. A single
query covers all ranks.

On a confirmed hang (debounced over several consecutive reports) the callback
captures native stack traces from every worker and, in ``fail`` mode, raises
:class:`~ray.train.v2.api.exceptions.NCCLHangError`. The default failure policy
treats this as terminal (non-retryable) -- NCCL hangs are usually deterministic,
so the run fails fast with the captured stacks rather than restarting into the
same hang.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import ray
from ray.exceptions import GetTimeoutError
from ray.train.v2._internal.constants import (
    DEFAULT_NCCL_RAS_ACTION,
    DEFAULT_NCCL_RAS_CONFIRM_COUNT,
    DEFAULT_NCCL_RAS_POLL_INTERVAL_S,
    DEFAULT_NCCL_RAS_QUERY_TIMEOUT_S,
    DEFAULT_NCCLRAS_BINARY_PATH,
    NCCL_RAS_ACTION_ENV_VAR,
    NCCL_RAS_ACTION_FAIL,
    NCCL_RAS_ACTION_OBSERVE,
    NCCL_RAS_ADDR_ENV_VAR,
    NCCL_RAS_CONFIRM_COUNT_ENV_VAR,
    NCCL_RAS_POLL_INTERVAL_S_ENV_VAR,
    NCCL_RAS_QUERY_TIMEOUT_S_ENV_VAR,
    NCCLRAS_BINARY_PATH_ENV_VAR,
)
from ray.train.v2._internal.execution.callback import (
    ControllerCallback,
    WorkerGroupCallback,
)
from ray.train.v2._internal.execution.context import TrainRunContext
from ray.train.v2._internal.logging.logging import LoggingManager
from ray.train.v2._internal.util import time_monotonic
from ray.train.v2.api.exceptions import NCCLHangError

logger = logging.getLogger(__name__)

# Overall budget for collecting stack traces from all workers, so capture can
# never itself block the failure path.
_STACK_DUMP_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Remote functions (executed on workers via ``Worker.execute_async``).
# ---------------------------------------------------------------------------


def _parse_ras_addr(addr: str) -> "tuple[str, int]":
    """Parse an ``NCCL_RAS_ADDR`` value (``host:port``) into ``(host, port)``.

    Handles bare hosts (default port), ``host:port``, and bracketed IPv6 such
    as ``[::1]:28028``. Falls back to ``localhost:28028`` on malformed input.
    """
    addr = (addr or "").strip()
    host, port = "localhost", 28028
    try:
        if addr.startswith("["):  # [ipv6](:port)?
            end = addr.index("]")
            host = addr[1:end]
            rest = addr[end + 1 :]
            if rest.startswith(":") and rest[1:]:
                port = int(rest[1:])
        elif ":" in addr:
            h, _, p = addr.rpartition(":")
            host = h or host
            port = int(p) if p else port
        elif addr:
            host = addr
    except ValueError:
        return "localhost", 28028
    return host, port


def _run_ncclras_query(binary_path: str, timeout_s: float) -> Dict[str, object]:
    """Run the ``ncclras`` client on the worker and return its JSON output.

    The RAS listen address is read from the worker's own ``NCCL_RAS_ADDR`` so
    the client connects to wherever NCCL's RAS subsystem is listening on this
    node (default ``localhost:28028``).

    Returns a dict ``{"ok": bool, ...}``. On success ``stdout`` holds the raw
    JSON string. On failure ``reason`` distinguishes a missing binary (so the
    detector can degrade to a no-op) from transient errors.
    """
    host, port = _parse_ras_addr(
        os.environ.get(NCCL_RAS_ADDR_ENV_VAR, "localhost:28028")
    )
    cmd = [
        binary_path,
        "-f",
        "json",
        "-h",
        host,
        "-p",
        str(port),
        "-t",
        str(int(timeout_s)),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s + 5
        )
    except FileNotFoundError:
        return {"ok": False, "reason": "binary_not_found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout"}
    except Exception as e:  # noqa: BLE001 - never let the query crash the worker
        return {"ok": False, "reason": f"error: {e}"}

    if proc.returncode != 0:
        return {
            "ok": False,
            "reason": f"exit_{proc.returncode}",
            "stderr": (proc.stderr or "")[:500],
        }
    return {"ok": True, "stdout": proc.stdout}


def _dump_self_native_stack(pyspy_timeout_s: float) -> str:
    """Dump native + Python stacks of the current worker process.

    Uses ``py-spy`` for native (C/C++ NCCL) frames, falling back to
    ``faulthandler`` (Python-only) if py-spy is unavailable.
    """
    pid = os.getpid()
    stderr = ""
    try:
        proc = subprocess.run(
            ["py-spy", "dump", "--pid", str(pid), "--native"],
            capture_output=True,
            text=True,
            timeout=pyspy_timeout_s,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
        stderr = (proc.stderr or "").strip() or f"py-spy exited {proc.returncode}"
    except FileNotFoundError:
        stderr = "py-spy not installed"
    except subprocess.TimeoutExpired:
        stderr = "py-spy timed out"
    except Exception as e:  # noqa: BLE001
        stderr = f"py-spy error: {e}"

    # Python-only fallback: dump every thread's stack. Unlike
    # ``faulthandler.dump_traceback`` (which requires a real file descriptor),
    # this works with an in-memory buffer and is cross-platform. It cannot show
    # native (C/C++ NCCL) frames -- install py-spy for those.
    import sys
    import traceback

    lines = [f"[py-spy unavailable: {stderr}; Python-only traceback follows]"]
    for thread_id, frame in sys._current_frames().items():
        lines.append(f"\n# Thread {thread_id}")
        lines.append("".join(traceback.format_stack(frame)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAS report parsing.
# ---------------------------------------------------------------------------


@dataclass
class RASReport:
    """Structured summary of an ``ncclras`` JSON report.

    ``dead_ranks`` and ``mismatched_ranks`` are the hang signals; a report is
    healthy when both are empty.
    """

    dead_ranks: Set[int] = field(default_factory=set)
    mismatched_ranks: Set[int] = field(default_factory=set)

    @property
    def healthy(self) -> bool:
        return not self.dead_ranks and not self.mismatched_ranks

    @property
    def bad_signature(self) -> frozenset:
        """Identity of the anomaly, used to debounce repeated identical reports."""
        return frozenset(self.dead_ranks | self.mismatched_ranks)


def _coerce_ranks(value) -> Set[int]:
    ranks: Set[int] = set()
    if isinstance(value, (list, tuple, set)):
        for item in value:
            try:
                ranks.add(int(item))
            except (TypeError, ValueError):
                continue
    return ranks


def _interpret_ras_status(stdout: str) -> Optional[RASReport]:
    """Parse ``ncclras -f json`` output into a :class:`RASReport`.

    Returns ``None`` if the output is not valid JSON.

    NOTE: The exact ``ncclras`` JSON schema can vary across NCCL versions. This
    parser reads the documented signals (dead/unresponsive processes and
    collective op-count mismatches) and tolerates a few key spellings. Validate
    against the ``ncclras -f json`` output of the deployed NCCL version before
    relying on automatic hang detection.
    """
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    report = RASReport()

    # Dead / unresponsive ranks reported at the top level.
    for key in ("deadRanks", "dead_ranks", "deadProcesses", "unresponsiveRanks"):
        report.dead_ranks |= _coerce_ranks(data.get(key))

    # Per-communicator op-count mismatches.
    communicators = data.get("communicators") or data.get("comms") or []
    if isinstance(communicators, list):
        for comm in communicators:
            if not isinstance(comm, dict):
                continue
            has_mismatch = bool(
                comm.get("collMismatch")
                or comm.get("mismatch")
                or comm.get("opCountMismatch")
            )
            mismatched = set()
            for key in ("mismatchedRanks", "mismatched_ranks", "outOfSyncRanks"):
                mismatched |= _coerce_ranks(comm.get(key))
            if has_mismatch or mismatched:
                report.mismatched_ranks |= mismatched

    return report


# ---------------------------------------------------------------------------
# Callback.
# ---------------------------------------------------------------------------


class NCCLRASCallback(WorkerGroupCallback, ControllerCallback):
    """Detects NCCL hangs via the RAS subsystem and fails the run.

    Registered (opt-in) on the controller. See module docstring for topology.
    """

    def __init__(self):
        # Read configuration on the driver so it travels with the pickled
        # callback to the controller (no env propagation required).
        self._poll_interval_s = float(
            os.environ.get(
                NCCL_RAS_POLL_INTERVAL_S_ENV_VAR, DEFAULT_NCCL_RAS_POLL_INTERVAL_S
            )
        )
        self._confirm_count = int(
            os.environ.get(
                NCCL_RAS_CONFIRM_COUNT_ENV_VAR, DEFAULT_NCCL_RAS_CONFIRM_COUNT
            )
        )
        self._binary_path = os.environ.get(
            NCCLRAS_BINARY_PATH_ENV_VAR, DEFAULT_NCCLRAS_BINARY_PATH
        )
        self._query_timeout_s = float(
            os.environ.get(
                NCCL_RAS_QUERY_TIMEOUT_S_ENV_VAR, DEFAULT_NCCL_RAS_QUERY_TIMEOUT_S
            )
        )
        self._action = os.environ.get(
            NCCL_RAS_ACTION_ENV_VAR, DEFAULT_NCCL_RAS_ACTION
        ).lower()
        if self._action not in (NCCL_RAS_ACTION_FAIL, NCCL_RAS_ACTION_OBSERVE):
            logger.warning(
                "Unknown %s=%r; defaulting to %r.",
                NCCL_RAS_ACTION_ENV_VAR,
                self._action,
                NCCL_RAS_ACTION_FAIL,
            )
            self._action = NCCL_RAS_ACTION_FAIL

        # Controller-side state (initialized in callbacks below).
        self._run_id: Optional[str] = None
        self._worker_group = None

        self._reset_detection_state()
        # One-time degradation (e.g. missing binary) so we stop querying.
        self._degraded = False

    # -- lifecycle ---------------------------------------------------------

    def after_controller_start(self, train_run_context: TrainRunContext):
        # Used to namespace the stack-dump directory written on a confirmed hang.
        self._run_id = train_run_context.run_id

    def after_worker_group_start(self, worker_group):
        # Cache the worker group to issue queries / stack dumps, and reset
        # detection state for the new attempt.
        self._worker_group = worker_group
        self._reset_detection_state()
        return super().after_worker_group_start(worker_group)

    def before_worker_group_shutdown(self, worker_group):
        self._worker_group = None
        return super().before_worker_group_shutdown(worker_group)

    # -- detection ---------------------------------------------------------

    def after_worker_group_poll_status(self, worker_group_status):
        if self._degraded or self._worker_group is None:
            return

        now = time_monotonic()
        if now - self._last_query_time < self._poll_interval_s:
            return
        self._last_query_time = now

        report = self._query_ras()
        if report is None:
            return

        if report.healthy:
            self._reset_detection_state(keep_query_time=True)
            return

        # Debounce: require the same anomaly across consecutive reports.
        if report.bad_signature == self._last_bad_signature:
            self._consecutive_bad += 1
        else:
            self._consecutive_bad = 1
            self._last_bad_signature = report.bad_signature

        logger.warning(
            "NCCL RAS anomaly (%d/%d): dead_ranks=%s mismatched_ranks=%s",
            self._consecutive_bad,
            self._confirm_count,
            sorted(report.dead_ranks),
            sorted(report.mismatched_ranks),
        )

        if self._consecutive_bad < self._confirm_count:
            return

        self._handle_confirmed_hang(report)

    def _handle_confirmed_hang(self, report: RASReport):
        # Capture stacks once per confirmed episode.
        if not self._hang_reported:
            self._hang_reported = True
            self._hang_total += 1
            dump_dir = self._capture_stacks()
            logger.error(
                "NCCL hang confirmed (dead_ranks=%s, mismatched_ranks=%s). "
                "Stack traces written to %s.",
                sorted(report.dead_ranks),
                sorted(report.mismatched_ranks),
                dump_dir,
            )

        if self._action == NCCL_RAS_ACTION_OBSERVE:
            return

        worker_failures: Dict[int, Exception] = {}
        for rank in report.dead_ranks:
            worker_failures[rank] = RuntimeError(
                "NCCL RAS declared this rank dead/unresponsive."
            )
        for rank in report.mismatched_ranks:
            worker_failures.setdefault(
                rank,
                RuntimeError("NCCL RAS reported a collective op-count mismatch."),
            )

        error_message = (
            f"NCCL RAS detected a hang confirmed over {self._consecutive_bad} "
            f"consecutive reports. Dead ranks: {sorted(report.dead_ranks)}; "
            f"mismatched ranks: {sorted(report.mismatched_ranks)}."
        )
        # Raising propagates through poll_status -> _poll_workers -> _step and
        # is routed through the failure policy, which treats NCCLHangError as
        # terminal (non-retryable), failing the run.
        raise NCCLHangError(error_message, worker_failures)

    # -- helpers -----------------------------------------------------------

    def _reset_detection_state(self, keep_query_time: bool = False):
        self._consecutive_bad = 0
        self._last_bad_signature: frozenset = frozenset()
        self._hang_reported = False
        if not keep_query_time:
            # Force a query on the next poll after (re)start.
            self._last_query_time = float("-inf")
        if not hasattr(self, "_hang_total"):
            self._hang_total = 0

    def _candidate_workers(self):
        """Workers to query, rank 0 first (a dead rank 0 falls back to peers)."""
        try:
            return list(self._worker_group.get_workers())
        except Exception:  # noqa: BLE001 - worker group may be inactive
            return []

    def _query_ras(self) -> Optional[RASReport]:
        for worker in self._candidate_workers():
            try:
                ref = worker.execute_async(
                    _run_ncclras_query,
                    self._binary_path,
                    self._query_timeout_s,
                )
                result = ray.get(ref, timeout=self._query_timeout_s + 10)
            except GetTimeoutError:
                logger.debug("ncclras query timed out on a worker; trying next.")
                continue
            except Exception as e:  # noqa: BLE001
                logger.debug("ncclras query failed on a worker: %s", e)
                continue

            if not result.get("ok"):
                if result.get("reason") == "binary_not_found":
                    logger.warning(
                        "`ncclras` binary %r not found on workers; disabling NCCL "
                        "RAS hang detection. Set %s to its path.",
                        self._binary_path,
                        NCCLRAS_BINARY_PATH_ENV_VAR,
                    )
                    self._degraded = True
                    return None
                logger.debug("ncclras query unsuccessful: %s", result.get("reason"))
                continue

            report = _interpret_ras_status(result["stdout"])
            if report is None:
                logger.debug("Could not parse ncclras JSON output.")
                continue
            return report
        return None

    def _capture_stacks(self) -> Optional[str]:
        """Fan out a native stack dump to every worker; write to the log dir.

        Bounded by ``_STACK_DUMP_TIMEOUT_S`` so it can never block the failure path.
        Returns the directory the dumps were written to (``None`` if not in a
        Ray session, in which case dumps are logged instead).
        """
        workers = self._candidate_workers()
        if not workers:
            return None

        ref_to_rank = {}
        for rank, worker in enumerate(workers):
            try:
                ref = worker.execute_async(
                    _dump_self_native_stack, _STACK_DUMP_TIMEOUT_S - 5
                )
                ref_to_rank[ref] = rank
            except Exception as e:  # noqa: BLE001
                logger.debug("Failed to launch stack dump on rank %d: %s", rank, e)

        if not ref_to_rank:
            return None

        ready, _ = ray.wait(
            list(ref_to_rank),
            num_returns=len(ref_to_rank),
            timeout=_STACK_DUMP_TIMEOUT_S,
        )

        dumps: Dict[int, str] = {}
        for ref in ready:
            rank = ref_to_rank[ref]
            try:
                dumps[rank] = ray.get(ref)
            except Exception as e:  # noqa: BLE001
                dumps[rank] = f"<failed to collect stack: {e}>"

        return self._write_stack_dumps(dumps)

    def _write_stack_dumps(self, dumps: Dict[int, str]) -> Optional[str]:
        log_dir = LoggingManager.get_log_directory()
        if log_dir is None:
            for rank, dump in sorted(dumps.items()):
                logger.error("NCCL hang stack trace (rank %d):\n%s", rank, dump)
            return None

        run_id = self._run_id or "unknown"
        dump_dir = os.path.join(log_dir, f"nccl_hang_{run_id}_{self._hang_total}")
        try:
            os.makedirs(dump_dir, exist_ok=True)
            for rank, dump in sorted(dumps.items()):
                with open(os.path.join(dump_dir, f"rank_{rank}.txt"), "w") as f:
                    f.write(dump)
        except OSError as e:
            logger.warning("Failed to write NCCL hang stack dumps: %s", e)
            return None
        return dump_dir
