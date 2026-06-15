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

A divergence is classified by whether the op-counts are still advancing:

  * HARD hang -- a dead/unresponsive rank, or an op-count mismatch whose counts
    are *frozen* (unchanged) across polls. A real deadlock. After
    ``CONFIRM_COUNT`` consecutive hard polls the callback captures native stack
    traces from every worker and, in ``fail`` mode, raises
    :class:`~ray.train.v2.api.exceptions.NCCLHangError` (terminal / non-retryable
    -- NCCL hangs are usually deterministic, so the run fails fast rather than
    restarting into the same hang). ``observe`` mode captures + logs but does
    not raise.
  * SOFT hang -- the same op-count mismatch persists but the counts keep
    *advancing*: the job is making progress while chronically uneven (a slow
    straggler, load imbalance). This is logged every ``CONFIRM_COUNT`` polls and
    tolerated; the run is never failed on a soft hang.

Both debounce over ``CONFIRM_COUNT`` consecutive polls; a healthy poll, or a
change in the *nature* of the divergence (different ranks or op types), resets
the counters.
"""

import json
import logging
import os
import re
import subprocess
from collections import Counter
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

    ``dead_ranks`` and the mismatched ranks are the hang signals; a report is
    healthy when both are empty. ``rank_counts`` maps each *mismatched* rank to
    its collective-count signature (a sorted tuple of ``(op_name, count)``
    pairs); the counts let the callback tell a *frozen* divergence (a hard hang)
    from one whose counts are still *advancing* (a soft hang) across polls.
    """

    dead_ranks: Set[int] = field(default_factory=set)
    rank_counts: Dict[int, tuple] = field(default_factory=dict)

    @property
    def mismatched_ranks(self) -> Set[int]:
        return set(self.rank_counts)

    @property
    def healthy(self) -> bool:
        return not self.dead_ranks and not self.rank_counts


# Keys an ``ncclras`` rank / missing-rank entry may use for the global rank.
_RANK_ID_KEYS = ("rank", "global_rank", "globalRank", "rank_id", "rankId")


def _rank_id(entry: Dict) -> Optional[int]:
    """Extract a global rank index from a rank / missing-rank entry."""
    for key in _RANK_ID_KEYS:
        if key in entry:
            try:
                return int(entry[key])
            except (TypeError, ValueError):
                return None
    return None


def _counts_signature(counts) -> Optional[tuple]:
    """Hashable signature of a per-rank ``collective_counts`` dict."""
    if not isinstance(counts, dict):
        return None
    sig = []
    for name, value in counts.items():
        try:
            sig.append((str(name), int(value)))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(sig))


def _dead_ranks_from_comm(comm: Dict) -> Set[int]:
    """Ranks a communicator's ``missing_ranks`` marks dead or unresponsive.

    The ``considered_dead`` / ``unresponsive`` flags appear either at the top
    level of a ``missing_ranks`` entry (documented schema) or nested under a
    ``"status"`` object (NCCL 2.28.9+ live output); both layouts are handled.
    """
    out: Set[int] = set()
    missing = comm.get("missing_ranks")
    if not isinstance(missing, list):
        return out
    for entry in missing:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if not isinstance(status, dict):
            status = entry
        if status.get("considered_dead") or status.get("unresponsive"):
            rid = _rank_id(entry)
            if rid is not None:
                out.add(rid)
    return out


def _mismatched_counts_from_comm(comm: Dict) -> Dict[int, tuple]:
    """Mismatched ranks -> their collective-count signature.

    RAS does not emit an explicit "mismatch" flag in JSON; it is derived here by
    comparing each rank's ``collective_counts`` within a communicator. Ranks
    whose counts differ from the modal (most common) signature are flagged, and
    their signature is returned so the callback can later tell a frozen
    divergence from an advancing one. Transient skew (a rank a step ahead) is
    expected -- the callback debounces over consecutive polls before acting.
    """
    ranks = comm.get("ranks")
    if not isinstance(ranks, list):
        return {}
    sigs: Dict[int, tuple] = {}
    for entry in ranks:
        if not isinstance(entry, dict):
            continue
        rid = _rank_id(entry)
        sig = _counts_signature(entry.get("collective_counts"))
        if rid is not None and sig is not None:
            sigs[rid] = sig
    # Need at least two distinct signatures to call anything an outlier.
    if len(sigs) < 2 or len(set(sigs.values())) < 2:
        return {}
    modal, _ = Counter(sigs.values()).most_common(1)[0]
    return {rid: sig for rid, sig in sigs.items() if sig != modal}


# Matches a primitive JSON value (number / string / true / false / null) that is
# immediately followed -- across a newline, with NO comma -- by the next object
# key. This is the shape of the NCCL 2.28.9 ``missing_ranks[]`` serializer bug
# (``"nvml_dev": 0`` then ``"status": {``). It only matches malformed input, so
# the repair is a no-op on valid JSON.
_MISSING_COMMA_RE = re.compile(r'([\d"el])(\s*\n\s*)("[^"\n]*"\s*:)')


def _repair_missing_commas(stdout: str) -> str:
    """Insert commas omitted by the NCCL 2.28.9 JSON serializer.

    NCCL 2.28.9 emits ``missing_ranks[]`` entries with no comma before the
    ``"status"`` field, e.g.::

        "nvml_dev": 0
        "status": { ... }

    which is invalid JSON. Insert the missing comma between such a value and the
    following key. Targets only that malformed pattern (see ``_MISSING_COMMA_RE``)
    so it leaves well-formed reports untouched.
    """
    return _MISSING_COMMA_RE.sub(r"\1,\2\3", stdout)


def _interpret_ras_status(stdout: str) -> Optional[RASReport]:
    """Parse ``ncclras -f json`` output into a :class:`RASReport`.

    Targets the documented NCCL >= 2.28.7 JSON schema::

        {
          "nccl_version": ..., "communicators_count": N,
          "communicators": [
            {
              "hash": ..., "secondary_hash": ...,
              "size": ..., "ranks_count": ..., "missing_ranks_count": ...,
              "ranks": [
                {"rank": 0, "status": {...}, "collective_counts": {...}}, ...
              ],
              "missing_ranks": [
                {"rank": 3, "unresponsive": true, "considered_dead": true}, ...
              ]
            }, ...
          ]
        }

    Hang signals, aggregated across all communicators:
      * ``dead_ranks``       -- ``missing_ranks`` entries flagged
        ``considered_dead`` or ``unresponsive``.
      * ``mismatched_ranks`` -- ranks whose per-rank ``collective_counts``
        deviate from the modal signature (op-count skew).

    Returns ``None`` if the output is not valid JSON or lacks a
    ``communicators`` array (an unexpected shape we refuse to guess at).

    NOTE: Validated against the *documented* schema, not yet against live
    ``ncclras`` output. The exact rank-id key inside ``ranks`` / ``missing_ranks``
    and the location of ``collective_counts`` may differ across NCCL versions;
    capture real output (see ``scratch/nccl-ras-examples/README.md``) and pin
    these before relying on automatic hang detection.
    """
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        # NCCL 2.28.9 serializes ``missing_ranks[]`` entries with a missing
        # comma before ``"status"`` (fixed in 2.30.7), producing invalid JSON.
        # Repair that one pattern and retry once before giving up.
        try:
            data = json.loads(_repair_missing_commas(stdout))
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(data, dict):
        return None

    communicators = data.get("communicators")
    if not isinstance(communicators, list):
        return None

    report = RASReport()
    for comm in communicators:
        if not isinstance(comm, dict):
            continue
        report.dead_ranks |= _dead_ranks_from_comm(comm)
        report.rank_counts.update(_mismatched_counts_from_comm(comm))
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
            self._reset_counters()
            self._prev_episode_key = None
            self._prev_frozen_key = None
            return

        # Classify this poll against the previous one. The *episode key* (dead
        # ranks + the mismatch op *shape*, counts dropped) decides whether this
        # is the same anomaly as last poll; the *frozen key* (the same, but with
        # op counts) decides whether it has advanced since.
        episode_key = self._episode_key(report)
        frozen_key = self._frozen_key(report)
        new_episode = episode_key != self._prev_episode_key
        frozen = frozen_key == self._prev_frozen_key
        self._prev_episode_key = episode_key
        self._prev_frozen_key = frozen_key

        if new_episode:
            # A new or changed anomaly restarts the debounce. Dead ranks are a
            # hard signal immediately; a fresh op-count mismatch needs a prior
            # poll to tell frozen from advancing, so it is only a baseline.
            self._reset_counters()
            if not report.dead_ranks:
                return

        if report.dead_ranks or frozen:
            self._record_hard_poll(report)
        else:
            self._record_soft_poll(report)

    def _record_hard_poll(self, report: RASReport):
        self._hard_polls += 1
        self._soft_polls = 0
        logger.warning(
            "NCCL RAS anomaly FROZEN (hard hang %d/%d): dead_ranks=%s "
            "mismatched_ranks=%s",
            self._hard_polls,
            self._confirm_count,
            sorted(report.dead_ranks),
            sorted(report.mismatched_ranks),
        )
        # One poll before failing, log the raw JSON at info level so successive
        # snapshots can be diffed to confirm the counts are genuinely frozen.
        if self._hard_polls == self._confirm_count - 1 and self._last_raw_json:
            logger.info(
                "NCCL RAS frozen snapshot (one poll before failing):\n%s",
                self._last_raw_json,
            )
        if self._hard_polls >= self._confirm_count:
            self._handle_confirmed_hang(report)

    def _record_soft_poll(self, report: RASReport):
        self._soft_polls += 1
        self._hard_polls = 0
        logger.warning(
            "NCCL RAS op-count divergence persisting but ADVANCING "
            "(soft hang %d/%d): mismatched_ranks=%s",
            self._soft_polls,
            self._confirm_count,
            sorted(report.mismatched_ranks),
        )
        # The job is still progressing, so do NOT fail the run. Log the latest
        # RAS snapshot on the first poll at the threshold and every
        # ``CONFIRM_COUNT`` polls after, so a chronic soft hang stays visible
        # without spamming on every poll.
        if self._soft_polls % self._confirm_count == 0 and self._last_raw_json:
            logger.warning(
                "NCCL RAS soft hang (job advancing but chronically uneven); "
                "mismatched_ranks=%s. Latest RAS JSON:\n%s",
                sorted(report.mismatched_ranks),
                self._last_raw_json,
            )

    def _handle_confirmed_hang(self, report: RASReport):
        # Capture stacks once per confirmed episode.
        if not self._hang_reported:
            self._hang_reported = True
            self._hang_total += 1
            dump_dir = self._capture_stacks()
            if self._last_raw_json:
                logger.error("NCCL RAS JSON at hang:\n%s", self._last_raw_json)
            logger.error(
                "NCCL hang confirmed: frozen for %d consecutive polls "
                "(dead_ranks=%s, mismatched_ranks=%s). Stack traces written "
                "to %s.",
                self._confirm_count,
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
                RuntimeError("NCCL RAS: collective ops frozen / diverged from peers."),
            )

        error_message = (
            f"NCCL RAS detected a hang: frozen for {self._confirm_count} "
            f"consecutive polls. Dead ranks: {sorted(report.dead_ranks)}; "
            f"mismatched ranks: {sorted(report.mismatched_ranks)}."
        )
        # Raising propagates through poll_status -> _poll_workers -> _step and
        # is routed through the failure policy, which treats NCCLHangError as
        # terminal (non-retryable), failing the run.
        raise NCCLHangError(error_message, worker_failures)

    # -- helpers -----------------------------------------------------------

    def _reset_detection_state(self):
        self._reset_counters()
        # Previous poll's episode / frozen keys, for the across-poll comparison.
        self._prev_episode_key = None
        self._prev_frozen_key = None
        # Raw JSON of the most recent successful query, reused for the pre-fail
        # snapshot and the soft-hang / confirmed-hang logs.
        self._last_raw_json: Optional[str] = None
        # Force a query on the next poll after (re)start.
        self._last_query_time = float("-inf")
        if not hasattr(self, "_hang_total"):
            self._hang_total = 0

    def _reset_counters(self):
        """Reset the consecutive-poll counters and the capture-once latch."""
        self._hard_polls = 0
        self._soft_polls = 0
        self._hang_reported = False

    @staticmethod
    def _episode_key(report: RASReport) -> tuple:
        """Identity of the anomaly ignoring op *counts*: the dead ranks plus, per
        mismatched rank, the set of op *names*. Two polls share an episode iff
        this matches; a change (different ranks or ops) restarts the debounce.
        """
        shape = frozenset(
            (rank, frozenset(op for op, _ in sig))
            for rank, sig in report.rank_counts.items()
        )
        return (frozenset(report.dead_ranks), shape)

    @staticmethod
    def _frozen_key(report: RASReport) -> tuple:
        """Identity *including* op counts: matches across polls only when nothing
        has advanced (a frozen / hard hang). Dead-only reports carry no counts,
        so they compare equal poll-to-poll and are inherently frozen.
        """
        return (frozenset(report.dead_ranks), frozenset(report.rank_counts.items()))

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

            # Stash the raw JSON for the pre-fail snapshot / soft-hang logs.
            self._last_raw_json = result["stdout"]
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
