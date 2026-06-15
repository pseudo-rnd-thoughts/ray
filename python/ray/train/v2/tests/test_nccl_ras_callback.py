import json
import sys
from unittest.mock import MagicMock

import pytest

import ray.train.v2._internal.callbacks.nccl_ras as nccl_ras_module
from ray.train.v2._internal.callbacks.nccl_ras import (
    NCCLRASCallback,
    RASReport,
    _interpret_ras_status,
)
from ray.train.v2._internal.constants import (
    NCCL_RAS_ACTION_ENV_VAR,
    NCCL_RAS_ACTION_FAIL,
    NCCL_RAS_ACTION_OBSERVE,
    NCCL_RAS_CONFIRM_COUNT_ENV_VAR,
    NCCL_RAS_POLL_INTERVAL_S_ENV_VAR,
)
from ray.train.v2.api.exceptions import NCCLHangError
from ray.train.v2.tests.util import create_dummy_run_context


@pytest.fixture
def increasing_time(monkeypatch):
    """Make time_monotonic strictly increasing so throttling never blocks."""
    counter = {"t": 0.0}

    def _time():
        counter["t"] += 1.0
        return counter["t"]

    monkeypatch.setattr(nccl_ras_module, "time_monotonic", _time)


def _make_callback(monkeypatch, action, confirm_count, reports):
    """Build a callback whose RAS query yields the given sequence of reports."""
    monkeypatch.setenv(NCCL_RAS_ACTION_ENV_VAR, action)
    monkeypatch.setenv(NCCL_RAS_CONFIRM_COUNT_ENV_VAR, str(confirm_count))
    monkeypatch.setenv(NCCL_RAS_POLL_INTERVAL_S_ENV_VAR, "0")

    callback = NCCLRASCallback()
    callback.after_controller_start(create_dummy_run_context())
    callback._worker_group = MagicMock()

    report_iter = iter(reports)
    callback._query_ras = lambda: next(report_iter, None)

    captured = []
    callback._capture_stacks = lambda: captured.append(True) or "/tmp/dump"

    return callback, captured


# --------------------------------------------------------------------------
# RAS JSON parsing
# --------------------------------------------------------------------------


def test_interpret_healthy():
    report = _interpret_ras_status(
        json.dumps(
            {
                "communicators": [
                    {
                        "size": 2,
                        "ranks": [
                            {"rank": 0, "collective_counts": {"AllReduce": 10}},
                            {"rank": 1, "collective_counts": {"AllReduce": 10}},
                        ],
                        "missing_ranks": [],
                    }
                ]
            }
        )
    )
    assert report is not None
    assert report.healthy
    assert report.mismatched_ranks == set()


def test_interpret_dead_ranks():
    report = _interpret_ras_status(
        json.dumps(
            {
                "communicators": [
                    {
                        "missing_ranks": [
                            {"rank": 3, "considered_dead": True, "unresponsive": True},
                            {"rank": 5, "considered_dead": True},
                        ]
                    }
                ]
            }
        )
    )
    assert report.dead_ranks == {3, 5}
    assert not report.healthy


def test_interpret_unresponsive_counts_as_dead():
    # Unresponsive-but-not-yet-dead is still a hang signal.
    report = _interpret_ras_status(
        json.dumps(
            {
                "communicators": [
                    {
                        "missing_ranks": [
                            {"rank": 2, "unresponsive": True, "considered_dead": False}
                        ]
                    }
                ]
            }
        )
    )
    assert report.dead_ranks == {2}


def test_interpret_mismatch():
    # Rank 2's op count lags the modal signature -> flagged as mismatched.
    report = _interpret_ras_status(
        json.dumps(
            {
                "communicators": [
                    {
                        "ranks": [
                            {"rank": 0, "collective_counts": {"AllReduce": 100}},
                            {"rank": 1, "collective_counts": {"AllReduce": 100}},
                            {"rank": 2, "collective_counts": {"AllReduce": 98}},
                        ]
                    }
                ]
            }
        )
    )
    assert report.mismatched_ranks == {2}
    # The laggard's count signature is carried for the frozen/advancing check.
    assert report.rank_counts == {2: (("AllReduce", 98),)}
    assert not report.healthy


def test_interpret_aligned_counts_are_healthy():
    # All ranks agree -> no mismatch even with multiple op types.
    report = _interpret_ras_status(
        json.dumps(
            {
                "communicators": [
                    {
                        "ranks": [
                            {
                                "rank": r,
                                "collective_counts": {"AllReduce": 5, "Bcast": 2},
                            }
                            for r in range(4)
                        ]
                    }
                ]
            }
        )
    )
    assert report.healthy


def test_interpret_invalid_json():
    assert _interpret_ras_status("not json") is None
    assert _interpret_ras_status(json.dumps([1, 2, 3])) is None
    # A dict without a communicators array is an unexpected shape -> None.
    assert _interpret_ras_status(json.dumps({"nccl_version": "2.28.7"})) is None


def test_interpret_nccl_2_28_9_missing_comma():
    # NCCL 2.28.9 emits missing_ranks[] with no comma before "status", which is
    # invalid JSON. The parser repairs that pattern and still detects the rank.
    malformed = """
    {
      "nccl_version": "2.28.9",
      "communicators": [
        {
          "missing_ranks": [
            {
              "rank": 1,
              "nvml_dev": 0
              "status": {"unresponsive": true, "considered_dead": false}
            }
          ]
        }
      ]
    }
    """
    # Sanity: it really is invalid JSON without the repair.
    with pytest.raises(json.JSONDecodeError):
        json.loads(malformed)

    report = _interpret_ras_status(malformed)
    assert report is not None
    assert report.dead_ranks == {1}


# --------------------------------------------------------------------------
# Debounce / actions
# --------------------------------------------------------------------------


def _dead(*ranks):
    return RASReport(dead_ranks=set(ranks))


def _mismatch(counts):
    """A mismatch report: {rank: AllReduce count} for the lagging rank(s)."""
    return RASReport(rank_counts={r: (("AllReduce", n),) for r, n in counts.items()})


def _healthy():
    return RASReport()


# -- dead ranks: a hard signal, confirmed from the first poll ----------------


def test_observe_mode_never_raises(monkeypatch, increasing_time):
    reports = [_dead(2)] * 3
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_OBSERVE, confirm_count=3, reports=reports
    )

    for _ in reports:
        callback.after_worker_group_poll_status(MagicMock())  # must not raise

    assert len(captured) == 1  # stacks captured once per episode


def test_fail_mode_raises_after_confirm(monkeypatch, increasing_time):
    reports = [_dead(1), _dead(1)]
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=reports
    )

    # First dead-rank poll: hard 1/2, no raise.
    callback.after_worker_group_poll_status(MagicMock())
    # Second consecutive identical poll: hard 2/2 -> raise.
    with pytest.raises(NCCLHangError) as exc_info:
        callback.after_worker_group_poll_status(MagicMock())

    assert 1 in exc_info.value.worker_failures
    assert len(captured) == 1


def test_changing_signature_resets_debounce(monkeypatch, increasing_time):
    # Different anomaly each round never accumulates to the confirm threshold.
    reports = [_dead(1), _dead(2), _dead(1)]
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=reports
    )

    for _ in reports:
        callback.after_worker_group_poll_status(MagicMock())  # never raises

    assert callback._hard_polls == 1
    assert not captured


def test_healthy_resets_state(monkeypatch, increasing_time):
    reports = [_dead(1), _healthy(), _dead(1)]
    callback, _ = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=reports
    )

    for _ in reports:
        callback.after_worker_group_poll_status(MagicMock())

    # The healthy report in the middle reset the counter back to 1.
    assert callback._hard_polls == 1


def test_throttle_skips_query(monkeypatch):
    # With a large interval and a fixed clock, only the first poll queries.
    monkeypatch.setattr(nccl_ras_module, "time_monotonic", lambda: 100.0)
    reports = [_dead(1)]
    callback, _ = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_OBSERVE, confirm_count=1, reports=reports
    )
    callback._poll_interval_s = 1000.0

    callback.after_worker_group_poll_status(MagicMock())  # queries
    callback.after_worker_group_poll_status(MagicMock())  # throttled, no query

    assert callback._hard_polls == 1  # only one query took effect


def test_degraded_skips(monkeypatch, increasing_time):
    callback, _ = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=1, reports=[_dead(1)]
    )
    callback._degraded = True
    callback.after_worker_group_poll_status(MagicMock())  # must not query/raise
    assert callback._hard_polls == 0


# -- op-count mismatch: hard if frozen, soft if advancing --------------------


def test_frozen_mismatch_fails(monkeypatch, increasing_time):
    # Same laggard with unchanged counts -> hard hang after baseline + confirm.
    reports = [_mismatch({1: 8})] * 3
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=reports
    )

    callback.after_worker_group_poll_status(MagicMock())  # baseline
    callback.after_worker_group_poll_status(MagicMock())  # frozen -> hard 1/2
    with pytest.raises(NCCLHangError) as exc_info:
        callback.after_worker_group_poll_status(MagicMock())  # hard 2/2 -> raise

    assert 1 in exc_info.value.worker_failures
    assert len(captured) == 1


def test_advancing_mismatch_never_fails(monkeypatch, increasing_time):
    # Same laggard but counts keep advancing -> soft hang, never raises.
    reports = [_mismatch({1: n}) for n in (8, 9, 10, 11)]
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=reports
    )

    for _ in reports:
        callback.after_worker_group_poll_status(MagicMock())  # never raises

    assert callback._hard_polls == 0
    assert callback._soft_polls == 3  # baseline, then 3 advancing polls
    assert not captured  # soft hangs do not capture stacks


def test_soft_then_freeze_escalates_to_hard(monkeypatch, increasing_time):
    # Advancing (soft) for a while, then the counts freeze -> escalates to hard.
    reports = [
        _mismatch({1: 8}),
        _mismatch({1: 9}),
        _mismatch({1: 9}),
        _mismatch({1: 9}),
    ]
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=reports
    )

    callback.after_worker_group_poll_status(MagicMock())  # baseline (1:8)
    callback.after_worker_group_poll_status(MagicMock())  # advancing -> soft 1
    callback.after_worker_group_poll_status(MagicMock())  # frozen -> hard 1/2
    with pytest.raises(NCCLHangError):
        callback.after_worker_group_poll_status(MagicMock())  # frozen -> hard 2/2


def test_dead_rank_overrides_advancing_mismatch(monkeypatch, increasing_time):
    # A dead rank alongside an advancing mismatch is still a hard hang.
    r1 = RASReport(dead_ranks={3}, rank_counts={1: (("AllReduce", 8),)})
    r2 = RASReport(dead_ranks={3}, rank_counts={1: (("AllReduce", 9),)})
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=[r1, r2]
    )

    callback.after_worker_group_poll_status(MagicMock())  # dead -> hard 1/2
    with pytest.raises(NCCLHangError) as exc_info:
        callback.after_worker_group_poll_status(MagicMock())  # dead -> hard 2/2

    assert 3 in exc_info.value.worker_failures


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-x", __file__]))
