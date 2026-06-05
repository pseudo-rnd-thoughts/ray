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
        json.dumps({"deadRanks": [], "communicators": [{"nRanks": 4}]})
    )
    assert report is not None
    assert report.healthy
    assert report.bad_signature == frozenset()


def test_interpret_dead_ranks():
    report = _interpret_ras_status(json.dumps({"deadRanks": [3, 5]}))
    assert report.dead_ranks == {3, 5}
    assert not report.healthy


def test_interpret_mismatch():
    report = _interpret_ras_status(
        json.dumps(
            {
                "communicators": [
                    {"collMismatch": True, "mismatchedRanks": [1, 2]},
                    {"collMismatch": False, "mismatchedRanks": []},
                ]
            }
        )
    )
    assert report.mismatched_ranks == {1, 2}
    assert not report.healthy


def test_interpret_alternate_keys():
    report = _interpret_ras_status(json.dumps({"dead_ranks": [7]}))
    assert report.dead_ranks == {7}


def test_interpret_invalid_json():
    assert _interpret_ras_status("not json") is None
    assert _interpret_ras_status(json.dumps([1, 2, 3])) is None


# --------------------------------------------------------------------------
# Debounce / actions
# --------------------------------------------------------------------------


def _bad(dead=(), mismatched=()):
    return RASReport(dead_ranks=set(dead), mismatched_ranks=set(mismatched))


def _healthy():
    return RASReport()


def test_observe_mode_never_raises(monkeypatch, increasing_time):
    reports = [_bad(dead=[2])] * 3
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_OBSERVE, confirm_count=3, reports=reports
    )

    for _ in reports:
        callback.after_worker_group_poll_status(MagicMock())  # must not raise

    assert len(captured) == 1  # stacks captured once per episode


def test_fail_mode_raises_after_confirm(monkeypatch, increasing_time):
    reports = [_bad(dead=[1]), _bad(dead=[1])]
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=reports
    )

    # First report: below confirm threshold, no raise.
    callback.after_worker_group_poll_status(MagicMock())
    # Second consecutive identical report: confirmed -> raise.
    with pytest.raises(NCCLHangError) as exc_info:
        callback.after_worker_group_poll_status(MagicMock())

    assert 1 in exc_info.value.worker_failures
    assert len(captured) == 1


def test_changing_signature_resets_debounce(monkeypatch, increasing_time):
    # Different anomaly each round never accumulates to the confirm threshold.
    reports = [_bad(dead=[1]), _bad(dead=[2]), _bad(dead=[1])]
    callback, captured = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=reports
    )

    for _ in reports:
        callback.after_worker_group_poll_status(MagicMock())  # never raises

    assert callback._consecutive_bad == 1
    assert not captured


def test_healthy_resets_state(monkeypatch, increasing_time):
    reports = [_bad(dead=[1]), _healthy(), _bad(dead=[1])]
    callback, _ = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=2, reports=reports
    )

    for _ in reports:
        callback.after_worker_group_poll_status(MagicMock())

    # The healthy report in the middle reset the counter back to 1.
    assert callback._consecutive_bad == 1


def test_throttle_skips_query(monkeypatch):
    # With a large interval and a fixed clock, only the first poll queries.
    monkeypatch.setattr(nccl_ras_module, "time_monotonic", lambda: 100.0)
    reports = [_bad(dead=[1])]
    callback, _ = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_OBSERVE, confirm_count=1, reports=reports
    )
    callback._poll_interval_s = 1000.0

    callback.after_worker_group_poll_status(MagicMock())  # queries
    callback.after_worker_group_poll_status(MagicMock())  # throttled, no query

    assert callback._consecutive_bad == 1  # only one query took effect


def test_degraded_skips(monkeypatch, increasing_time):
    callback, _ = _make_callback(
        monkeypatch, NCCL_RAS_ACTION_FAIL, confirm_count=1, reports=[_bad(dead=[1])]
    )
    callback._degraded = True
    callback.after_worker_group_poll_status(MagicMock())  # must not query/raise
    assert callback._consecutive_bad == 0


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-x", __file__]))
