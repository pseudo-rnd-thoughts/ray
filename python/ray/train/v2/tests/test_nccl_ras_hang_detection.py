"""End-to-end tests for the NCCL RAS hang detector on real GPUs.

Each test deliberately induces one class of NCCL desync inside a real
``TorchTrainer`` (``backend="nccl"``, ``use_gpu=True``) with the
:class:`NCCLRASCallback` registered, and asserts the callback's whole-job
behavior: query RAS on a worker -> parse -> classify frozen (hard) vs advancing
(soft) -> capture stacks + raise :class:`NCCLHangError`. RAS only exists inside
real NCCL, so these require real GPUs and the ``ncclras`` client binary (NCCL
>= 2.28.7); the whole module skips when either is missing.

This consolidates the standalone reproduction scripts in
``scratch/nccl-ras-examples/`` into one runnable suite. The straggler and
multi-communicator cases deadlock regardless of tensor size and are asserted to
raise; the size-dependent mismatch cases (op-count skew, op-type, shape) may
instead complete with garbage on small tensors, so they only assert that the
callback never turns a non-hang into a spurious failure -- run them with
``SIZE=large`` to push them over NCCL's staging buffer and exercise the
hang-detection path. A genuinely undetected hang surfaces as a test timeout.

Run (needs GPUs + ncclras):
    SIZE=large pytest python/ray/train/v2/tests/test_nccl_ras_e2e.py
"""
import os
import shutil

import pytest

# Worker-side NCCL RAS switch + fast detection so a hang is caught in seconds.
# Set before ray.init so spawned workers inherit them.
os.environ.setdefault("NCCL_RAS_ENABLE", "1")
os.environ.setdefault("RAY_TRAIN_NCCL_RAS_POLL_INTERVAL_S", "2")
os.environ.setdefault("RAY_TRAIN_NCCL_RAS_CONFIRM_COUNT", "2")
os.environ.setdefault("RAY_TRAIN_NCCL_RAS_QUERY_TIMEOUT_S", "5")

torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    pytest.skip(
        "NCCL RAS e2e tests require >= 2 visible GPUs.", allow_module_level=True
    )
if shutil.which(os.environ.get("RAY_TRAIN_NCCLRAS_PATH", "ncclras")) is None:
    pytest.skip(
        "`ncclras` client binary not found on PATH (NCCL >= 2.28.7).",
        allow_module_level=True,
    )

import torch.distributed as dist  # noqa: E402

import ray  # noqa: E402
import ray.train  # noqa: E402
from ray.train import RunConfig, ScalingConfig  # noqa: E402
from ray.train.torch import TorchConfig, TorchTrainer, get_device  # noqa: E402
from ray.train.v2._internal.callbacks.nccl_ras import NCCLRASCallback  # noqa: E402
from ray.train.v2.api.exceptions import (  # noqa: E402
    NCCLHangError,
    WorkerGroupError,
)

# Step at which each scenario diverges, and a short loop so the non-hanging
# cases finish quickly. The hanging cases block in NCCL well before STEPS.
HANG_STEP = 3
STEPS = 12
STEP_SLEEP_S = 1.0

# fp32 element counts. ``large`` (256 MiB) mirrors an LLM/NN gradient bucket and
# is far more likely to actually deadlock a mismatch than ``small`` (32 B).
_SIZE_PRESETS = {"small": 8, "large": 64 * 1024 * 1024}


def _numel() -> int:
    override = os.environ.get("TENSOR_ELEMS")
    if override:
        return int(override)
    return _SIZE_PRESETS.get(os.environ.get("SIZE", "small").lower(), 8)


# ---------------------------------------------------------------------------
# Scenario train functions (top-level so they pickle to the workers).
# ---------------------------------------------------------------------------


def _straggler_train(config):
    """A rank stops calling collectives but stays alive -> survivors wedge."""
    import time

    rank = ray.train.get_context().get_world_rank()
    device = get_device()
    numel = config["numel"]
    for step in range(STEPS):
        if step == HANG_STEP and rank == 1:
            while True:
                time.sleep(30)  # alive but never collectives again
        dist.all_reduce(torch.ones(numel, device=device))
        ray.train.report({"step": step})
        time.sleep(STEP_SLEEP_S)


def _op_count_skew_train(config):
    """Rank 0 issues one EXTRA unmatched all_reduce -> op-count skew."""
    import time

    rank = ray.train.get_context().get_world_rank()
    device = get_device()
    numel = config["numel"]
    for step in range(STEPS):
        dist.all_reduce(torch.ones(numel, device=device))
        ray.train.report({"step": step})
        if step == HANG_STEP and rank == 0:
            dist.all_reduce(torch.ones(numel, device=device))  # unmatched
        time.sleep(STEP_SLEEP_S)


def _collective_mismatch_train(config):
    """Ranks call different collectives at the same step -> op-type mismatch."""
    import time

    rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    device = get_device()
    numel = config["numel"]
    for step in range(STEPS):
        if step < HANG_STEP or rank == 0:
            dist.all_reduce(torch.ones(numel, device=device))
        else:
            out = [torch.empty(numel, device=device) for _ in range(world_size)]
            dist.all_gather(out, torch.ones(numel, device=device))
        ray.train.report({"step": step})
        time.sleep(STEP_SLEEP_S)


def _shape_mismatch_train(config):
    """Same collective, different tensor size per rank -> shape mismatch.

    Undetectable by RAS (op type/count are identical), so this only checks the
    callback does not spuriously fail a run.
    """
    import time

    rank = ray.train.get_context().get_world_rank()
    device = get_device()
    numel = config["numel"]
    for step in range(STEPS):
        size = 2 * numel if (step >= HANG_STEP and rank == 0) else numel
        dist.all_reduce(torch.ones(size, device=device))
        ray.train.report({"step": step})
        time.sleep(STEP_SLEEP_S)


def _dead_rank_train(config):
    """A rank hard-exits mid-collective -> dead/unresponsive rank."""
    import time

    rank = ray.train.get_context().get_world_rank()
    device = get_device()
    numel = config["numel"]
    for step in range(STEPS):
        dist.all_reduce(torch.ones(numel, device=device))
        ray.train.report({"step": step})
        if step == HANG_STEP and rank == 1:
            os._exit(1)  # die without finalizing NCCL
        time.sleep(STEP_SLEEP_S)


def _multicomm_subset_train(config):
    """Freeze one of two disjoint sub-communicators while the other advances.

    NOTE: ``ray.train.report`` is intentionally not called -- it synchronizes
    across all workers and would freeze the healthy subgroup too.
    """
    import time

    ctx = ray.train.get_context()
    rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    device = get_device()
    numel = config["numel"]

    half = world_size // 2
    group_a_ranks = list(range(half))
    group_b_ranks = list(range(half, world_size))
    # new_group is collective over the world: every rank must call it.
    group_a = dist.new_group(group_a_ranks)
    group_b = dist.new_group(group_b_ranks)

    in_a = rank in group_a_ranks
    my_group = group_a if in_a else group_b
    straggler = group_b_ranks[-1]

    step = 0
    while True:
        if step == HANG_STEP and rank == straggler:
            while True:
                time.sleep(30)
        dist.all_reduce(torch.ones(numel, device=device), group=my_group)
        step += 1
        time.sleep(STEP_SLEEP_S)


# ---------------------------------------------------------------------------
# Runner + fixtures.
# ---------------------------------------------------------------------------


def _run_scenario(train_func, num_workers):
    """Run ``train_func`` with the RAS callback; return the raised error or None."""
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"numel": _numel()},
        torch_config=TorchConfig(backend="nccl"),
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True),
        run_config=RunConfig(callbacks=[NCCLRASCallback()]),
    )
    try:
        trainer.fit()
        return None
    except BaseException as e:  # noqa: BLE001 - return whatever fit raised
        return e


@pytest.fixture
def ray_start_4_cpus_4_gpus():
    ray.init(num_cpus=4, num_gpus=4)
    yield
    ray.shutdown()


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_straggler_detected(ray_start_4_cpus_2_gpus):
    # Wedges regardless of tensor size -> a hard hang the callback must raise.
    err = _run_scenario(_straggler_train, num_workers=2)
    assert isinstance(err, NCCLHangError), f"expected NCCLHangError, got {err!r}"


def test_dead_rank_fails_run(ray_start_4_cpus_2_gpus):
    # A dead rank fails the run (RAS dead-rank detection or Ray's own health
    # check, whichever fires first) -- in both cases a WorkerGroupError.
    err = _run_scenario(_dead_rank_train, num_workers=2)
    assert isinstance(err, WorkerGroupError), f"expected a failure, got {err!r}"


@pytest.mark.parametrize(
    "train_func",
    [_op_count_skew_train, _collective_mismatch_train, _shape_mismatch_train],
    ids=["op_count_skew", "collective_mismatch", "shape_mismatch"],
)
def test_mismatch_scenarios(train_func, ray_start_4_cpus_2_gpus):
    # Size-dependent: on small tensors these often complete with garbage rather
    # than hang. Assert only that the callback adds no spurious failure -- if it
    # does hang (e.g. SIZE=large), the failure must be a clean NCCLHangError.
    err = _run_scenario(train_func, num_workers=2)
    assert err is None or isinstance(err, NCCLHangError), f"unexpected: {err!r}"


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="multi-communicator case needs >= 4 GPUs"
)
def test_multicomm_subset_detected(ray_start_4_cpus_4_gpus):
    # One frozen subgroup next to one advancing subgroup -> hard hang.
    err = _run_scenario(_multicomm_subset_train, num_workers=4)
    assert isinstance(err, NCCLHangError), f"expected NCCLHangError, got {err!r}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", "-x", __file__]))
