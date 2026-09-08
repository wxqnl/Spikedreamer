"""Opt-in integration tests use real DMC and a real training checkpoint, never mocks."""

import os
from pathlib import Path
import numpy as np
import pytest
import torch

from spikedreamer.config import load_config
from spikedreamer.env import DMC, DMC19
from spikedreamer.evaluate import load_agent
from spikedreamer.model import Agent
from spikedreamer.profile import checkpoint_batch
from spikedreamer.runtime import restore_rng


@pytest.mark.skipif(os.environ.get("SPIKEDREAMER_TEST_DMC") != "1",
                    reason="Set SPIKEDREAMER_TEST_DMC=1 to run actual MuJoCo render checks")
@pytest.mark.parametrize("task", DMC19)
def test_dmc_task_render_action_and_frame_accounting(task):
    c = load_config(overrides=[f"task={task}"])
    env = DMC(c, 123)
    try:
        first = env.reset()
        assert first["is_first"] and not first["is_terminal"]
        obs, reward, done, count = env.step(np.zeros(env.actions, np.float32))
        assert obs["image"].shape == (64, 64, 3) and obs["image"].dtype == np.uint8
        assert count == 2 and not done and np.isfinite(reward)
        assert not obs["is_first"]
    finally:
        env.close()


@pytest.mark.skipif(not os.environ.get("SPIKEDREAMER_TEST_CHECKPOINT"),
                    reason="Supply a locally trusted real training checkpoint")
def test_full_optimizer_checkpoint_continuation():
    path = Path(os.environ["SPIKEDREAMER_TEST_CHECKPOINT"])
    first, c, saved = load_agent(path, "cuda:0")
    batch = checkpoint_batch(path, c, saved)
    first.load_training_state(saved["agent"])
    restore_rng(saved["rng"])
    first_metrics = first.train_batch(batch)
    second = Agent(c, saved["actions"])
    second.load_training_state(saved["agent"])
    restore_rng(saved["rng"])
    second_metrics = second.train_batch(batch)
    assert first.updates == second.updates == saved["agent"]["updates"] + 1
    for name in first_metrics:
        assert np.isfinite(first_metrics[name])
        np.testing.assert_allclose(first_metrics[name], second_metrics[name], rtol=1e-5, atol=1e-6)
    for name, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[name], rtol=1e-5, atol=1e-6, msg=name)
