from copy import deepcopy
import numpy as np
import pytest
import torch
from spikedreamer.config import load_config
from spikedreamer.rssm import TimeAlignedRSSM, LegacyRSSM
from spikedreamer.model import lambda_returns, clip_gradients
from spikedreamer.replay import Replay


def core(preset="ta", *overrides):
    torch.manual_seed(9)
    c = load_config(preset, overrides=["device=cpu", *overrides])
    return TimeAlignedRSSM(c, 3, 64)


def equal(a, b):
    for key in a:
        torch.testing.assert_close(a[key], b[key], msg=key)


def test_episode_reset_and_branch_isolation():
    m = core()
    with torch.no_grad():
        initial = m.initial(2)
        state = m.img_step(initial, torch.ones(2, 3), sample=False)
        original = {k: v.clone() for k, v in state.items()}
        embed = torch.randn(1, 2, 64)
        post, _ = m.obs_step(state, torch.randn(2, 3), embed, torch.tensor([True, False]), False)
        fresh, _ = m.obs_step(m.initial(1), torch.zeros(1, 3), embed[:, :1],
                             torch.ones(1, dtype=torch.bool), False)
        equal({k: v[:1] for k, v in post.items()}, fresh)
        equal(state, original)
        branch = m.img_step(state, torch.zeros(2, 3), False)
        m.img_step(state, torch.ones(2, 3), False)
        repeat = m.img_step(state, torch.zeros(2, 3), False)
        equal(branch, repeat)
        equal(state, original)


def test_state_carries_neuron_memory_and_binary_communication():
    m = core()
    state = m.initial(2)
    action = torch.randn(2, 3)
    prior = m.img_step(state, action, sample=False)
    prior["logit"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.input.parameters())
    torch.testing.assert_close(prior["stoch"].sum(-1), torch.ones(2, 32))
    assert ((prior["stoch"] == 0) | (prior["stoch"] == 1)).all()
    assert ((prior["deter"] == 0) | (prior["deter"] == 1)).all()
    assert "input_mem" in prior and "soma" in prior and "prior_mem" in prior
    modified = {**prior, "soma": prior["soma"] + 0.2}
    with torch.no_grad():
        a = m.img_step(prior, action, False)
        b = m.img_step(modified, action, False)
    assert not torch.equal(a["soma"], b["soma"])


def test_adaptive_exit_compaction_and_gradients():
    fixed = core()
    early = core("adaptive", "entropy_threshold=1.0")
    all_steps = core("adaptive", "entropy_threshold=0.0")
    early.load_state_dict(fixed.state_dict())
    all_steps.load_state_dict(fixed.state_dict())
    state, action = fixed.initial(4), torch.randn(4, 3)
    with torch.no_grad():
        equal(fixed.img_step(state, action, False), early.img_step(state, action, False))
    full = all_steps.img_step(state, action, False)
    assert (full["steps"] == 4).all()
    full["logit"].square().mean().backward()
    assert torch.isfinite(all_steps.cell.basal_w.weight.grad).all()
    # Batched compact execution must equal independent trajectory execution.
    with torch.no_grad():
        reference = [
            all_steps.img_step({k: v[i:i+1] for k, v in state.items()}, action[i:i+1], False)
            for i in range(4)]
        equal(full, {k: torch.cat([s[k] for s in reference]) for k in full})


def test_observation_scan_equals_stepwise_and_prefix_is_causal():
    m = core()
    action, embed = torch.randn(2, 4, 3), torch.randn(1, 2, 4, 64)
    first = torch.tensor([[True, False, True, False], [True, False, False, False]])
    with torch.no_grad():
        seq, _ = m.observe(embed, action, first, sample=False)
        state = m.initial(2)
        for h in range(4):
            state, _ = m.obs_step(state, action[:, h], embed[:, :, h], first[:, h], False)
            equal({k: v[:, h] for k, v in seq.items()}, state)
        changed = embed.clone()
        changed[:, :, -1] += 5
        other, _ = m.observe(changed, action, first, sample=False)
        equal({k: v[:, :-1] for k, v in seq.items()},
              {k: v[:, :-1] for k, v in other.items()})


def test_adaptive_mixed_exit_matches_individual_trajectories():
    m = core("adaptive")
    state, action = m.initial(8), torch.randn(8, 3)
    state["stoch"] = torch.nn.functional.one_hot(torch.randint(32, (8, 32)), 32).float()
    with torch.no_grad():
        m.c.adaptive_max_steps = 1
        first = m.img_step(state, action, False)
        entropy = m.get_dist(first).entropy() / (32 * np.log(32))
        m.c.entropy_threshold = float(entropy.median())
        m.c.adaptive_max_steps = 4
        combined = m.img_step(state, action, False)
        assert (combined["steps"] == 1).any() and (combined["steps"] > 1).any()
        separate = [m.img_step({k: v[i:i+1] for k, v in state.items()},
                               action[i:i+1], False) for i in range(8)]
        equal(combined, {k: torch.cat([s[k] for s in separate]) for k in combined})


def test_legacy_adapter_matches_released_core():
    torch.manual_seed(1)
    c = load_config("legacy", overrides=["device=cpu"])
    m = LegacyRSSM(c, 3, 64)
    reference = deepcopy(m.raw)
    action, embed = torch.randn(2, 3), torch.randn(8, 2, 64)
    with torch.no_grad():
        state = m.initial(2)
        expected, _ = reference.obs_step(
            {k: v.clone() for k, v in m.internal(state).items()},
            action.clone(), embed, torch.zeros(2), False)
        actual, _ = m.obs_step(state, action, embed, torch.zeros(2, dtype=torch.bool), False)
        equal(actual, m.external(expected))
        expected, _ = reference.obs_step(reference.initial(2), action.clone(), embed,
                                         torch.ones(2), False)
        actual, _ = m.obs_step(state, action, embed, torch.ones(2, dtype=torch.bool), False)
        equal(actual, m.external(expected))


def test_lambda_return_alignment():
    reward = torch.tensor([1., 2., 3.]).reshape(3, 1, 1)
    discount = torch.tensor([0.5, 0., 0.5]).reshape(3, 1, 1)
    values = torch.tensor([10., 20., 30.]).reshape(3, 1, 1)
    torch.testing.assert_close(lambda_returns(reward, discount, values, 0).flatten(),
                               torch.tensor([6., 2., 18.]))
    torch.testing.assert_close(lambda_returns(reward, discount, values, 1).flatten(),
                               torch.tensor([2., 2., 18.]))


def test_large_finite_gradient_clipping_and_nonfinite_failure():
    p = torch.nn.Parameter(torch.ones(2))
    p.grad = torch.tensor([3e23, 4e23])
    norm = clip_gradients([p], 1000)
    torch.testing.assert_close(norm, torch.tensor(5e23, dtype=torch.float64))
    torch.testing.assert_close(p.grad, torch.tensor([600., 800.]))
    p.grad[0] = float("inf")
    with pytest.raises(FloatingPointError):
        clip_gradients([p], 1000)


def test_replay_sample_is_not_a_view_and_rng_restores(tmp_path):
    replay = Replay(100, 3, tmp_path)
    obs = dict(image=np.zeros((64, 64, 3), np.uint8), is_first=np.bool_(True),
               is_terminal=np.bool_(False))
    replay.start(0, obs, 3)
    for i in range(5):
        replay.add(0, {**obs, "is_first": np.bool_(False)}, np.ones(3), i, i == 4)
    before = {k: v.copy() for k, v in next(iter(replay.episodes.values())).items()}
    saved = replay.state_dict()
    expected = replay.sample(2, 8)
    restored = Replay(100, 7, tmp_path)
    restored.load_state_dict(saved)
    actual = restored.sample(2, 8)
    for k in before:
        np.testing.assert_array_equal(before[k], next(iter(replay.episodes.values()))[k])
        np.testing.assert_array_equal(expected[k], actual[k])
