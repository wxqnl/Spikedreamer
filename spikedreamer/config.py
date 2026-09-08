"""Configuration shared by every paired experiment."""

import argparse
from copy import deepcopy
from types import SimpleNamespace
from ruamel.yaml import YAML


DEFAULTS = dict(
    method="ta", task="walker_walk", seed=0, device="cuda:0",
    frames=1_000_000, action_repeat=2, envs=4, time_limit=1000,
    eval_every=10_000, eval_episodes=10, eval_seed=100_000, log_every=1000,
    prefill=2500, pretrain=100, train_ratio=512,
    batch_size=16, batch_length=64, burn_in=0, replay_capacity=1_000_000,
    hidden=512, deter=512, stoch=32, classes=32, cnn_depth=32,
    io_steps=1, core_steps=8, input_layers=1, output_layers=1,
    persistent=True, reset_dendrites=True, readout="spike",
    adaptive_max_steps=1, entropy_threshold=0.8, spike_reg=0.0,
    threshold=0.5, tau=2.0, tau_basal=2.0, tau_apical=2.0, core_norm_gain=2.0,
    dyn_scale=0.5, rep_scale=0.1, kl_free=1.0, unimix=0.01,
    model_lr=1e-4, actor_lr=3e-5, value_lr=3e-5, opt_eps=1e-8,
    grad_clip=1000.0, imag_horizon=15, discount=0.997,
    lambda_=0.95, actor_entropy=3e-4, slow_fraction=0.02,
    precision="fp32", cpu_threads=4,
    max_updates=0, save_replay=True, resume=False,
)

PRESETS = {
    "legacy": dict(method="legacy", io_steps=8, core_steps=8),
    "ta": dict(method="ta", io_steps=1, adaptive_max_steps=1),
    "ta_core": dict(method="ta", io_steps=8, adaptive_max_steps=1),
    "adaptive": dict(method="ta", io_steps=1, adaptive_max_steps=4),
    "binary": dict(method="binary", io_steps=1, persistent=False),
    "gru": dict(method="gru", io_steps=1),
}


def validate(c):
    if c.method not in ("legacy", "ta", "binary", "gru"):
        raise ValueError(f"Unsupported method: {c.method}")
    for name in ("frames", "action_repeat", "envs", "time_limit", "eval_every",
                 "log_every", "batch_size", "batch_length", "hidden", "deter",
                 "stoch", "classes", "io_steps", "core_steps", "imag_horizon",
                 "adaptive_max_steps", "replay_capacity", "cpu_threads"):
        if getattr(c, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if c.prefill < c.envs or c.prefill * c.action_repeat >= c.frames:
        raise ValueError("prefill counts agent transitions and must fit within frames")
    if c.frames % (c.envs * c.action_repeat):
        raise ValueError("frames must be divisible by envs * action_repeat")
    if c.time_limit % c.action_repeat:
        raise ValueError("time_limit must be divisible by action_repeat")
    if c.batch_length < 2 or c.classes < 2 or c.input_layers < 1 or c.output_layers < 1:
        raise ValueError("Invalid sequence, categorical, or network depth configuration")
    if c.burn_in < 0 or c.spike_reg < 0 or c.train_ratio < 0:
        raise ValueError("burn_in, spike_reg and train_ratio cannot be negative")
    if c.readout not in ("spike", "membrane"):
        raise ValueError("readout must be spike or membrane")
    if c.method == "legacy" and c.io_steps != c.core_steps:
        raise ValueError("Legacy core_steps and io_steps must match")
    if c.adaptive_max_steps > 1 and c.method != "ta":
        raise ValueError("Adaptive refinement requires the MCN core")
    if not 0 <= c.entropy_threshold <= 1:
        raise ValueError("entropy_threshold must be in [0, 1]")
    if min(c.tau, c.tau_basal, c.tau_apical) < 1:
        raise ValueError("Neuron time constants must be >= 1")
    if c.core_norm_gain <= 0 or c.grad_clip <= 0:
        raise ValueError("core_norm_gain and grad_clip must be positive")
    if min(c.eval_episodes, c.pretrain, c.max_updates) < 0:
        raise ValueError("eval_episodes, pretrain and max_updates cannot be negative")
    if c.resume and not c.save_replay:
        raise ValueError("Resume requires save_replay=True")
    if c.precision not in ("fp32", "bf16"):
        raise ValueError("Use fp32 or bf16")
    if c.precision == "bf16" and not c.device.startswith("cuda"):
        raise ValueError("bf16 training currently requires CUDA")
    return c


def load_config(preset="ta", path=None, overrides=()):
    data = deepcopy(DEFAULTS)
    data.update(PRESETS[preset])
    if path:
        with open(path) as f:
            values = YAML(typ="safe").load(f)
        unknown = set(values) - set(data)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        data.update(values)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or key not in data:
            raise ValueError(f"Expected a supported key=value override, got {item}")
        data[key] = YAML(typ="safe").load(value)
    return validate(SimpleNamespace(**data))


def upstream_config(c):
    """Supply upstream network constructor options without changing their shapes."""
    lif = dict(threshold=c.threshold, tau=c.tau, act_fun="QGateGrad")
    norm = dict(threshold=c.threshold, v_reset=0.0, eps=1e-3)
    return SimpleNamespace(
        spike_times=c.io_steps, dyn_cell_p=dict(act="MCNode", norm="PopNorm"),
        LIFNode=lif, PopNorm=norm,
        MCNode=dict(**lif, tau_basal=c.tau_basal, tau_apical=c.tau_apical,
                    v_reset=0.0, comps=["basal", "apical", "soma"]),
    )


def add_config_args(parser: argparse.ArgumentParser):
    parser.add_argument("--preset", choices=PRESETS, default="ta")
    parser.add_argument("--config")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
