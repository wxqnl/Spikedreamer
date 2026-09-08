"""Profile real replay posterior states and complete network updates on CUDA."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .evaluate import load_agent
from .metrics import measure, OperationCounter, energy_proxy
from .replay import Replay
from .rssm import flatten_state


def checkpoint_batch(checkpoint, c, saved):
    replay = Replay(c.replay_capacity, c.seed, Path(checkpoint).parent / "replay")
    # Read-only loading: unlike training resume, do not finalize partial episodes.
    for name in saved["replay"]["names"]:
        with np.load(replay.directory / f"{name}.npz", allow_pickle=False) as ep:
            replay.episodes[name] = {k: ep[k] for k in ep.files}
    replay.active = saved["replay"]["active"]
    return replay.sample(c.batch_size, c.burn_in + c.batch_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--minimum", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parts", nargs="+",
                        choices=["rssm", "imagination", "policy", "world_update",
                                 "behavior_update", "train_update"],
                        default=["rssm", "imagination", "policy", "train_update"])
    parser.add_argument("--idle-watts", type=float,
                        help="Measured idle power on this same otherwise idle GPU")
    parser.add_argument("--mac-pj", type=float)
    parser.add_argument("--ac-pj", type=float)
    parser.add_argument("--neuron-pj", type=float)
    parser.add_argument("--coefficient-source",
                        help="Source and assumptions for the three arithmetic energy coefficients")
    args = parser.parse_args()
    if args.seconds <= 0 or args.minimum < 1:
        parser.error("Measurement duration and minimum iterations must be positive")
    coefficients = [args.mac_pj, args.ac_pj, args.neuron_pj]
    if any(value is not None for value in coefficients):
        if any(value is None for value in coefficients) or not args.coefficient_source:
            parser.error("Supply all three energy coefficients and --coefficient-source")
        if any(not np.isfinite(value) or value < 0 for value in coefficients):
            parser.error("Energy coefficients must be finite and nonnegative")
    agent, c, saved = load_agent(args.checkpoint, args.device)
    if not c.device.startswith("cuda"):
        parser.error("H100/GPU measurements require CUDA")
    agent.load_training_state(saved["agent"])
    torch.cuda.set_device(c.device)
    batch = checkpoint_batch(args.checkpoint, c, saved)
    with torch.no_grad(), agent.autocast():
        data = agent.wm.preprocess(batch)
        post, _ = agent.wm.infer(data)
        start = flatten_state(post)
        action = agent.actor(agent.wm.dynamics.get_feat(start)).mode()
    # Timing passes have no counting hooks or forced host transfers.
    @torch.no_grad()
    def rssm():
        with agent.autocast():
            return agent.wm.dynamics.img_step(start, action, sample=False)
    @torch.no_grad()
    def imagination():
        with agent.autocast():
            return agent.imagine(start, sample=False)
    index = c.burn_in + 1
    single_obs = dict(image=batch["image"][:1, index],
                      is_first=batch["is_first"][:1, index])
    single_state = {k: v[:1, 0] for k, v in post.items()}
    previous_action = data["action"][:1, index]
    def world_update():
        with agent.autocast():
            loss = agent.wm.loss(batch)[0]
        return agent.update_parameters(loss, agent.model_opt, agent.wm.parameters())
    def behavior_update():
        with agent.autocast():
            actor, value, _ = agent.behavior_loss(
                post, data["is_terminal"][:, c.burn_in:])
        agent.update_parameters(actor, agent.actor_opt, agent.actor.parameters())
        agent.update_parameters(value, agent.value_opt, agent.value.parameters())
    functions = dict(rssm=rssm, imagination=imagination,
                     policy=lambda: agent.act(single_obs, single_state, previous_action,
                                               evaluation=True),
                     world_update=world_update, behavior_update=behavior_update,
                     train_update=lambda: agent.train_batch(batch))
    results = dict(method=c.method, config=vars(c), checkpoint_frames=saved["frames"],
                   gpu=torch.cuda.get_device_name(c.device), torch=torch.__version__,
                   batch=c.batch_size, sequence=c.batch_length, horizon=c.imag_horizon,
                   measurements={}, note="Short checkpoint performance is not a control result.")
    # Count the loaded checkpoint, before timing updates mutate in-memory weights.
    with torch.no_grad():
        with OperationCounter(agent) as counter:
            _, states, _, _ = imagination()
        results["imagination_operation_ledger"] = counter.result()
        if "steps" in states:
            results["effective_steps"] = float(states["steps"][1:].mean())
    if args.mac_pj is not None:
        results["imagination_energy_proxy"] = energy_proxy(
            results["imagination_operation_ledger"], *coefficients, args.coefficient_source)
    # Forward measurements precede optimizer measurements, which mutate temporary
    # in-memory parameters. Checkpoint files are never overwritten.
    order = [p for p in functions if p in args.parts]
    for part in order:
        result = measure(functions[part], c.device, args.seconds, args.minimum)
        if args.idle_watts is not None and result["raw_joules_per_call"] is not None:
            result["idle_subtracted_joules_per_call"] = (
                result["raw_joules_per_call"] -
                args.idle_watts * result["wall_mean_ms"] / 1000)
            result["idle_watts"] = args.idle_watts
        results["measurements"][part] = result
        print(json.dumps({part: result}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
