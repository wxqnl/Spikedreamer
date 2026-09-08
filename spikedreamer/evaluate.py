"""Evaluate saved models on real DMC episodes without changing training RNG."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import imageio.v2 as imageio

from .env import DMC, stack_obs
from .model import Agent
from .runtime import rng_state, restore_rng, seed_everything


def evaluate(agent, c, episodes=None, video=None):
    saved = rng_state()
    mode = agent.training
    scores, lengths = [], []
    frames = []
    agent.eval()
    try:
        for index in range(episodes or c.eval_episodes):
            # Common held-out initial conditions for every method and checkpoint.
            env = DMC(c, c.eval_seed + index)
            try:
                obs, state, previous = env.reset(), None, None
                score, length, done = 0.0, 0, False
                while not done:
                    if video and index == 0:
                        frames.append(obs["image"])
                    action, state = agent.act(stack_obs([obs]), state, previous, evaluation=True)
                    previous = action
                    obs, reward, done, count = env.step(action[0].cpu().numpy())
                    score += reward
                    length += count
                scores.append(score)
                lengths.append(length)
            finally:
                env.close()
        if video:
            imageio.mimsave(video, frames, fps=30)
        return dict(eval_return=float(np.mean(scores)),
                    eval_std=float(np.std(scores)), eval_scores=scores,
                    eval_length=float(np.mean(lengths)), eval_episodes=len(scores))
    finally:
        agent.train(mode)
        restore_rng(saved)


def load_agent(checkpoint, device=None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    c = SimpleNamespace(**saved["config"])
    if device:
        c.device = device
    seed_everything(c.seed, c.cpu_threads)
    agent = Agent(c, saved["actions"])
    agent.load_state_dict(saved["agent"]["model"])
    return agent, c, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    agent, c, saved = load_agent(args.checkpoint, args.device)
    result = dict(frames=saved["frames"], **evaluate(agent, c, args.episodes, args.video))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
