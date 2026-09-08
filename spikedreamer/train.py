"""Online DMC training with exact frame accounting and resumable learning state."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from .config import load_config, add_config_args
from .env import DMC, stack_obs
from .evaluate import evaluate
from .model import Agent
from .replay import Replay
from .runtime import Logger, seed_everything, rng_state, restore_rng, atomic_save, write_manifest


def run(c, directory):
    directory = Path(directory)
    if c.resume and not (directory / "latest.pt").is_file():
        raise FileNotFoundError(f"No checkpoint in {directory}")
    if not c.resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {directory}; use resume=True or a new path")
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(c.seed, c.cpu_threads)
    envs, logger = [], None
    started = time.monotonic()
    try:
        for i in range(c.envs):
            envs.append(DMC(c, c.seed * 1000 + i))
        agent = Agent(c, envs[0].actions)
        replay = Replay(c.replay_capacity, c.seed, directory / "replay" if c.save_replay else None)
        frames, decisions, update_credit, completed = 0, 0, 0.0, 0
        episode_returns = np.zeros(c.envs)
        rng = np.random.default_rng(c.seed + 17)
        if c.resume:
            saved = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
            mutable = {"frames", "eval_every", "eval_episodes", "log_every",
                       "max_updates", "resume", "device", "compile", "cpu_threads"}
            differences = [k for k in saved["config"]
                           if k not in mutable and saved["config"][k] != getattr(c, k)]
            if differences:
                raise ValueError(f"Resume changes training semantics: {differences}")
            agent.load_training_state(saved["agent"])
            replay.load_state_dict(saved["replay"])
            frames, decisions = saved["frames"], saved["decisions"]
            update_credit, completed = saved["update_credit"], saved["completed_episodes"]
            rng.bit_generator.state = saved["environment_rng"]
            restore_rng(saved["rng"])
        write_manifest(directory / ("resume_manifest.json" if c.resume else "manifest.json"),
                       c, sum(p.numel() for p in agent.parameters()))
        logger = Logger(directory)
        obs = [e.reset() for e in envs]
        for i, o in enumerate(obs):
            replay.start(i, o, agent.actions)
        state, previous = None, None
        next_eval = ((frames // c.eval_every) + 1) * c.eval_every
        next_log = ((frames // c.log_every) + 1) * c.log_every
        training_started = agent.updates > 0
        origin_frames, origin_updates = frames, agent.updates
        latest_metrics = {}
        last_eval_frames = None

        def save(status):
            atomic_save(dict(
                format_version=1, config=vars(c), actions=agent.actions,
                agent=agent.training_state(), replay=replay.state_dict(),
                frames=frames, decisions=decisions, update_credit=update_credit,
                completed_episodes=completed, environment_rng=rng.bit_generator.state,
                rng=rng_state(), status=status), directory / "latest.pt")
            replay.prune_disk()

        def update():
            nonlocal latest_metrics
            batch = replay.sample(c.batch_size, c.burn_in + c.batch_length)
            latest_metrics = agent.train_batch(batch)

        logger.write(frames, event="started", method=c.method,
                     parameters=sum(p.numel() for p in agent.parameters()), updates=agent.updates)
        while frames < c.frames:
            if c.max_updates and agent.updates >= c.max_updates:
                break
            if decisions < c.prefill:
                actions = rng.uniform(-1, 1, (c.envs, agent.actions)).astype(np.float32)
                # The learned recurrent state has not consumed prefill observations.
                state, previous = None, None
            else:
                if not training_started:
                    for _ in range(c.pretrain):
                        if c.max_updates and agent.updates >= c.max_updates:
                            break
                        update()
                    training_started = True
                    if c.max_updates and agent.updates >= c.max_updates:
                        break
                action_tensor, state = agent.act(stack_obs(obs), state, previous)
                previous = action_tensor
                actions = action_tensor.cpu().numpy()
            for i, env in enumerate(envs):
                if frames >= c.frames:
                    break
                observation, reward, done, count = env.step(actions[i])
                replay.add(i, observation, actions[i], reward, done)
                frames += count
                decisions += 1
                episode_returns[i] += reward
                obs[i] = observation
                if training_started:
                    update_credit += c.train_ratio / (c.batch_size * c.batch_length)
                if done:
                    completed += 1
                    logger.write(frames, train_return=float(episode_returns[i]),
                                 completed_episodes=completed, replay_transitions=replay.size)
                    episode_returns[i] = 0
                    obs[i] = env.reset()
                    replay.start(i, obs[i], agent.actions)
            while update_credit >= 1:
                if c.max_updates and agent.updates >= c.max_updates:
                    break
                update()
                update_credit -= 1
            if frames >= next_log:
                seconds = time.monotonic() - started
                logger.write(frames, **latest_metrics, updates=agent.updates,
                             elapsed_seconds=seconds,
                             frames_per_second=(frames - origin_frames) / seconds,
                             updates_per_second=(agent.updates - origin_updates) / seconds)
                next_log = ((frames // c.log_every) + 1) * c.log_every
            if frames >= next_eval:
                if c.eval_episodes:
                    logger.write(frames, **evaluate(agent, c), updates=agent.updates)
                    last_eval_frames = frames
                save("running")
                next_eval = ((frames // c.eval_every) + 1) * c.eval_every
        status = "completed" if frames >= c.frames else "update_limit"
        if c.eval_episodes and last_eval_frames != frames:
            logger.write(frames, **evaluate(agent, c), updates=agent.updates)
        save(status)
        result = dict(status=status, frames=frames, updates=agent.updates,
                      elapsed_seconds=time.monotonic() - started,
                      checkpoint=str(directory / "latest.pt"), **latest_metrics)
        logger.write(frames, event="finished", **{k: v for k, v in result.items() if k != "frames"})
        return result
    finally:
        if logger:
            logger.close()
        for env in envs:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    add_config_args(parser)
    args = parser.parse_args()
    result = run(load_config(args.preset, args.config, args.set), args.logdir)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
