"""Parallel CPU evaluation with independent, unchanged B1 policy trajectories."""

from .env import DMC, stack_obs
from .runtime import rng_state, restore_rng, seed_everything


def evaluate_parallel(agent, c, episodes, frames, video):
    """Overlap CPU rendering while keeping B1 policy calls and per-episode RNG."""
    scores, lengths = [None] * episodes, [None] * episodes
    active = []
    next_index = 0

    def start(index):
        # Match the serial evaluator's initialization order and RNG state.
        seed_everything(c.eval_seed + index, c.cpu_threads)
        env = DMC(c, c.eval_seed + index)
        try:
            obs = env.reset()
            return dict(index=index, env=env, obs=obs, state=None, previous=None,
                        score=0.0, length=0, rng=rng_state())
        except BaseException:
            env.close()
            raise

    try:
        while active or next_index < episodes:
            while len(active) < min(c.envs, episodes) and next_index < episodes:
                active.append(start(next_index))
                next_index += 1
            for episode in active:
                # Never combine posterior sampling across episodes into a new
                # batched RNG stream. Each B1 trajectory retains its old draws.
                restore_rng(episode["rng"])
                if video and episode["index"] == 0:
                    frames.append(episode["obs"]["image"])
                action, state = agent.act(
                    stack_obs([episode["obs"]]), episode["state"],
                    episode["previous"], evaluation=True)
                episode.update(state=state, previous=action, rng=rng_state())
                episode["env"].step_async(action[0].cpu().numpy())
            remaining = []
            for episode in active:
                obs, reward, done, count = episode["env"].step_wait()
                episode.update(obs=obs, score=episode["score"] + reward,
                               length=episode["length"] + count)
                if done:
                    scores[episode["index"]] = episode["score"]
                    lengths[episode["index"]] = episode["length"]
                    episode["env"].close()
                else:
                    remaining.append(episode)
            active = remaining
    finally:
        for episode in active:
            episode["env"].close()
    return scores, lengths
