"""Real dm_control environments with explicit incoming-action transitions."""

import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

DMC19 = [
    "acrobot_swingup", "cartpole_balance", "cartpole_balance_sparse",
    "cartpole_swingup", "cartpole_swingup_sparse", "cheetah_run", "cup_catch",
    "finger_spin", "finger_turn_easy", "finger_turn_hard", "hopper_hop",
    "hopper_stand", "pendulum_swingup", "quadruped_run", "quadruped_walk",
    "reacher_easy", "walker_run", "walker_stand", "walker_walk",
]
DEV6 = ["cartpole_balance", "cartpole_swingup_sparse", "cup_catch",
        "finger_spin", "walker_walk", "cheetah_run"]


class DMC:
    def __init__(self, c, seed):
        from dm_control import suite
        task = c.task.removeprefix("dmc_")
        domain, name = task.split("_", 1)
        domain = "ball_in_cup" if domain == "cup" else domain
        self.env = suite.load(domain, name, task_kwargs={"random": seed})
        self.repeat, self.limit = c.action_repeat, c.time_limit
        self.camera = 2 if domain == "quadruped" else 0
        spec = self.env.action_spec()
        self.low, self.high = spec.minimum, spec.maximum
        self.actions = int(np.prod(spec.shape))
        self.elapsed = 0

    def observation(self, time_step, first=False):
        return dict(image=self.env.physics.render(64, 64, camera_id=self.camera),
                    is_first=np.bool_(first),
                    is_terminal=np.bool_(time_step.last() and time_step.discount == 0))

    def reset(self):
        self.elapsed = 0
        return self.observation(self.env.reset(), first=True)

    def step(self, action):
        if not np.isfinite(action).all():
            raise FloatingPointError("Non-finite environment action")
        action = np.clip(action, -1, 1)
        physical = self.low + (action + 1) * 0.5 * (self.high - self.low)
        reward, frames = 0.0, 0
        for _ in range(self.repeat):
            time_step = self.env.step(physical)
            reward += float(time_step.reward or 0)
            frames += 1
            self.elapsed += 1
            if time_step.last() or self.elapsed >= self.limit:
                break
        done = time_step.last() or self.elapsed >= self.limit
        return self.observation(time_step), reward, done, frames

    def close(self):
        self.env.close()


def stack_obs(observations):
    return {k: np.stack([o[k] for o in observations]) for k in observations[0]}
