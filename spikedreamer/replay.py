"""Bounded episode replay with independent, checkpointable sampling RNG."""

from collections import OrderedDict
from pathlib import Path
import numpy as np


class Replay:
    def __init__(self, capacity, seed, directory=None):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.episodes = OrderedDict()
        self.active = {}
        self.counter = 0
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def start(self, worker, obs, actions):
        first = {**obs, "action": np.zeros(actions, np.float32), "reward": np.float32(0)}
        self.active[worker] = {k: [v] for k, v in first.items()}

    def add(self, worker, obs, action, reward, done):
        item = {**obs, "action": np.asarray(action, np.float32), "reward": np.float32(reward)}
        for k, v in item.items():
            self.active[worker][k].append(v)
        if done:
            self._finish(worker)

    def _finish(self, worker):
        episode = {k: np.stack(v) for k, v in self.active.pop(worker).items()}
        if len(episode["reward"]) <= 1:
            return
        name = f"episode-{self.counter:08d}"
        self.counter += 1
        self.episodes[name] = episode
        if self.directory:
            path = self.directory / f"{name}.npz"
            # An episode filename is never reused in an existing run.
            with path.open("xb") as f:
                np.savez_compressed(f, **episode)
        self._evict()

    def _evict(self):
        while len(self.episodes) > 1 and self.size > self.capacity:
            name, _ = self.episodes.popitem(last=False)
            # Disk pruning happens only at a checkpoint, so the previous
            # checkpoint's manifest remains loadable after a crash.

    @property
    def size(self):
        return sum(len(e["reward"]) - 1 for e in self.episodes.values())

    def sample(self, batch, length, burn_in=0):
        """Sample an unchanged learning window, optionally preceded by its history.

        Prefix construction consumes no replay RNG. Left padding repeats a real
        observation only for fixed tensor shapes; warmup_valid excludes it from
        state updates, and the entire prefix is excluded from training targets.
        """
        if batch < 1 or length < 1 or burn_in < 0:
            raise ValueError("Invalid replay batch, learning length or burn-in")
        candidates = list(self.episodes.values()) + [
            e for e in self.active.values() if len(e["reward"]) > 1]
        if not candidates:
            raise RuntimeError("Replay needs real environment transitions before sampling")
        sizes = np.asarray([len(e["reward"]) for e in candidates])
        probabilities = (sizes - 1) / (sizes - 1).sum()
        rows = []
        for _ in range(batch):
            pieces, remaining, prefix = [], length, None
            while remaining:
                episode = candidates[self.rng.choice(len(candidates), p=probabilities)]
                index = int(self.rng.integers(len(episode["reward"]) - 1)) if not pieces else 0
                take = min(remaining, len(episode["reward"]) - index)
                if burn_in and not pieces:
                    count = min(burn_in, index)
                    prefix = {}
                    for key, values in episode.items():
                        first = np.array(values[:1], copy=True)
                        # Active trajectories use lists. An empty list loses both
                        # observation shape and dtype; slice a shaped array instead.
                        history = (np.array(values[index - count:index], copy=True)
                                   if count else first[:0])
                        prefix[key] = np.concatenate((
                            np.repeat(first, burn_in - count, axis=0), history), axis=0)
                    if count:
                        # A truncated prefix still needs a defined initial state.
                        prefix["is_first"][burn_in - count] = True
                    valid = np.arange(burn_in) >= burn_in - count
                # Copy before setting sequence-boundary flags: replay is immutable.
                part = {k: np.array(v[index:index + take], copy=True) for k, v in episode.items()}
                if not burn_in or pieces or index == 0:
                    part["is_first"][0] = True
                pieces.append(part)
                remaining -= take
            row = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
            if prefix is not None:
                row = {k: np.concatenate((prefix[k], row[k])) for k in row}
                row["warmup_valid"] = valid
            rows.append(row)
        return {k: np.stack([row[k] for row in rows]) for k in rows[0]}

    def state_dict(self):
        return dict(names=list(self.episodes), counter=self.counter, active=self.active,
                    rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        if not self.directory:
            raise ValueError("Resume requires persistent replay")
        self.episodes.clear()
        for name in state["names"]:
            with np.load(self.directory / f"{name}.npz", allow_pickle=False) as data:
                self.episodes[name] = {k: data[k] for k in data.files}
        # Episodes created after the last checkpoint remain unreferenced; avoid
        # reusing those names, but never sample their future data on resume.
        disk_next = max([int(p.stem.split("-")[-1]) + 1
                         for p in self.directory.glob("episode-*.npz")] + [0])
        self.counter = max(state["counter"], disk_next)
        self.rng.bit_generator.state = state["rng"]
        self.active = state["active"]
        # Resume starts fresh simulator episodes. Retain partial experience as
        # non-terminal truncated trajectories, not as continuations of new resets.
        for worker in list(self.active):
            self._finish(worker)

    def prune_disk(self):
        if self.directory:
            keep = {f"{name}.npz" for name in self.episodes}
            for path in self.directory.glob("episode-*.npz"):
                if path.name not in keep:
                    path.unlink()
