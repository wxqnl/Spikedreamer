"""Run manifests, atomic checkpoints, and JSONL/TensorBoard metrics."""

import json
import os
import platform
import random
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_manifest(path, c, parameters):
    payload = dict(config=vars(c), parameters=parameters, python=platform.python_version(),
                   packages={k: version(k) for k in
                             ("torch", "numpy", "dm-control", "mujoco", "einops")},
                   cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(c.device) if c.device.startswith("cuda") else "cpu",
                   created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


class Logger:
    def __init__(self, directory):
        self.path = Path(directory) / "metrics.jsonl"
        self.writer = SummaryWriter(str(directory))

    def write(self, frames, **metrics):
        record = dict(frames=int(frames), wall_time=time.time(), **metrics)
        with self.path.open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, frames)
        self.writer.flush()
        print(json.dumps(record, allow_nan=False), flush=True)

    def close(self):
        self.writer.close()
