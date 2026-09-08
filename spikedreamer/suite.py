"""Run independent task/seed experiments, one process per explicitly selected GPU."""

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import PRESETS, load_config
from .env import DEV6, DMC19


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--suite", choices=["dev6", "dmc19"], default="dev6")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=["legacy", "ta"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--gpus", nargs="+", required=True,
                        help="Physical GPU indices; CUDA and EGL use the same selected GPU")
    parser.add_argument("--frames", type=int, default=1_000_000)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print commands without launching or writing files")
    args = parser.parse_args()
    tasks = args.tasks or (DEV6 if args.suite == "dev6" else DMC19)
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("GPU IDs must be unique")
    if not all(gpu.isdecimal() for gpu in args.gpus):
        parser.error("Use numeric physical GPU indices for matched CUDA/EGL selection")
    reserved = {"task", "seed", "frames", "device", "resume"}
    if any(item.split("=", 1)[0] in reserved for item in args.set):
        parser.error("Use dedicated arguments for task/seed/frames; device and resume are managed per job")
    jobs = []
    for preset, task, seed in itertools.product(args.presets, tasks, args.seeds):
        overrides = [*args.set, f"task={task}", f"seed={seed}",
                     f"frames={args.frames}", "device=cuda:0"]
        c = load_config(preset, overrides=overrides)
        path = Path(args.root) / preset / task / f"seed-{seed}"
        if path.exists() and any(path.iterdir()) and not args.list:
            if not args.resume:
                raise FileExistsError(f"{path} exists; use --resume to continue")
            manifest = json.loads((path / "manifest.json").read_text())["config"]
            mutable = {"frames", "eval_every", "eval_episodes", "log_every",
                       "max_updates", "resume", "device", "compile", "cpu_threads"}
            differences = [k for k, v in manifest.items()
                           if k not in mutable and v != getattr(c, k)]
            if differences:
                raise ValueError(f"{path}: incompatible resume fields {differences}")
            with (path / "metrics.jsonl").open() as source:
                rows = [json.loads(line) for line in source]
            if any(r.get("status") == "completed" and r["frames"] == c.frames for r in rows):
                print(json.dumps(dict(event="skip_completed", run=str(path))), flush=True)
                continue
            overrides.append("resume=True")
        command = [sys.executable, "-m", "spikedreamer.train", "--preset", preset,
                   "--logdir", str(path), "--set", *overrides]
        jobs.append((f"{preset}-{task}-{seed}", command))
    if len({name for name, _ in jobs}) != len(jobs):
        parser.error("Duplicate task/preset/seed combinations")
    if args.list:
        for name, command in jobs:
            print(json.dumps(dict(run=name, argv=command)))
        return
    logs = Path(args.root) / "_workers"
    logs.mkdir(parents=True, exist_ok=True)
    active, queue = {}, iter(jobs)
    try:
        pending = True
        while pending or active:
            for gpu in args.gpus:
                if gpu in active or not pending:
                    continue
                job = next(queue, None)
                if job is None:
                    pending = False
                    break
                name, command = job
                stream = (logs / f"{name}.log").open("a")
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                                   MUJOCO_EGL_DEVICE_ID=gpu)
                try:
                    process = subprocess.Popen(command, env=environment, stdout=stream,
                                               stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                active[gpu] = process, stream, name
                print(json.dumps(dict(event="launched", run=name, gpu=gpu)), flush=True)
            for gpu, (process, stream, name) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[gpu]
                print(json.dumps(dict(event="finished", run=name, returncode=code)), flush=True)
                if code:
                    raise RuntimeError(f"{name} failed; inspect {logs / (name + '.log')}")
            if active:
                time.sleep(0.5)
    finally:
        # Only terminate child experiments started by this invocation.
        for process, stream, _ in active.values():
            process.terminate()
        for process, stream, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()


if __name__ == "__main__":
    main()
