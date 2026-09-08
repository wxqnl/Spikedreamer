"""Paired task/seed aggregation; never treat evaluation episodes as seeds."""

import argparse
import json
from pathlib import Path
import numpy as np


def read_runs(paths, frames):
    runs = {}
    for path in paths:
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        config = manifest["config"]
        rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        eligible = [r for r in rows if r["frames"] == frames and "eval_return" in r]
        if not eligible:
            raise ValueError(f"{path}: no evaluation at exactly {frames} frames")
        key = (config["task"], config["seed"])
        if key in runs:
            raise ValueError(f"Duplicate task/seed: {key}")
        evaluations = {}
        for r in rows:
            if "eval_return" in r and r["frames"] <= frames:
                evaluations[r["frames"]] = r["eval_return"]
        x = np.array(sorted(evaluations))
        y = np.array([evaluations[k] for k in x])
        runs[key] = dict(score=eligible[-1]["eval_return"], config=config,
                         # Do not invent a pre-training score to fill missing data.
                         observed_auc=float(np.trapz(y, x) / (x[-1] - x[0]))
                         if len(x) > 1 else None,
                         observed_auc_start=int(x[0]))
    return runs


def compare(base, candidate, samples=5000, seed=0):
    if base.keys() != candidate.keys() or not base:
        raise ValueError("Paired comparisons require the same nonempty task/seed set")
    controls = ("task", "action_repeat", "envs", "time_limit", "batch_size",
                "batch_length", "burn_in", "train_ratio", "prefill", "pretrain",
                "imag_horizon", "eval_seed", "eval_episodes", "precision",
                "hidden", "deter", "stoch", "classes", "cnn_depth",
                "model_lr", "actor_lr", "value_lr", "discount", "lambda_")
    for key in base:
        mismatch = [k for k in controls if base[key]["config"][k] != candidate[key]["config"][k]]
        if mismatch:
            raise ValueError(f"{key}: confounded protocol fields {mismatch}")
    tasks = sorted({key[0] for key in base})
    matrices = [np.array([[base[k]["score"], candidate[k]["score"]]
                         for k in sorted(base) if k[0] == task]) for task in tasks]
    if min(len(m) for m in matrices) < 3:
        raise ValueError("Non-inferiority report requires at least three training seeds per task")
    means = np.stack([m.mean(0) for m in matrices])
    if (means[:, 0] <= 0).any():
        raise ValueError("Relative performance requires positive baseline task means")
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(samples):
        paired = [matrices[i][rng.integers(len(matrices[i]), size=len(matrices[i]))].mean(0)
                  for i in rng.integers(len(matrices), size=len(matrices))]
        avg = np.mean(paired, axis=0)
        if avg[0] <= 0:
            raise ValueError("A bootstrap baseline mean is zero; relative non-inferiority is undefined")
        ratios.append(avg[1] / avg[0])
    interval = np.percentile(ratios, [2.5, 97.5])
    point = means.mean(0)
    return dict(
        tasks=len(tasks), training_runs_per_method=len(base),
        baseline_mean=float(point[0]), candidate_mean=float(point[1]),
        ratio_of_task_mean_scores=float(point[1] / point[0]),
        ratio_95_ci=interval.tolist(), noninferior_5_percent=bool(interval[0] >= 0.95),
        bootstrap="paired hierarchical task/seed percentile bootstrap",
        task_guardrail_violations=int((means[:, 1] < 0.9 * means[:, 0]).sum()),
        task_scores={task: dict(baseline=float(m[0]), candidate=float(m[1]),
                               ratio=float(m[1] / m[0]))
                     for task, m in zip(tasks, means)},
        note="A 5% relative non-inferiority margin is not a fixed 50-point DMC margin.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=1_000_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(read_runs(args.baseline, args.frames), read_runs(args.candidate, args.frames))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
