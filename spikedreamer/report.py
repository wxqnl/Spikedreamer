"""Export the archived Walker study without loading checkpoints or starting runs.

The JSON contains all complete evaluation points, not a selected best score.
Only an explicit allowlist is exported: no credentials, user messages or PIDs.
"""
import argparse
import html
import json
import math
from pathlib import Path
import statistics

ARCHIVES = (
    "spikedreamer-fixed-t8-walker-20260910",
    "spikedreamer-walker-gru-lif-20260911",
    "spikedreamer-stateful-gatedmem-walker-20260912",
)
STOPPED = (
    ("spikedreamer-stateful-context32-walker-20260912-rerun1", "stateful_context32"),
    ("spikedreamer-stateful-slowmem-walker-20260912", "stateful_slowmem32"),
)
LABELS = {
    "stateful_gatedmem32": "Gated Memory-T8",
    "ann_gru": "ANN-GRU",
    "stateful_t8": "Stateful-T8",
    "lif_t8": "LIF-T8",
    "legacy": "Legacy-T8",
}
EVAL_KEYS = ("frames", "updates", "eval_return", "eval_std",
             "eval_scores", "eval_episodes", "eval_length")
BEGIN, END = "<!-- walker-results:start -->", "<!-- walker-results:end -->"


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"{label}: {actual} != {expected}")


def evaluation(row):
    result = {key: row[key] for key in EVAL_KEYS}
    scores = result["eval_scores"]
    if len(scores) != 10 or result["eval_episodes"] != 10 or result["eval_length"] != 1000:
        raise ValueError("Expected ten complete 1,000-frame evaluation episodes")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Non-finite evaluation score")
    close(statistics.mean(scores), result["eval_return"], "evaluation mean")
    close(statistics.pstdev(scores), result["eval_std"], "episode population SD")
    return result


def curve(path):
    retained = {}
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            # A resumed run owns only the prefix retained in its checkpoint.
            if row.get("event") == "started":
                retained = {frame: value for frame, value in retained.items()
                            if frame <= row["frames"]}
            if "eval_return" in row:
                value = evaluation(row)
                retained[value["frames"]] = value
    return [retained[frame] for frame in sorted(retained)]


def collect(root):
    runs = {}
    for directory in ARCHIVES:
        report = json.loads((root / directory / "FINAL_RESULTS.json").read_text())
        if report["status"] != "completed":
            raise ValueError(f"Archive not completed: {directory}")
        for run in report["runs"]:
            preset = run["preset"]
            if preset not in LABELS or preset in runs:
                raise ValueError(f"Unexpected or duplicate preset: {preset}")
            if (run["frames"], run["updates"], run["process_exit_code"]) != (1_000_000, 71_171, 0):
                raise ValueError(f"Incomplete endpoint: {preset}")
            relative = Path(directory) / "runs" / preset / "walker_walk" / "seed-0"
            points = curve(root / relative / "metrics.jsonl")
            if [point["frames"] for point in points] != list(range(10_000, 1_000_001, 10_000)):
                raise ValueError(f"Missing exact evaluation frame: {preset}")
            if points[-1] != evaluation(run["final_evaluation"]):
                raise ValueError(f"Final archive and raw metrics differ: {preset}")
            late_mean = statistics.mean(point["eval_return"] for point in points[-10:])
            close(late_mean, run["final_10_evaluations_mean"], "last-ten mean")
            config = run["config"]
            if config["task"] != "walker_walk" or config["seed"] != 0:
                raise ValueError("This export is restricted to Walker seed 0")
            manifest = run["original_manifest"]
            monitor = run["final_monitor_state"]
            runs[preset] = dict(
                preset=preset, label=LABELS[preset], status="completed",
                frames=run["frames"], updates=run["updates"],
                parameters=run["parameters"], trainable_parameters=run["trainable_parameters"],
                frozen_target_parameters=run["frozen_target_parameters"],
                config=config, evaluations=points, final_evaluation=points[-1],
                final_10_evaluations_mean=late_mean,
                process_active_hours=run["timing"]["observed_process_active_seconds"] / 3600,
                end_to_end_hours=run["timing"]["end_to_end_seconds"] / 3600,
                process_exit_code=0, failure_count=monitor["failure_count"],
                automatic_restarts=monitor["auto_restarts"],
                user_resumes=monitor.get("user_resumes", 0),
                runtime={key: manifest[key] for key in ("python", "packages", "cuda", "gpu")},
                source_archive=str(Path(directory) / "FINAL_RESULTS.json"),
                source_metrics=str(relative / "metrics.jsonl"),
            )
    if set(runs) != set(LABELS):
        raise ValueError("Expected exactly five completed methods")
    for index in range(100):
        if len({run["evaluations"][index]["updates"] for run in runs.values()}) != 1:
            raise ValueError("Methods do not have matching update counts")
    stopped = []
    for directory, preset in STOPPED:
        record = json.loads((root / directory / "STOPPED_BY_USER.json").read_text())
        if record["status"] != "stopped_by_user":
            raise ValueError("Stopped exploration must not be relabelled completed")
        relative = Path(directory) / "runs" / preset / "walker_walk" / "seed-0"
        point = curve(root / relative / "metrics.jsonl")[-1]
        for key in ("frames", "updates", "eval_return", "eval_std"):
            close(point[key], record["latest_evaluation"][key], key)
        stopped.append(dict(
            preset=preset, status="stopped_by_user",
            last_logged_frames=record["last_logged_frames"],
            last_logged_updates=record["last_logged_updates"],
            checkpoint={key: record["checkpoint"][key] for key in
                        ("frames", "updates", "status", "format_version")},
            latest_evaluation=point, process_exit_code=record["process_exit_code"],
            source_record=str(Path(directory) / "STOPPED_BY_USER.json"),
            source_metrics=str(relative / "metrics.jsonl"),
        ))
    return dict(
        schema_version=1, study="walker-fixed-t8-gated-memory-seed0",
        evidence_cutoff="2026-09-14",
        verification_scope="Export checks raw evaluation scores against completed endpoint archives; "
                           "it does not rerun checkpoint, optimizer or replay validation.",
        statistic="Mean and population SD of ten evaluation episodes; one training seed.",
        timing_note="Observed process time includes compilation, sampling, training, evaluation, I/O "
                    "and discarded tails. End-to-end time also includes pauses. Not pure GPU time.",
        limitations=[
            "One task and one development seed; not a multi-seed significance or non-inferiority test.",
            "Gated Memory has burn-in32 and more parameters; original four methods have burn-in0.",
            "ANN-GRU uses ANN peripheral networks; it is not an isolated recurrent-core control.",
            "No matched-compute, energy-efficiency, hardware-sparsity or generalization claim.",
            "Stopped explorations are incomplete and do not establish their final performance.",
        ],
        completed=[runs[preset] for preset in LABELS], stopped=stopped,
    )


def table(data):
    lines = [
        "| 方法 | 1M 回报（均值 ± 回合标准差） | 末十次评估均值 | 总参数 | 累计进程时长 / h |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in data["completed"]:
        score = run["final_evaluation"]
        lines.append(f'| {run["label"]} | {score["eval_return"]:.2f} ± {score["eval_std"]:.2f} '
                     f'| {run["final_10_evaluations_mean"]:.2f} | {run["parameters"]:,} '
                     f'| {run["process_active_hours"]:.2f} |')
    return "\n".join(lines)


def learning_curve_svg(data):
    """A dependency-free vector figure; no smoothing, resampling or fake seeds."""
    width, height = 1000, 620
    left, top, plot_width, plot_height = 90, 100, 690, 410
    x = lambda frame: left + frame / 1_000_000 * plot_width
    y = lambda score: top + plot_height * (1 - score / 1100)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Walker Walk: complete learning curves, training seed 0</title>',
        '<desc id="desc">Five methods, 100 evaluations each at identical raw-frame budgets. '
        'Lines show ten-episode means and light bands show episode standard deviations, '
        'not uncertainty across training seeds.</desc>',
        '<defs><clipPath id="plot"><rect x="90" y="100" width="690" height="410"/>'
        '</clipPath></defs>',
        '<rect width="1000" height="620" fill="white"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#222">',
        '<text x="90" y="38" font-size="23">Walker Walk · training seed 0</text>',
        '<text x="90" y="66" font-size="14">100 evaluations per method · 10 episodes per evaluation</text>',
    ]
    for value in range(0, 1001, 200):
        elements.extend([
            f'<line x1="{left}" y1="{y(value):.2f}" x2="{left + plot_width}" '
            f'y2="{y(value):.2f}" stroke="#dedede"/>',
            f'<text x="78" y="{y(value) + 5:.2f}" text-anchor="end" font-size="13">{value}</text>',
        ])
    for frame in range(0, 1_000_001, 200_000):
        elements.append(f'<text x="{x(frame):.2f}" y="536" text-anchor="middle" '
                        f'font-size="13">{frame // 1000}K</text>')
    elements += [
        '<path d="M90 100 V510 H780" fill="none" stroke="#555"/>',
        '<text x="435" y="568" text-anchor="middle" font-size="15">Raw environment frames</text>',
        '<text transform="translate(25 305) rotate(-90)" text-anchor="middle" '
        'font-size="15">Episode return</text>',
    ]
    styles = {
        "legacy": ("#EE7733", "3 4"),
        "lif_t8": ("#AA3377", "10 3 2 3"),
        "stateful_t8": ("#009988", "6 4"),
        "ann_gru": ("#222222", "12 4"),
        "stateful_gatedmem32": ("#0077BB", ""),
    }
    runs = {run["preset"]: run for run in data["completed"]}
    for index, (preset, (color, dash)) in enumerate(styles.items()):
        run = runs[preset]
        points = run["evaluations"]
        upper = [(x(p["frames"]), y(p["eval_return"] + p["eval_std"])) for p in points]
        lower = [(x(p["frames"]), y(p["eval_return"] - p["eval_std"])) for p in reversed(points)]
        band = " ".join(f"{a:.2f},{b:.2f}" for a, b in upper + lower)
        line = " ".join(f'{x(p["frames"]):.2f},{y(p["eval_return"]):.2f}' for p in points)
        thickness = 2.8 if preset == "stateful_gatedmem32" else 1.8
        elements += [
            f'<g clip-path="url(#plot)"><polygon points="{band}" fill="{color}" opacity="0.07"/>',
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="{thickness}" '
            f'stroke-dasharray="{dash}" stroke-linejoin="round"/></g>',
            f'<line x1="805" y1="{125 + index * 34}" x2="839" y2="{125 + index * 34}" '
            f'stroke="{color}" stroke-width="{thickness}" stroke-dasharray="{dash}"/>',
            f'<text x="847" y="{130 + index * 34}" font-size="13">{html.escape(run["label"])}</text>',
        ]
    elements += [
        '<text x="90" y="605" font-size="13">Bands: ±1 episode SD (clipped at plot limits); '
        'no smoothing; not multi-seed confidence intervals.</text>',
        '</g></svg>',
    ]
    return "\n".join(elements) + "\n"


def render(data, output, readme=None):
    output.mkdir(parents=True, exist_ok=True)
    (output / "walker_seed0.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "walker_learning_curves.svg").write_text(learning_curve_svg(data))
    (output / "walker_summary.md").write_text(
        "# Walker seed 0 完整结果\n\n" + table(data) +
        "\n\n五组均为 1M 原始帧、71,171 次更新。± 是十个评估回合的标准差；"
        "末十次评估均值取 910K–1M 的十个位置，不挑最高点。\n\n"
        "原始分数、完整曲线、配置、时长和提前停止记录见 [JSON](walker_seed0.json)。\n")
    if readme:
        content = readme.read_text()
        if content.count(BEGIN) != 1 or content.count(END) != 1:
            raise ValueError("README must have exactly one result-table marker pair")
        before, body = content.split(BEGIN, 1)
        _, after = body.split(END, 1)
        readme.write_text(before + BEGIN + "\n\n" + table(data) + "\n\n" + END + after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive-root", type=Path)
    source.add_argument("--data", type=Path, help="Regenerate figures from the published JSON")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--readme", type=Path, help="Update only the marked results table")
    args = parser.parse_args()
    data = collect(args.archive_root) if args.archive_root else json.loads(args.data.read_text())
    render(data, args.output, args.readme)
    print(f'Exported {len(data["completed"])} complete runs and {len(data["stopped"])} stopped explorations.')


if __name__ == "__main__":
    main()
