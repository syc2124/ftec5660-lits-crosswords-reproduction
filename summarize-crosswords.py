"""Aggregate six completed runs from the upstream evaluator and inference logs."""

import csv
import json
import re
from pathlib import Path
from statistics import mean, variance


ROOT = Path(__file__).resolve().parent


def collect_runs():
    runs = []
    puzzle_scores = []
    trajectories = []
    for folder in sorted((ROOT / "crossword_results").glob("r*_steps*")):
        match = re.fullmatch(r"r([1-3])_steps(5|10)", folder.name)
        if not match:
            continue
        repeat, steps = map(int, match.groups())
        score = json.loads((folder / "eval_results.json").read_text(encoding="utf-8"))
        predictions = score["predictions"]
        ids = sorted(int(Path(item["file"]).stem) for item in predictions)
        if ids != list(range(1, 31)):
            raise ValueError(f"{folder.name}: expected the 30 fixed puzzle IDs, got {ids}")
        calls = [
            json.loads(line)
            for line in (folder / "inferencelogger.log").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        runs.append({
            "repeat": repeat,
            "steps": steps,
            "n": len(ids),
            "correct_count": score["correct_count"],
            "exact": score["accuracy"],
            "clue": score["soft_accuracy"],
            "calls": len(calls),
            "input_tokens": sum(call["input_tokens"] for call in calls),
            "output_tokens": sum(call["output_tokens"] for call in calls),
            "api_seconds": sum(call["running_time"] for call in calls),
        })
        for item in predictions:
            puzzle_scores.append({
                "repeat": repeat,
                "steps": steps,
                "puzzle": int(Path(item["file"]).stem),
                "exact": int(item["is_reached"]),
                "clue": item["score"],
            })
            checkpoint = json.loads(
                (folder / "checkpoints" / item["file"]).read_text(encoding="utf-8")
            )
            trajectory = checkpoint["state"]["steps"]
            trajectories.append({
                "repeat": repeat,
                "steps": steps,
                "puzzle": int(Path(item["file"]).stem),
                "actions": [step.get("action") for step in trajectory],
                "final_board": trajectory[-1].get("next_state") if trajectory else None,
            })
    if {(run["repeat"], run["steps"]) for run in runs} != {
        (repeat, steps) for repeat in (1, 2, 3) for steps in (5, 10)
    }:
        raise ValueError("Expected three complete paired repeats")
    return runs, puzzle_scores, trajectories


def summarize(runs):
    metrics = ("exact", "clue", "calls", "input_tokens", "output_tokens", "api_seconds")
    return {
        str(steps): {
            metric: {
                "mean": mean(run[metric] for run in runs if run["steps"] == steps),
                "sample_variance": variance(run[metric] for run in runs if run["steps"] == steps),
            }
            for metric in metrics
        }
        for steps in (10, 5)
    }


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    runs, puzzle_scores, trajectories = collect_runs()
    output = ROOT / "reproduction-evidence"
    output.mkdir(exist_ok=True)
    write_csv(output / "runs.csv", runs)
    write_csv(output / "puzzles.csv", puzzle_scores)
    (output / "trajectories.json").write_text(
        json.dumps(trajectories, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summarize(runs), indent=2) + "\n", encoding="utf-8"
    )
