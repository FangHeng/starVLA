#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path


STEP_RE = re.compile(r">> Step (?P<step>\d+), Loss:")
PROGRESS_RE = re.compile(r"\|\s*(?P<step>\d+)/\d+\s+\[(?P<minutes>\d+):(?P<seconds>\d+)<")
PROGRESS_TIMING_RE = re.compile(
    r"\|\s*(?P<step>\d+)/\d+\s+\[(?P<minutes>\d+):(?P<seconds>\d+)<[^\]]*"
    r"data_times=(?P<data_time>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?),\s*"
    r"model_times=(?P<model_time>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"(?P<date>\d{2}/\d{2}) \[(?P<time>\d{2}:\d{2}:\d{2})\]")
METRIC_RE = re.compile(r"'(?P<key>[^']+)'\s*:\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.IGNORECASE)
ERROR_RE = re.compile(r"Traceback|FloatingPointError|ERROR")


def _parse_metrics(block):
    normalized = " ".join(block.split())
    return {match.group("key"): float(match.group("value")) for match in METRIC_RE.finditer(normalized)}


def _parse_progress_elapsed(line, step):
    matches = list(PROGRESS_RE.finditer(line))
    for match in reversed(matches):
        if int(match.group("step")) == step:
            return int(match.group("minutes")) * 60 + int(match.group("seconds"))
    return None


def _parse_progress_timings(lines):
    by_step = {}
    for line in lines:
        for match in PROGRESS_TIMING_RE.finditer(line):
            step = int(match.group("step"))
            by_step[step] = {
                "step": step,
                "elapsed_s": int(match.group("minutes")) * 60 + int(match.group("seconds")),
                "data_time": float(match.group("data_time")),
                "model_time": float(match.group("model_time")),
            }
    return [by_step[step] for step in sorted(by_step)]


def _nearest_rank_percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100.0 * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _timing_stats(progress_steps, key):
    values = [step[key] for step in progress_steps]
    if not values:
        return {"mean": None, "p95": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "p95": _nearest_rank_percentile(values, 95),
        "max": max(values),
    }


def summarize_train_log(log_path):
    log_path = Path(log_path)
    text = log_path.read_text(errors="replace").replace("\r", "\n")
    lines = text.splitlines()
    logged_steps = []
    progress_steps = _parse_progress_timings(lines)

    for index, line in enumerate(lines):
        step_match = STEP_RE.search(line)
        if not step_match:
            continue

        step = int(step_match.group("step"))
        block = "\n".join(lines[index : index + 16])
        time_match = TIME_RE.search(line)
        metrics = _parse_metrics(block)
        logged_steps.append(
            {
                "step": step,
                "progress_elapsed_s": _parse_progress_elapsed(line, step),
                "log_time": f"{time_match.group('date')} {time_match.group('time')}" if time_match else None,
                "metrics": metrics,
            }
        )

    intervals = []
    for previous, current in zip(logged_steps, logged_steps[1:]):
        previous_elapsed = previous.get("progress_elapsed_s")
        current_elapsed = current.get("progress_elapsed_s")
        if previous_elapsed is not None and current_elapsed is not None:
            intervals.append(current_elapsed - previous_elapsed)

    latest = logged_steps[-1] if logged_steps else {}
    error_markers = [line for line in lines if ERROR_RE.search(line)]
    return {
        "log_path": str(log_path),
        "logged_step_count": len(logged_steps),
        "latest_step": latest.get("step"),
        "latest_log_time": latest.get("log_time"),
        "latest_metrics": latest.get("metrics", {}),
        "avg_logged_interval_s": sum(intervals) / len(intervals) if intervals else None,
        "logged_steps": logged_steps,
        "progress_step_count": len(progress_steps),
        "progress_stats": {
            "data_time": _timing_stats(progress_steps, "data_time"),
            "model_time": _timing_stats(progress_steps, "model_time"),
        },
        "progress_top_data_spikes": sorted(progress_steps, key=lambda item: item["data_time"], reverse=True)[:10],
        "progress_top_model_spikes": sorted(progress_steps, key=lambda item: item["model_time"], reverse=True)[:10],
        "error_markers": error_markers,
    }


def _metric_value(step, key):
    value = step["metrics"].get(key)
    return f"{value:.6f}" if value is not None else ""


def render_markdown(summary, tail=10):
    lines = [
        f"# train log summary: `{summary['log_path']}`",
        "",
        f"- logged steps: {summary['logged_step_count']}",
        f"- latest step: {summary['latest_step']}",
        f"- latest log time: {summary['latest_log_time']}",
        f"- progress samples: {summary['progress_step_count']}",
    ]
    if summary["avg_logged_interval_s"] is not None:
        lines.append(f"- average logged interval: {summary['avg_logged_interval_s']:.3f}s")
    lines.extend(
        [
            f"- error markers: {len(summary['error_markers'])}",
            "",
            "## Recent Logged Steps",
            "",
            "| step | elapsed s | loss | model_time | data_time | mse_score | learning_rate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for step in summary["logged_steps"][-tail:]:
        elapsed = step["progress_elapsed_s"] if step["progress_elapsed_s"] is not None else ""
        loss = step["metrics"].get("loss", step["metrics"].get("action_dit_loss"))
        loss_text = f"{loss:.6f}" if loss is not None else ""
        lines.append(
            "| {step} | {elapsed} | {loss} | {model_time} | {data_time} | {mse_score} | {learning_rate} |".format(
                step=step["step"],
                elapsed=elapsed,
                loss=loss_text,
                model_time=_metric_value(step, "model_time"),
                data_time=_metric_value(step, "data_time"),
                mse_score=_metric_value(step, "mse_score"),
                learning_rate=_metric_value(step, "learning_rate"),
            )
        )

    if summary["progress_step_count"]:
        lines.extend(
            [
                "",
                "## Progress Timing",
                "",
                "| metric | mean | p95 | max |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for metric, stats in summary["progress_stats"].items():
            lines.append(
                "| {metric} | {mean:.6f} | {p95:.6f} | {max:.6f} |".format(
                    metric=metric,
                    mean=stats["mean"],
                    p95=stats["p95"],
                    max=stats["max"],
                )
            )

        lines.extend(
            [
                "",
                "## Top Data Spikes",
                "",
                "| step | elapsed s | data_time | model_time |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for step in summary["progress_top_data_spikes"][:tail]:
            lines.append(
                f"| {step['step']} | {step['elapsed_s']} | {step['data_time']:.6f} | {step['model_time']:.6f} |"
            )

    if summary["error_markers"]:
        lines.extend(["", "## Error Markers", ""])
        for marker in summary["error_markers"][-20:]:
            lines.append(f"- {marker}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize a StarVLA training log.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()

    summary = summarize_train_log(args.log_path)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_markdown(summary, tail=args.tail), end="")


if __name__ == "__main__":
    main()
