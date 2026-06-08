#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


TIMESTAMP_RE = re.compile(r"^===\s*(?P<timestamp>.*?)\s*===")
NPU_RE = re.compile(r"^\|\s*(?P<npu>\d+)\s+910")
CHIP_RE = re.compile(
    r"^\|\s*\d+\s+\|\s+[^|]+\|\s*"
    r"(?P<aicore>[-+]?\d+(?:\.\d+)?)\s+"
    r"\d+\s*/\s*\d+\s+"
    r"(?P<hbm_used>\d+)\s*/\s*(?P<hbm_total>\d+)"
)


def parse_npu_smi(log_path):
    current_timestamp = None
    current_npu = None
    samples = []
    for line in Path(log_path).read_text(errors="replace").splitlines():
        timestamp_match = TIMESTAMP_RE.search(line)
        if timestamp_match:
            current_timestamp = timestamp_match.group("timestamp")
            current_npu = None
            continue

        npu_match = NPU_RE.search(line)
        if npu_match:
            current_npu = int(npu_match.group("npu"))
            continue

        chip_match = CHIP_RE.search(line)
        if chip_match and current_npu is not None:
            samples.append(
                {
                    "timestamp": current_timestamp,
                    "npu": current_npu,
                    "aicore": float(chip_match.group("aicore")),
                    "hbm_used_mb": int(chip_match.group("hbm_used")),
                    "hbm_total_mb": int(chip_match.group("hbm_total")),
                }
            )
            current_npu = None
    return samples


def _avg(values):
    return sum(values) / len(values) if values else None


def summarize_npu_smi(log_path, loaded_hbm_threshold=10000):
    samples = parse_npu_smi(log_path)
    loaded = [sample for sample in samples if sample["hbm_used_mb"] >= loaded_hbm_threshold]
    loaded_aicore = [sample["aicore"] for sample in loaded]
    nonzero_loaded = [value for value in loaded_aicore if value > 0]
    per_npu = {}
    for sample in loaded:
        bucket = per_npu.setdefault(
            sample["npu"],
            {
                "count": 0,
                "avg_aicore": 0.0,
                "max_aicore": 0.0,
                "avg_hbm_used_mb": 0.0,
                "max_hbm_used_mb": 0,
            },
        )
        bucket["count"] += 1
        bucket["avg_aicore"] += sample["aicore"]
        bucket["max_aicore"] = max(bucket["max_aicore"], sample["aicore"])
        bucket["avg_hbm_used_mb"] += sample["hbm_used_mb"]
        bucket["max_hbm_used_mb"] = max(bucket["max_hbm_used_mb"], sample["hbm_used_mb"])

    for bucket in per_npu.values():
        if bucket["count"]:
            bucket["avg_aicore"] /= bucket["count"]
            bucket["avg_hbm_used_mb"] /= bucket["count"]

    return {
        "log_path": str(log_path),
        "loaded_hbm_threshold": loaded_hbm_threshold,
        "sample_count": len(samples),
        "loaded_sample_count": len(loaded),
        "avg_loaded_aicore": _avg(loaded_aicore),
        "max_loaded_aicore": max(loaded_aicore) if loaded_aicore else None,
        "nonzero_loaded_ratio": len(nonzero_loaded) / len(loaded_aicore) if loaded_aicore else None,
        "avg_loaded_hbm_used_mb": _avg([sample["hbm_used_mb"] for sample in loaded]),
        "max_loaded_hbm_used_mb": max([sample["hbm_used_mb"] for sample in loaded], default=None),
        "max_hbm_used_mb": max([sample["hbm_used_mb"] for sample in samples], default=None),
        "per_npu": dict(sorted(per_npu.items())),
    }


def _fmt(value, digits=2):
    return "" if value is None else f"{value:.{digits}f}"


def render_markdown(summary):
    lines = [
        f"# npu-smi summary: `{summary['log_path']}`",
        "",
        f"- samples: {summary['sample_count']}",
        f"- loaded samples: {summary['loaded_sample_count']}",
        f"- loaded HBM threshold MB: {summary['loaded_hbm_threshold']}",
        f"- average loaded AICore %: {_fmt(summary['avg_loaded_aicore'])}",
        f"- max loaded AICore %: {_fmt(summary['max_loaded_aicore'])}",
        f"- nonzero loaded ratio: {_fmt(summary['nonzero_loaded_ratio'])}",
        f"- average loaded HBM MB: {_fmt(summary['avg_loaded_hbm_used_mb'])}",
        f"- max loaded HBM MB: {_fmt(summary['max_loaded_hbm_used_mb'])}",
        f"- max observed HBM MB: {_fmt(summary['max_hbm_used_mb'])}",
        "",
        "## Per NPU",
        "",
        "| npu | samples | avg AICore % | max AICore % | avg HBM MB | max HBM MB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for npu, item in summary["per_npu"].items():
        lines.append(
            f"| {npu} | {item['count']} | {item['avg_aicore']:.2f} | "
            f"{item['max_aicore']:.2f} | {item['avg_hbm_used_mb']:.2f} | "
            f"{item['max_hbm_used_mb']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize raw npu-smi sampling logs.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--loaded-hbm-threshold", type=int, default=10000)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    summary = summarize_npu_smi(args.log_path, loaded_hbm_threshold=args.loaded_hbm_threshold)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_markdown(summary), end="")


if __name__ == "__main__":
    main()
