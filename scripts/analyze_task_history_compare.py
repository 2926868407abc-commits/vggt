#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from pathlib import Path


TASKS = ("pose_trans", "pose_rot", "depth", "point", "track")


def finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def median(values):
    values = finite(values)
    return statistics.median(values) if values else None


def summarize(rows, key):
    values = finite(row.get(key) for row in rows)
    if not values:
        return None
    width = max(1, len(values) // 10)
    first = median(values[:width])
    last = median(values[-width:])
    return {
        "first10_median": first,
        "last10_median": last,
        "last_over_first": None if first == 0 else last / first,
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def load(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="LABEL=training_history.jsonl")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        rows = load(path)
        item = {
            "path": str(Path(path)),
            "updates": len(rows),
            "first_update": rows[0].get("update") if rows else None,
            "last_update": rows[-1].get("update") if rows else None,
            "attack_metric": summarize(rows, "attack_metric"),
            "tasks": {},
        }
        for task in TASKS:
            item["tasks"][task] = {
                "score": summarize(rows, f"projected_{task}_score"),
                "grad_norm": summarize(rows, f"projected_{task}_grad_norm"),
                "weight": summarize(rows, f"projected_{task}_weight"),
                "gain": summarize(rows, f"projected_{task}_gain"),
            }
        report[label] = item

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
