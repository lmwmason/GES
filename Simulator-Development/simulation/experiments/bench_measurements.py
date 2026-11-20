#!/usr/bin/env python3
"""Summarize measured bench CSVs. No hardware values are built in."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def unique(seen, key):
    if key in seen:
        raise ValueError(f"Repeated observation: {key}")
    seen.add(key)


def p95(values):
    ordered = sorted(values)
    index = 0.95 * (len(ordered) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo])


def latency(rows):
    counts = defaultdict(lambda: {"ok": 0, "timeout": 0})
    timings = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        direction = row["direction"]
        if direction not in ("nearer", "farther"):
            raise ValueError("direction must be nearer or farther")
        unique(seen, (direction, row["trial"]))
        stimulus = number(row, "stimulus_ms")
        if row["status"] not in ("ok", "timeout"):
            raise ValueError("status must be ok or timeout")
        counts[direction][row["status"]] += 1
        if row["status"] == "timeout":
            continue
        received = number(row, "received_ms")
        if received < stimulus:
            raise ValueError("received_ms precedes stimulus_ms")
        timings[direction]["stimulus_to_output_ms"].append(received - stimulus)
        if row.get("acquisition_start_ms"):
            start = number(row, "acquisition_start_ms")
            if received < start:
                raise ValueError("received_ms precedes acquisition_start_ms")
            timings[direction]["acquisition_to_output_ms"].append(received - start)
    return {
        direction: {
            **count,
            **{key: {"n": len(vals), "median": statistics.median(vals), "p95": p95(vals)}
               for key, vals in timings[direction].items()},
        }
        for direction, count in counts.items()
    }


def thrust(rows, rpm_tolerance):
    baseline = {}
    near = []
    seen = set()
    for row in rows:
        cell = (row["trial"], number(row, "target_rpm"))
        rpm, force = number(row, "rpm"), number(row, "thrust_N")
        if min(cell[1], rpm, force) <= 0:
            raise ValueError("target_rpm, rpm and thrust_N must be positive")
        if abs(rpm - cell[1]) / cell[1] > rpm_tolerance:
            raise ValueError(f"{cell} outside target RPM tolerance; retain raw row and retest.")
        if row["condition"] == "far":
            unique(seen, (cell, "far"))
            baseline[cell] = (rpm, force)
        elif row["condition"] == "near":
            height = number(row, "height_m")
            if height <= 0:
                raise ValueError("height_m must be positive for near observations")
            unique(seen, (cell, height))
            near.append((cell, height, rpm, force))
        else:
            raise ValueError("condition must be near or far")
    if not near or set(baseline) != {cell for cell, *_ in near}:
        raise ValueError("Each trial/target RPM needs one far reference and at least one near observation")
    grouped = defaultdict(list)
    paired = []
    for cell, height, rpm, force in near:
        ref_rpm, ref_force = baseline[cell]
        if abs(rpm - ref_rpm) / ref_rpm > rpm_tolerance:
            raise ValueError(f"{cell} near/far RPM mismatch; retain raw row and retest.")
        ratio = force / ref_force
        grouped[(cell[1], height)].append(ratio)
        paired.append({"trial": cell[0], "target_rpm": cell[1], "height_m": height,
                       "near_rpm": rpm, "far_rpm": ref_rpm, "near_thrust_N": force,
                       "far_thrust_N": ref_force, "factor": ratio})
    return {"paired": paired, "summary": [
        {"target_rpm": rpm, "height_m": height, "n": len(vals),
         "mean_factor": statistics.mean(vals),
         "sample_sd": statistics.stdev(vals) if len(vals) > 1 else None}
        for (rpm, height), vals in sorted(grouped.items())
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement", choices=["latency", "thrust"])
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--rpm-tolerance", type=float,
                        help="Required for thrust. Fractional tolerance fixed before collecting data.")
    args = parser.parse_args()
    if args.measurement == "thrust" and not (args.rpm_tolerance is not None
                                             and 0 < args.rpm_tolerance < 1):
        parser.error("thrust requires --rpm-tolerance between 0 and 1")
    try:
        with args.csv_file.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("CSV contains no observations")
        result = latency(rows) if args.measurement == "latency" else thrust(rows, args.rpm_tolerance)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
