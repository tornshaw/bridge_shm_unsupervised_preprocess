#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多桥梁数据快速体检（无第三方依赖）。"""

from __future__ import annotations

import csv
import datetime as dt
import glob
import os
import statistics
from typing import List, Tuple


def parse_time(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def run_one(csv_path: str) -> dict:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    timestamps = [parse_time(r[0]) for r in rows]
    sensor_count = len(header) - 1
    missing_ratio_per_row = []
    for row in rows:
        miss = sum(1 for v in row[1:] if v == "")
        missing_ratio_per_row.append(miss / max(sensor_count, 1))

    deltas = [(timestamps[i] - timestamps[i - 1]).total_seconds() / 60 for i in range(1, len(timestamps))]
    median_delta = statistics.median(deltas) if deltas else 10.0
    large_gaps = [d for d in deltas if d > median_delta * 3]

    def missing_rate_between(start: dt.datetime, end: dt.datetime) -> float:
        idx = [i for i, t in enumerate(timestamps) if start <= t <= end]
        if not idx:
            return 0.0
        return sum(1 for i in idx if missing_ratio_per_row[i] >= 0.8) / len(idx)

    return {
        "bridge_file": os.path.basename(csv_path),
        "rows": len(rows),
        "sensors": sensor_count,
        "median_sampling_minutes": round(median_delta, 2),
        "high_missing_row_ratio": round(sum(1 for r in missing_ratio_per_row if r >= 0.8) / max(len(rows), 1), 4),
        "max_gap_minutes": round(max(large_gaps) if large_gaps else median_delta, 1),
        "iot_period_high_missing_ratio": round(
            missing_rate_between(dt.datetime(2026, 1, 25), dt.datetime(2026, 2, 5, 23, 59, 59)),
            4,
        ),
        "db_period_high_missing_ratio": round(
            missing_rate_between(dt.datetime(2026, 2, 20), dt.datetime(2026, 3, 10, 23, 59, 59)),
            4,
        ),
    }


def main() -> None:
    rows: List[dict] = []
    for path in sorted(glob.glob("data/*.csv")):
        rows.append(run_one(path))

    out_path = "outputs/data_quick_test_summary.csv"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
