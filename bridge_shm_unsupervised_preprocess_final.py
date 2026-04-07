#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""桥梁健康监测无监督预处理（最终版，标准库实现）"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


KNOWN_IOT_START = dt.datetime(2026, 1, 25)
KNOWN_IOT_END = dt.datetime(2026, 2, 5, 23, 59, 59)
KNOWN_DB_START = dt.datetime(2026, 2, 20)
KNOWN_DB_END = dt.datetime(2026, 3, 10, 23, 59, 59)
STARTUP_END = dt.datetime(2026, 1, 10, 23, 59, 59)


@dataclass
class SensorStats:
    sensor: str
    health_index: float
    abnormal_ratio: float
    dominant_label: str
    missing_count: int
    known_event_gap_count: int
    spike_count: int
    drift_count: int
    stuck_count: int
    startup_jump_count: int


def parse_time(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def fmt_time(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def is_known_event_window(t: dt.datetime) -> bool:
    return (KNOWN_IOT_START <= t <= KNOWN_IOT_END) or (KNOWN_DB_START <= t <= KNOWN_DB_END)


def median(values: Sequence[float], default: float = 0.0) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(vals) if vals else default


def mad(values: Sequence[float], center: Optional[float] = None, eps: float = 1e-9) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return eps
    c = statistics.median(vals) if center is None else center
    return statistics.median([abs(v - c) for v in vals]) + eps


def robust_threshold(values: Sequence[float], k: float) -> float:
    c = median(values, default=0.0)
    return c + k * mad(values, center=c)


def linear_interpolate_column(ts: List[dt.datetime], x: List[Optional[float]]) -> List[Optional[float]]:
    out = list(x)
    n = len(out)
    valid_idx = [i for i, v in enumerate(out) if v is not None and math.isfinite(v)]
    if not valid_idx:
        return out
    first, last = valid_idx[0], valid_idx[-1]
    for i in range(0, first):
        out[i] = out[first]
    for i in range(last + 1, n):
        out[i] = out[last]
    i = first
    while i < last:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j <= last and out[j] is None:
            j += 1
        left = i - 1
        right = j
        if right > last:
            break
        t0, t1 = ts[left], ts[right]
        v0, v1 = out[left], out[right]
        total = (t1 - t0).total_seconds()
        for k in range(i, right):
            ratio = (ts[k] - t0).total_seconds() / total if total > 0 else 0.0
            out[k] = v0 + ratio * (v1 - v0)
        i = right
    return out


def infer_interval_minutes(ts: List[dt.datetime]) -> int:
    if len(ts) < 2:
        return 10
    deltas = [(ts[i] - ts[i - 1]).total_seconds() / 60.0 for i in range(1, len(ts))]
    deltas = [d for d in deltas if d > 0]
    return max(1, int(round(statistics.median(deltas)))) if deltas else 10


def align_regular_grid(
    timestamps: List[dt.datetime],
    rows: List[List[Optional[float]]],
) -> Tuple[List[dt.datetime], List[List[Optional[float]]], List[bool], int]:
    interval = infer_interval_minutes(timestamps)
    by_time = {t: r for t, r in zip(timestamps, rows)}
    t = timestamps[0]
    end = timestamps[-1]
    step = dt.timedelta(minutes=interval)
    all_ts: List[dt.datetime] = []
    all_rows: List[List[Optional[float]]] = []
    inserted: List[bool] = []
    sensor_n = len(rows[0])
    while t <= end:
        all_ts.append(t)
        if t in by_time:
            all_rows.append(by_time[t])
            inserted.append(False)
        else:
            all_rows.append([None] * sensor_n)
            inserted.append(True)
        t += step
    return all_ts, all_rows, inserted, interval


def detect_sensor_labels(
    ts: List[dt.datetime],
    values: List[Optional[float]],
    inserted_gap: List[bool],
) -> Tuple[List[str], List[Optional[float]]]:
    n = len(values)
    labels = ["normal"] * n

    for i, v in enumerate(values):
        if v is None:
            labels[i] = "missing"
        if inserted_gap[i] and labels[i] == "missing":
            labels[i] = "system_gap"
        if labels[i] in {"missing", "system_gap"} and is_known_event_window(ts[i]):
            labels[i] = "known_event_gap"

    filled = linear_interpolate_column(ts, values)

    diffs = []
    for i in range(1, n):
        a, b = filled[i - 1], filled[i]
        if a is not None and b is not None:
            diffs.append(abs(b - a))
    spike_thr = robust_threshold(diffs, k=8.0)

    startup_mask = [t <= STARTUP_END for t in ts]
    for i in range(1, n):
        a, b = filled[i - 1], filled[i]
        if a is None or b is None:
            continue
        d = abs(b - a)
        if d > spike_thr and labels[i] in {"normal", "missing"}:
            labels[i] = "startup_jump" if startup_mask[i] else "spike"

    # 简化 drift：24 点窗口均值偏差（约 4 小时）
    window = 36
    valid_vals = [v for v in filled if v is not None]
    center = median(valid_vals, 0.0)
    spread = mad(valid_vals, center=center)
    drift_thr = max(5.0 * spread, 1e-6)
    drift_candidate = [False] * n
    for i in range(n):
        if i < window or filled[i] is None:
            continue
        seg = [v for v in filled[i - window + 1 : i + 1] if v is not None]
        if len(seg) < window // 2:
            continue
        m = sum(seg) / len(seg)
        if abs(m - center) > drift_thr and labels[i] == "normal":
            drift_candidate[i] = True
    for i in range(5, n):
        if labels[i] != "normal":
            continue
        if sum(1 for x in drift_candidate[i - 5 : i + 1] if x) >= 5:
            labels[i] = "drift"

    # stuck：连续 12 点变化几乎为 0（约 2 小时）
    run = 24
    eps = max(spread * 0.02, 1e-5)
    for i in range(run - 1, n):
        seg = filled[i - run + 1 : i + 1]
        if any(v is None for v in seg):
            continue
        if max(seg) - min(seg) <= eps and labels[i] == "normal":
            labels[i] = "stuck"

    cleaned = list(filled)
    for i, lb in enumerate(labels):
        if lb in {"spike", "drift", "stuck", "missing", "system_gap", "known_event_gap"}:
            cleaned[i] = filled[i]
        elif lb == "startup_jump":
            # 启动阶段跳变降低误报，保守不修复
            cleaned[i] = values[i] if values[i] is not None else filled[i]

    return labels, cleaned


def health_from_labels(labels: List[str]) -> Tuple[float, float, str]:
    n = len(labels)
    c = Counter(labels)
    weighted = (
        c.get("missing", 0) * 1.0
        + c.get("system_gap", 0) * 1.0
        + c.get("known_event_gap", 0) * 0.6
        + c.get("spike", 0) * 2.0
        + c.get("drift", 0) * 1.5
        + c.get("stuck", 0) * 2.0
        + c.get("startup_jump", 0) * 0.4
    )
    abnormal_ratio = (n - c.get("normal", 0)) / max(n, 1)
    hi = max(0.0, min(100.0, 100.0 * (1.0 - weighted / max(n, 1))))
    dominant = c.most_common(1)[0][0] if c else "normal"
    return hi, abnormal_ratio, dominant


def save_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_health_svg(path: str, health_rows: List[SensorStats], top_k: int = 20) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    target = sorted(health_rows, key=lambda x: x.health_index)[:top_k]
    width = 1200
    bar_h = 22
    gap = 8
    left = 260
    height = 80 + len(target) * (bar_h + gap)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append('<text x="20" y="30" font-size="20" font-family="Arial">Worst Sensor Health Index (Top)</text>')
    for i, row in enumerate(target):
        y = 60 + i * (bar_h + gap)
        bw = int((width - left - 80) * max(0.0, min(100.0, row.health_index)) / 100.0)
        color = "#d73027" if row.health_index < 60 else ("#fc8d59" if row.health_index < 80 else "#91cf60")
        lines.append(f'<text x="20" y="{y+16}" font-size="12" font-family="Arial">{row.sensor}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bw}" height="{bar_h}" fill="{color}"/>')
        lines.append(f'<text x="{left+bw+8}" y="{y+16}" font-size="12" font-family="Arial">{row.health_index:.1f}</text>')
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def process_bridge(csv_path: str, output_root: str) -> Dict[str, object]:
    bridge = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(output_root, bridge)
    os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw = list(reader)

    timestamps = [parse_time(r[0]) for r in raw]
    sensors = header[1:]
    rows: List[List[Optional[float]]] = []
    for r in raw:
        vals: List[Optional[float]] = []
        for v in r[1:]:
            if v == "":
                vals.append(None)
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(None)
        rows.append(vals)

    ts, data, inserted, sampling_min = align_regular_grid(timestamps, rows)

    per_sensor_labels: Dict[str, List[str]] = {}
    per_sensor_clean: Dict[str, List[Optional[float]]] = {}
    health_rows: List[SensorStats] = []

    for j, s in enumerate(sensors):
        col = [row[j] for row in data]
        labels, cleaned = detect_sensor_labels(ts, col, inserted)
        per_sensor_labels[s] = labels
        per_sensor_clean[s] = cleaned
        hi, abnormal_ratio, dominant = health_from_labels(labels)
        c = Counter(labels)
        health_rows.append(
            SensorStats(
                sensor=s,
                health_index=round(hi, 3),
                abnormal_ratio=round(abnormal_ratio, 4),
                dominant_label=dominant,
                missing_count=c.get("missing", 0),
                known_event_gap_count=c.get("known_event_gap", 0),
                spike_count=c.get("spike", 0),
                drift_count=c.get("drift", 0),
                stuck_count=c.get("stuck", 0),
                startup_jump_count=c.get("startup_jump", 0),
            )
        )

    # 输出 cleaned_data
    cleaned_rows = []
    label_rows = []
    point_rows = []
    for i, t in enumerate(ts):
        c_row = [fmt_time(t)]
        l_row = [fmt_time(t)]
        abnormal_count = 0
        for s in sensors:
            v = per_sensor_clean[s][i]
            c_row.append("" if v is None else f"{v:.6f}")
            lb = per_sensor_labels[s][i]
            l_row.append(lb)
            if lb != "normal":
                abnormal_count += 1
        cleaned_rows.append(c_row)
        label_rows.append(l_row)
        point_rows.append([fmt_time(t), abnormal_count, "abnormal" if abnormal_count else "normal"])

    save_csv(os.path.join(out_dir, "cleaned_data.csv"), ["timestamp"] + sensors, cleaned_rows)
    save_csv(os.path.join(out_dir, "label_data.csv"), ["timestamp"] + sensors, label_rows)

    health_table = [[
        h.sensor,
        f"{h.health_index:.3f}",
        f"{h.abnormal_ratio:.4f}",
        h.dominant_label,
        h.missing_count,
        h.known_event_gap_count,
        h.spike_count,
        h.drift_count,
        h.stuck_count,
        h.startup_jump_count,
    ] for h in sorted(health_rows, key=lambda x: x.health_index)]
    save_csv(
        os.path.join(out_dir, "sensor_health.csv"),
        [
            "sensor",
            "health_index",
            "abnormal_ratio",
            "dominant_label",
            "missing_count",
            "known_event_gap_count",
            "spike_count",
            "drift_count",
            "stuck_count",
            "startup_jump_count",
        ],
        health_table,
    )
    save_csv(os.path.join(out_dir, "point_status.csv"), ["timestamp", "abnormal_count", "point_status"], point_rows)
    save_health_svg(os.path.join(out_dir, "sensor_health_top.svg"), health_rows)

    # 桥级指标
    all_labels = [lb for s in sensors for lb in per_sensor_labels[s]]
    c_all = Counter(all_labels)
    bridge_health = sum(h.health_index for h in health_rows) / max(len(health_rows), 1)
    return {
        "bridge": bridge,
        "sampling_minutes": sampling_min,
        "rows_after_align": len(ts),
        "sensor_count": len(sensors),
        "bridge_health_index": round(bridge_health, 3),
        "missing": c_all.get("missing", 0),
        "known_event_gap": c_all.get("known_event_gap", 0),
        "spike": c_all.get("spike", 0),
        "drift": c_all.get("drift", 0),
        "stuck": c_all.get("stuck", 0),
        "startup_jump": c_all.get("startup_jump", 0),
        "system_gap": c_all.get("system_gap", 0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="多桥梁无监督数据预处理最终模型")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="outputs/final")
    args = p.parse_args()

    csv_files = sorted([os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir) if x.endswith(".csv")])
    results = []
    for fp in csv_files:
        results.append(process_bridge(fp, args.output_dir))
        print(f"processed: {os.path.basename(fp)}")

    summary_header = [
        "bridge",
        "sampling_minutes",
        "rows_after_align",
        "sensor_count",
        "bridge_health_index",
        "missing",
        "known_event_gap",
        "system_gap",
        "startup_jump",
        "spike",
        "drift",
        "stuck",
    ]
    summary_rows = [[r[h] for h in summary_header] for r in results]
    save_csv(os.path.join(args.output_dir, "bridge_test_metrics.csv"), summary_header, summary_rows)
    print(f"saved summary: {os.path.join(args.output_dir, 'bridge_test_metrics.csv')}")


if __name__ == "__main__":
    main()
