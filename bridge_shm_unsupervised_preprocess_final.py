#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""桥梁健康监测无监督预处理（最终版，双层健康评分）"""
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
STARTUP_START = dt.datetime(2025, 12, 1)
STARTUP_END = dt.datetime(2026, 1, 10, 23, 59, 59)


@dataclass
class SensorStats:
    bridge_name: str
    sensor: str
    sensor_id: str
    sensor_type: str
    device_health: float
    availability: float
    project_score: float
    dominant_issue: str
    missing_ratio: float
    stuck_ratio: float
    drift_ratio: float
    spike_ratio: float


def parse_time(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def fmt_time(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def is_known_event_window(t: dt.datetime) -> bool:
    return (KNOWN_IOT_START <= t <= KNOWN_IOT_END) or (KNOWN_DB_START <= t <= KNOWN_DB_END)


def is_startup_window(t: dt.datetime) -> bool:
    return STARTUP_START <= t <= STARTUP_END


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


def run_lengths(flags: List[bool]) -> List[int]:
    out = [0] * len(flags)
    cnt = 0
    for i, f in enumerate(flags):
        cnt = cnt + 1 if f else 0
        out[i] = cnt
    return out


def calc_health_scores(labels: List[str]) -> Tuple[float, float, float, str, Dict[str, int]]:
    n = max(1, len(labels))
    c = Counter(labels)

    device_weights = {
        "device_gap": 2.2,
        "stuck": 2.4,
        "drift": 1.8,
        "step_change": 1.8,
        "spike": 1.3,
        "noise": 1.0,
        "cross_sensor_conflict": 1.5,
        "startup_jump": 0.2,
    }
    avail_weights = {
        "known_system_gap": 1.0,
        "bridge_wide_gap": 1.2,
    }

    d_penalty = sum(c.get(k, 0) * w for k, w in device_weights.items())
    a_penalty = sum(c.get(k, 0) * w for k, w in avail_weights.items())

    device_health = max(0.0, min(100.0, 100.0 * (1.0 - d_penalty / n)))
    availability = max(0.0, min(100.0, 100.0 * (1.0 - a_penalty / n)))
    project_score = 0.7 * device_health + 0.3 * availability

    dominant_issue = c.most_common(1)[0][0] if c else "normal"
    if dominant_issue == "normal" and len(c) > 1:
        dominant_issue = max((k for k in c if k != "normal"), key=lambda x: c[x], default="normal")
    return device_health, availability, project_score, dominant_issue, c


def save_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def split_sensor_name(sensor: str) -> Tuple[str, str]:
    if "_" in sensor:
        sid, stype = sensor.rsplit("_", 1)
        return sid, stype
    return sensor, "unknown"


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
    n = len(ts)

    bridge_missing = [0] * n
    for i, row in enumerate(data):
        bridge_missing[i] = sum(1 for v in row if v is None)
    bridge_wide_gap = [bridge_missing[i] >= max(2, int(0.8 * len(sensors))) or inserted[i] for i in range(n)]

    typed_values: Dict[str, List[List[Optional[float]]]] = defaultdict(list)
    sensor_to_type: Dict[str, str] = {}
    for j, s in enumerate(sensors):
        _, stype = split_sensor_name(s)
        sensor_to_type[s] = stype
        typed_values[stype].append([row[j] for row in data])

    type_medians: Dict[str, List[Optional[float]]] = {}
    for stype, cols in typed_values.items():
        med_col = []
        for i in range(n):
            vals = [col[i] for col in cols if col[i] is not None]
            med_col.append(statistics.median(vals) if vals else None)
        type_medians[stype] = med_col

    per_sensor_labels: Dict[str, List[str]] = {}
    per_sensor_clean: Dict[str, List[Optional[float]]] = {}
    sensor_health_rows: List[SensorStats] = []
    bridge_event_counter = Counter()

    for j, s in enumerate(sensors):
        stype = sensor_to_type[s]
        col = [row[j] for row in data]
        filled = linear_interpolate_column(ts, col)
        labels = ["normal"] * n

        abs_diff = [abs(filled[i] - filled[i - 1]) for i in range(1, n) if filled[i] is not None and filled[i - 1] is not None]
        diff_thr = robust_threshold(abs_diff, k=6.0)
        mad_center = median([v for v in filled if v is not None], 0.0)
        val_mad = mad([v for v in filled if v is not None], mad_center)

        diff_evidence = [False] * n
        mad_evidence = [False] * n
        var_evidence = [False] * n
        cross_evidence = [False] * n

        for i in range(1, n):
            if filled[i] is None or filled[i - 1] is None:
                continue
            local_thr = diff_thr * (2.0 if is_startup_window(ts[i]) else 1.0)
            if abs(filled[i] - filled[i - 1]) > local_thr:
                diff_evidence[i] = True

        for i in range(n):
            if filled[i] is None:
                continue
            local_k = 8.0 if is_startup_window(ts[i]) else 5.0
            if abs(filled[i] - mad_center) > local_k * val_mad:
                mad_evidence[i] = True

        win = 24
        for i in range(win - 1, n):
            seg = [v for v in filled[i - win + 1 : i + 1] if v is not None]
            if len(seg) < max(6, win // 2):
                continue
            seg_var = statistics.pvariance(seg) if len(seg) > 1 else 0.0
            if seg_var > (9.0 * (val_mad ** 2)):
                var_evidence[i] = True

        type_ref = type_medians[stype]
        for i in range(n):
            if filled[i] is None or type_ref[i] is None:
                continue
            if abs(filled[i] - type_ref[i]) > 6.0 * max(val_mad, 1e-6):
                cross_evidence[i] = True

        missing_flags = [v is None for v in col]
        missing_runs = run_lengths(missing_flags)
        for i in range(n):
            if not missing_flags[i]:
                continue
            if is_known_event_window(ts[i]) and bridge_wide_gap[i]:
                labels[i] = "known_system_gap"
            elif bridge_wide_gap[i]:
                labels[i] = "bridge_wide_gap"
            else:
                labels[i] = "device_gap"

        step_cand = [False] * n
        for i in range(1, n):
            if labels[i] != "normal":
                continue
            evidence = sum([diff_evidence[i], mad_evidence[i], cross_evidence[i]])
            if evidence >= 2:
                step_cand[i] = True
        for i in range(2, n):
            if step_cand[i] and step_cand[i - 1] and labels[i] == "normal":
                labels[i] = "startup_jump" if is_startup_window(ts[i]) else "step_change"

        low_var_eps = max(val_mad * 0.05, 1e-5)
        stable_run = 0
        for i in range(1, n):
            if filled[i] is None or filled[i - 1] is None:
                stable_run = 0
                continue
            if abs(filled[i] - filled[i - 1]) <= low_var_eps:
                stable_run += 1
            else:
                stable_run = 0
            if i >= 35 and labels[i] == "normal" and stable_run >= 20:
                seg = [v for v in filled[i - 35 : i + 1] if v is not None]
                if len(seg) >= 30 and (max(seg) - min(seg)) <= low_var_eps:
                    labels[i] = "stuck"

        # drift: 滑窗斜率 + 同类偏离
        for i in range(48, n):
            if labels[i] != "normal":
                continue
            x = []
            y = []
            for k in range(i - 47, i + 1):
                if filled[k] is not None:
                    x.append(k)
                    y.append(filled[k])
            if len(y) < 36:
                continue
            mx = sum(x) / len(x)
            my = sum(y) / len(y)
            den = sum((xx - mx) ** 2 for xx in x) + 1e-9
            slope = sum((xx - mx) * (yy - my) for xx, yy in zip(x, y)) / den
            if abs(slope) > max(0.03 * val_mad, 1e-5) and cross_evidence[i]:
                labels[i] = "drift"

        for i in range(n):
            if labels[i] != "normal":
                continue
            evidence = sum([mad_evidence[i], diff_evidence[i], var_evidence[i], cross_evidence[i]])
            if evidence >= 2:
                labels[i] = "spike"
            elif var_evidence[i] and mad_evidence[i]:
                labels[i] = "noise"
            elif cross_evidence[i] and mad_evidence[i]:
                labels[i] = "cross_sensor_conflict"

        cleaned = list(filled)
        for i in range(n):
            if labels[i] in {"known_system_gap", "bridge_wide_gap"}:
                cleaned[i] = None
            elif labels[i] == "device_gap":
                cleaned[i] = filled[i] if missing_runs[i] <= 6 else None

        per_sensor_labels[s] = labels
        per_sensor_clean[s] = cleaned

        device_health, availability, project_score, dominant_issue, counter = calc_health_scores(labels)
        sensor_id, sensor_type = split_sensor_name(s)
        sensor_health_rows.append(
            SensorStats(
                bridge_name=bridge,
                sensor=s,
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                device_health=round(device_health, 3),
                availability=round(availability, 3),
                project_score=round(project_score, 3),
                dominant_issue=dominant_issue,
                missing_ratio=round((counter.get("device_gap", 0) + counter.get("known_system_gap", 0) + counter.get("bridge_wide_gap", 0)) / n, 4),
                stuck_ratio=round(counter.get("stuck", 0) / n, 4),
                drift_ratio=round(counter.get("drift", 0) / n, 4),
                spike_ratio=round((counter.get("spike", 0) + counter.get("noise", 0)) / n, 4),
            )
        )
        bridge_event_counter.update(counter)

    cleaned_rows, label_rows = [], []
    for i, t in enumerate(ts):
        c_row = [fmt_time(t)]
        l_row = [fmt_time(t)]
        for s in sensors:
            v = per_sensor_clean[s][i]
            c_row.append("" if v is None else f"{v:.6f}")
            l_row.append(per_sensor_labels[s][i])
        cleaned_rows.append(c_row)
        label_rows.append(l_row)

    save_csv(os.path.join(out_dir, "cleaned_data.csv"), ["timestamp"] + sensors, cleaned_rows)
    save_csv(os.path.join(out_dir, "label_data.csv"), ["timestamp"] + sensors, label_rows)

    sensor_summary_rows = []
    for h in sorted(sensor_health_rows, key=lambda x: x.project_score):
        sensor_summary_rows.append([
            h.bridge_name,
            h.sensor_id,
            h.sensor_type,
            f"{h.device_health:.3f}",
            f"{h.availability:.3f}",
            f"{h.project_score:.3f}",
            h.dominant_issue,
            f"{h.missing_ratio:.4f}",
            f"{h.stuck_ratio:.4f}",
            f"{h.drift_ratio:.4f}",
            f"{h.spike_ratio:.4f}",
        ])
    save_csv(
        os.path.join(out_dir, "sensor_health_summary.csv"),
        [
            "bridge_name",
            "sensor_id",
            "sensor_type",
            "device_health",
            "availability",
            "project_score",
            "dominant_issue",
            "missing_ratio",
            "stuck_ratio",
            "drift_ratio",
            "spike_ratio",
        ],
        sensor_summary_rows,
    )

    bridge_device_health = sum(h.device_health for h in sensor_health_rows) / max(1, len(sensor_health_rows))
    bridge_availability = sum(h.availability for h in sensor_health_rows) / max(1, len(sensor_health_rows))
    bridge_project_score = 0.7 * bridge_device_health + 0.3 * bridge_availability

    total_points = max(1, len(ts) * len(sensors))
    system_missing = bridge_event_counter.get("known_system_gap", 0) + bridge_event_counter.get("bridge_wide_gap", 0)
    device_missing = bridge_event_counter.get("device_gap", 0)
    total_missing = system_missing + device_missing

    event_summary = {
        "bridge_name": bridge,
        "startup_jump_count": bridge_event_counter.get("startup_jump", 0),
        "known_system_gap_hours": round(bridge_event_counter.get("known_system_gap", 0) * sampling_min / 60.0, 2),
        "device_gap_hours": round(device_missing * sampling_min / 60.0, 2),
        "stuck_events": bridge_event_counter.get("stuck", 0),
        "drift_events": bridge_event_counter.get("drift", 0),
        "step_events": bridge_event_counter.get("step_change", 0),
        "spike_events": bridge_event_counter.get("spike", 0) + bridge_event_counter.get("noise", 0),
    }

    return {
        "bridge_name": bridge,
        "samples": len(ts),
        "sensor_count": len(sensors),
        "total_missing_ratio": round(total_missing / total_points, 4),
        "system_missing_ratio": round(system_missing / total_points, 4),
        "device_missing_ratio": round(device_missing / total_points, 4),
        "avg_device_health": round(bridge_device_health, 3),
        "avg_availability": round(bridge_availability, 3),
        "bridge_project_score": round(bridge_project_score, 3),
        "event_summary": event_summary,
        "sensor_summary_rows": sensor_summary_rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="多桥梁无监督数据预处理最终模型（双层评分）")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="outputs/final")
    args = p.parse_args()

    csv_files = sorted([os.path.join(args.data_dir, x) for x in os.listdir(args.data_dir) if x.endswith(".csv")])
    bridge_rows = []
    sensor_rows = []
    event_rows = []

    for fp in csv_files:
        r = process_bridge(fp, args.output_dir)
        bridge_rows.append([r[k] for k in [
            "bridge_name",
            "samples",
            "sensor_count",
            "total_missing_ratio",
            "system_missing_ratio",
            "device_missing_ratio",
            "avg_device_health",
            "avg_availability",
            "bridge_project_score",
        ]])
        sensor_rows.extend(r["sensor_summary_rows"])
        e = r["event_summary"]
        event_rows.append([e[k] for k in [
            "bridge_name",
            "startup_jump_count",
            "known_system_gap_hours",
            "device_gap_hours",
            "stuck_events",
            "drift_events",
            "step_events",
            "spike_events",
        ]])
        print(f"processed: {os.path.basename(fp)}")

    save_csv(
        os.path.join(args.output_dir, "bridge_test_metrics.csv"),
        [
            "bridge_name",
            "samples",
            "sensor_count",
            "total_missing_ratio",
            "system_missing_ratio",
            "device_missing_ratio",
            "avg_device_health",
            "avg_availability",
            "bridge_project_score",
        ],
        bridge_rows,
    )
    save_csv(
        os.path.join(args.output_dir, "sensor_health_summary.csv"),
        [
            "bridge_name",
            "sensor_id",
            "sensor_type",
            "device_health",
            "availability",
            "project_score",
            "dominant_issue",
            "missing_ratio",
            "stuck_ratio",
            "drift_ratio",
            "spike_ratio",
        ],
        sensor_rows,
    )
    save_csv(
        os.path.join(args.output_dir, "bridge_event_summary.csv"),
        [
            "bridge_name",
            "startup_jump_count",
            "known_system_gap_hours",
            "device_gap_hours",
            "stuck_events",
            "drift_events",
            "step_events",
            "spike_events",
        ],
        event_rows,
    )
    print(f"saved summary: {os.path.join(args.output_dir, 'bridge_test_metrics.csv')}")


if __name__ == "__main__":
    main()
