#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""几个可复现的测试 Demo：快速验证多桥双层评分结果。"""

from __future__ import annotations

import csv
import os
from typing import Dict, List

from bridge_shm_unsupervised_preprocess_final import process_bridge


# demo-1: 系统缺失较明显 + 相对健康 + 常规桥梁
DEMO_BRIDGES = [
    "沥心沙大桥_20251201_20260318.csv",
    "浅海大桥_20251201_20260318.csv",
    "新榄核大桥_20251201_20260318.csv",
]


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_top3_worst_sensor(sensor_summary_csv: str) -> List[str]:
    rows = load_rows(sensor_summary_csv)
    rows.sort(key=lambda x: float(x["project_score"]))
    return [
        f"{r['sensor_id']}|{r['sensor_type']}(D={float(r['device_health']):.1f},A={float(r['availability']):.1f},P={float(r['project_score']):.1f})"
        for r in rows[:3]
    ]


def read_bridge_event_highlights(event_summary_csv: str) -> str:
    rows = load_rows(event_summary_csv)
    if not rows:
        return "no-events"
    r = rows[0]
    return (
        f"known_system_gap_hours={r['known_system_gap_hours']}, "
        f"device_gap_hours={r['device_gap_hours']}, "
        f"stuck_events={r['stuck_events']}, drift_events={r['drift_events']}, "
        f"step_events={r['step_events']}, spike_events={r['spike_events']}"
    )


def main() -> None:
    data_dir = "data"
    out_root = "outputs/demo_cases"
    os.makedirs(out_root, exist_ok=True)

    report_lines = [
        "# 多桥测试 Demo（双层评分）",
        "",
        "## 说明",
        "- DeviceHealth 低 / Availability 高：偏设备侧问题",
        "- DeviceHealth 高 / Availability 低：偏系统侧问题",
        "- 两者都低：设备与系统可能同时有问题",
        "",
    ]

    metric_header = [
        "bridge_name",
        "samples",
        "sensor_count",
        "total_missing_ratio",
        "system_missing_ratio",
        "device_missing_ratio",
        "avg_device_health",
        "avg_availability",
        "bridge_project_score",
    ]
    metric_rows: List[Dict[str, object]] = []

    for file_name in DEMO_BRIDGES:
        csv_path = os.path.join(data_dir, file_name)
        bridge_result = process_bridge(csv_path, out_root)

        bridge_name = bridge_result["bridge_name"]
        bridge_dir = os.path.join(out_root, bridge_name)

        top3 = read_top3_worst_sensor(os.path.join(bridge_dir, "sensor_health_summary.csv"))

        event_tmp_csv = os.path.join(bridge_dir, "bridge_event_summary.csv")
        with open(event_tmp_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "bridge_name",
                    "startup_jump_count",
                    "known_system_gap_hours",
                    "device_gap_hours",
                    "stuck_events",
                    "drift_events",
                    "step_events",
                    "spike_events",
                ],
            )
            writer.writeheader()
            writer.writerow(bridge_result["event_summary"])

        event_text = read_bridge_event_highlights(event_tmp_csv)

        metric_rows.append({k: bridge_result[k] for k in metric_header})
        report_lines.extend(
            [
                f"## {bridge_name}",
                f"- samples={bridge_result['samples']}, sensors={bridge_result['sensor_count']}",
                f"- DeviceHealth={bridge_result['avg_device_health']}, Availability={bridge_result['avg_availability']}, ProjectScore={bridge_result['bridge_project_score']}",
                f"- missing(total/system/device)={bridge_result['total_missing_ratio']}/{bridge_result['system_missing_ratio']}/{bridge_result['device_missing_ratio']}",
                f"- 事件摘要: {event_text}",
                f"- 最差3个设备: {', '.join(top3)}",
                "",
            ]
        )

    metric_csv = os.path.join(out_root, "demo_metrics.csv")
    with open(metric_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_header)
        writer.writeheader()
        writer.writerows(metric_rows)

    report_md = os.path.join(out_root, "demo_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"saved: {metric_csv}")
    print(f"saved: {report_md}")
    print("\n建议再跑一次全量：python bridge_shm_unsupervised_preprocess_final.py")


if __name__ == "__main__":
    main()
