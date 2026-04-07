#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""几个可复现的测试 Demo：覆盖 IoT 掉线 / DB 扩容缺失 / 相对健康桥梁。"""

from __future__ import annotations

import csv
import os
from typing import List

from bridge_shm_unsupervised_preprocess_final import process_bridge


DEMO_BRIDGES = [
    "沥心沙大桥_20251201_20260318.csv",  # IoT 时段缺失较明显
    "浅海大桥_20251201_20260318.csv",    # DB 时段缺失较明显
    "新榄核大桥_20251201_20260318.csv",  # 相对健康
]


def read_top3_worst_sensor(health_csv: str) -> List[str]:
    rows = []
    with open(health_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    rows.sort(key=lambda x: float(x["health_index"]))
    return [f"{r['sensor']}({r['health_index']})" for r in rows[:3]]


def main() -> None:
    data_dir = "data"
    out_root = "outputs/demo_cases"
    os.makedirs(out_root, exist_ok=True)

    metric_rows = []
    report_lines = ["# Demo 测试结果", ""]

    for file_name in DEMO_BRIDGES:
        csv_path = os.path.join(data_dir, file_name)
        r = process_bridge(csv_path, out_root)
        metric_rows.append(r)

        bridge_dir = os.path.join(out_root, os.path.splitext(file_name)[0])
        top3 = read_top3_worst_sensor(os.path.join(bridge_dir, "sensor_health.csv"))
        report_lines.extend(
            [
                f"## {r['bridge']}",
                f"- 采样间隔: {r['sampling_minutes']} 分钟",
                f"- 桥级健康指数: {r['bridge_health_index']}",
                f"- 异常计数: missing={r['missing']}, known_event_gap={r['known_event_gap']}, spike={r['spike']}, drift={r['drift']}, stuck={r['stuck']}",
                f"- 最差3个传感器: {', '.join(top3)}",
                "",
            ]
        )

    csv_out = os.path.join(out_root, "demo_metrics.csv")
    header = [
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
    with open(csv_out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(metric_rows)

    md_out = os.path.join(out_root, "demo_report.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"saved: {csv_out}")
    print(f"saved: {md_out}")


if __name__ == "__main__":
    main()
