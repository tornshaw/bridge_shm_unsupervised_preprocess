from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .core import (
    BridgeSHMUnsupervisedPreprocessor,
    infer_bridge_name_from_path,
    infer_sampling_minutes,
    infer_time_column,
    print_chinese_anomaly_summary,
    read_input,
    set_seed,
)


@dataclass
class BridgeAnalysisTask:
    bridge_name: str
    source_type: str  # offline | online
    csv_path: str
    selected_sensors: Optional[List[str]] = None


def _filter_df(df: pd.DataFrame, start_time: Optional[pd.Timestamp], end_time: Optional[pd.Timestamp], sensors: Optional[Sequence[str]]) -> pd.DataFrame:
    out = df.copy()
    time_axis, has_dt = infer_time_column(out)
    if has_dt:
        t = pd.to_datetime(time_axis, errors="coerce")
        mask = pd.Series(True, index=out.index)
        if start_time is not None:
            mask &= t >= pd.Timestamp(start_time)
        if end_time is not None:
            mask &= t <= pd.Timestamp(end_time)
        out = out.loc[mask].copy()

    if sensors:
        keep = [out.columns[0]] + [c for c in sensors if c in out.columns]
        out = out[keep].copy()
    return out


def _window_size_by_days(df: pd.DataFrame, sample_days: float) -> int:
    time_axis, has_dt = infer_time_column(df)
    if not has_dt:
        return 12
    sampling_minutes = infer_sampling_minutes(pd.to_datetime(time_axis, errors="coerce"))
    return int(max(4, round(24 * 60 * sample_days / max(1, sampling_minutes))))


def run_single_bridge_task(
    task: BridgeAnalysisTask,
    output_root: str,
    analysis_mode: str,
    start_time: Optional[pd.Timestamp],
    end_time: Optional[pd.Timestamp],
    sample_days: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    set_seed(42)
    df = read_input(task.csv_path)
    df = _filter_df(df, start_time=start_time, end_time=end_time, sensors=task.selected_sensors)
    if len(df) < 2:
        raise ValueError(f"桥梁 {task.bridge_name} 在所选时段内数据不足。")

    if analysis_mode == "普通预处理分析":
        epochs, latent_dim = 25, 16
    else:
        epochs, latent_dim = 60, 32

    window_size = _window_size_by_days(df, sample_days=sample_days)
    pre = BridgeSHMUnsupervisedPreprocessor(
        window_size=window_size,
        stride=1,
        latent_dim=latent_dim,
        epochs=epochs,
    )

    pre.fit(df)
    outputs = pre.transform(df, bridge_name=task.bridge_name)
    print_chinese_anomaly_summary(outputs)

    bridge_out = os.path.join(output_root, task.bridge_name)
    os.makedirs(bridge_out, exist_ok=True)
    for name, out_df in outputs.items():
        out_df.to_csv(os.path.join(bridge_out, f"{name}.csv"), index=False, encoding="utf-8-sig")

    pre.save_visualizations(df, outputs, bridge_out, plot_top_k=4)
    return outputs


def run_multi_bridge_tasks(
    tasks: Sequence[BridgeAnalysisTask],
    output_root: str,
    analysis_mode: str,
    start_time: Optional[pd.Timestamp],
    end_time: Optional[pd.Timestamp],
    sample_days: float = 1.0,
) -> pd.DataFrame:
    os.makedirs(output_root, exist_ok=True)
    metric_rows = []
    for t in tasks:
        outputs = run_single_bridge_task(
            task=t,
            output_root=output_root,
            analysis_mode=analysis_mode,
            start_time=start_time,
            end_time=end_time,
            sample_days=sample_days,
        )
        row = outputs["bridge_test_metrics"].iloc[0].to_dict()
        row["bridge_name"] = t.bridge_name
        metric_rows.append(row)

    summary = pd.DataFrame(metric_rows)
    summary_path = os.path.join(output_root, "multi_bridge_metrics_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    _plot_multi_bridge_compare(summary, output_root)
    return summary


def _plot_multi_bridge_compare(summary: pd.DataFrame, output_root: str) -> None:
    if summary.empty:
        return
    cols = [c for c in ["bridge_project_score", "avg_device_health", "avg_availability"] if c in summary.columns]
    if not cols:
        return

    plot_df = summary[["bridge_name"] + cols].set_index("bridge_name")
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_df.plot(kind="bar", ax=ax, rot=25)
    ax.set_title("多桥对比结果")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_root, "multi_bridge_compare.png"), dpi=180)
    plt.close(fig)


def build_offline_tasks(csv_paths: Sequence[str], bridge_sensor_select: Dict[str, List[str]]) -> List[BridgeAnalysisTask]:
    tasks: List[BridgeAnalysisTask] = []
    for p in csv_paths:
        bridge = infer_bridge_name_from_path(p)
        tasks.append(
            BridgeAnalysisTask(
                bridge_name=bridge,
                source_type="offline",
                csv_path=p,
                selected_sensors=bridge_sensor_select.get(bridge),
            )
        )
    return tasks
