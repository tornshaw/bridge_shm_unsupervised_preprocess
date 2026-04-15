from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BridgeAnalysisTask:
    bridge_name: str
    source_type: str  # offline | online
    csv_path: str
    selected_sensors: Optional[List[str]] = None


def _infer_time_column(df: pd.DataFrame) -> tuple[pd.Series, bool]:
    if df.empty:
        return pd.Series(dtype=float), False
    first = df.iloc[:, 0]
    t = pd.to_datetime(first, errors="coerce")
    if t.notna().mean() > 0.7:
        return t, True
    return pd.Series(np.arange(len(df))), False


def _infer_sampling_minutes(time_axis: pd.Series, fallback: int = 10) -> int:
    if not pd.api.types.is_datetime64_any_dtype(time_axis):
        return fallback
    d = pd.Series(time_axis).sort_values().diff().dropna().dt.total_seconds() / 60.0
    d = d[(d > 0) & np.isfinite(d)]
    if d.empty:
        return fallback
    return int(max(1, round(float(d.median()))))


def _infer_bridge_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    for suf in ["_20251201_20260318", "_online"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem


def _read_input(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _filter_df(df: pd.DataFrame, start_time: Optional[pd.Timestamp], end_time: Optional[pd.Timestamp], sensors: Optional[Sequence[str]]) -> pd.DataFrame:
    out = df.copy()
    time_axis, has_dt = _infer_time_column(out)
    if has_dt:
        mask = pd.Series(True, index=out.index)
        if start_time is not None:
            mask &= time_axis >= pd.Timestamp(start_time)
        if end_time is not None:
            mask &= time_axis <= pd.Timestamp(end_time)
        out = out.loc[mask].copy()

    if sensors:
        keep = [out.columns[0]] + [c for c in sensors if c in out.columns]
        out = out[keep].copy()
    return out


def _basic_preprocess_no_torch(df: pd.DataFrame, bridge_name: str) -> Dict[str, pd.DataFrame]:
    t, _ = _infer_time_column(df)
    sensor_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    sensor_names = sensor_df.columns.tolist()

    raw = sensor_df.to_numpy(dtype=float)
    missing = np.isnan(raw)
    interp = sensor_df.interpolate(limit_direction="both").bfill().ffill()
    cleaned = interp.to_numpy(dtype=float)

    med = np.nanmedian(cleaned, axis=0)
    mad = np.nanmedian(np.abs(cleaned - med), axis=0) + 1e-6
    z = np.abs((cleaned - med) / mad)

    labels = np.full_like(cleaned, "normal", dtype=object)
    labels[missing] = "device_gap"
    labels[(z > 3.5) & (~missing)] = "spike"

    cleaned_df = pd.DataFrame(cleaned, columns=sensor_names)
    cleaned_df.insert(0, "timestamp", t.values)

    score_df = pd.DataFrame(z, columns=sensor_names)
    score_df.insert(0, "timestamp", t.values)

    label_df = pd.DataFrame(labels, columns=sensor_names)
    label_df.insert(0, "timestamp", t.values)

    point_status_df = pd.DataFrame(
        {
            "timestamp": t.values,
            "abnormal_count": (label_df.drop(columns=["timestamp"]) != "normal").sum(axis=1).values,
            "point_score_mean": score_df.drop(columns=["timestamp"]).mean(axis=1).values,
            "point_score_max": score_df.drop(columns=["timestamp"]).max(axis=1).values,
        }
    )
    point_status_df["point_status"] = np.where(point_status_df["abnormal_count"] > 0, "abnormal", "normal")

    health_rows = []
    for s in sensor_names:
        l = label_df[s].astype(str)
        n = max(1, len(l))
        miss_ratio = float((l == "device_gap").sum()) / n
        spike_ratio = float((l == "spike").sum()) / n
        device_health = max(0.0, 100.0 * (1 - (0.65 * miss_ratio + 0.35 * spike_ratio)))
        availability = max(0.0, 100.0 * (1 - miss_ratio))
        project_score = 0.7 * device_health + 0.3 * availability
        dominant_issue = "device_gap" if miss_ratio >= spike_ratio and miss_ratio > 0 else ("spike" if spike_ratio > 0 else "normal")
        health_rows.append(
            {
                "sensor_name": s,
                "bridge_name": bridge_name,
                "sensor_id": s,
                "sensor_type": "unknown",
                "device_health": round(device_health, 3),
                "availability": round(availability, 3),
                "project_score": round(project_score, 3),
                "dominant_issue": dominant_issue,
                "missing_ratio": round(miss_ratio, 4),
                "stuck_ratio": 0.0,
                "drift_ratio": 0.0,
                "spike_ratio": round(spike_ratio, 4),
            }
        )

    health_df = pd.DataFrame(health_rows).sort_values(by=["project_score"], ascending=True)
    bridge_device_health = float(health_df["device_health"].mean()) if not health_df.empty else 100.0
    bridge_availability = float(health_df["availability"].mean()) if not health_df.empty else 100.0
    bridge_project_score = 0.7 * bridge_device_health + 0.3 * bridge_availability
    total_points = max(1, raw.size)
    device_missing = int((label_df.drop(columns=["timestamp"]) == "device_gap").sum().sum())

    bridge_metrics_df = pd.DataFrame(
        [
            {
                "bridge_name": bridge_name,
                "samples": len(sensor_df),
                "sensor_count": len(sensor_names),
                "total_missing_ratio": round(device_missing / total_points, 4),
                "system_missing_ratio": 0.0,
                "device_missing_ratio": round(device_missing / total_points, 4),
                "avg_device_health": round(bridge_device_health, 3),
                "avg_availability": round(bridge_availability, 3),
                "bridge_project_score": round(bridge_project_score, 3),
            }
        ]
    )

    bridge_event_df = pd.DataFrame(
        [
            {
                "bridge_name": bridge_name,
                "startup_jump_point_count": 0,
                "known_system_gap_sensor_hours": 0.0,
                "device_gap_sensor_hours": round(device_missing * _infer_sampling_minutes(t) / 60.0, 2),
                "stuck_point_count": 0,
                "drift_point_count": 0,
                "step_change_point_count": 0,
                "spike_noise_point_count": int((label_df.drop(columns=["timestamp"]) == "spike").sum().sum()),
            }
        ]
    )

    return {
        "cleaned_data": cleaned_df,
        "score_data": score_df,
        "label_data": label_df,
        "sensor_health_summary": health_df,
        "bridge_test_metrics": bridge_metrics_df,
        "bridge_event_summary": bridge_event_df,
        "point_status": point_status_df,
    }


def _save_basic_visualizations(df: pd.DataFrame, outputs: Dict[str, pd.DataFrame], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    t_raw, _ = _infer_time_column(df)
    t_clean = pd.to_datetime(outputs["cleaned_data"].iloc[:, 0], errors="coerce")
    health = outputs["sensor_health_summary"]
    top = health["sensor_name"].head(min(4, len(health))).tolist()

    # raw vs cleaned
    if top:
        fig, axes = plt.subplots(nrows=len(top), ncols=1, figsize=(12, max(4, 2.4 * len(top))), sharex=True)
        if len(top) == 1:
            axes = [axes]
        for ax, s in zip(axes, top):
            raw_series = pd.to_numeric(df[s], errors="coerce")
            clean_series = pd.to_numeric(outputs["cleaned_data"][s], errors="coerce")
            n = min(len(raw_series), len(clean_series), len(t_raw), len(t_clean))
            if n < 2:
                continue
            ax.plot(t_raw.iloc[:n], raw_series.iloc[:n], lw=0.8, alpha=0.5, label="Raw")
            ax.plot(t_clean.iloc[:n], clean_series.iloc[:n], lw=1.1, label="Cleaned")
            ax.set_title(f"{s}: 原始 vs 修复")
            ax.grid(alpha=0.25)
        axes[0].legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "raw_vs_cleaned_top_sensors.png"), dpi=160)
        plt.close(fig)

    # health
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(health) + 1)))
    worst = health.head(min(12, len(health)))
    ax.barh(worst["sensor_name"], worst["project_score"])
    ax.invert_yaxis()
    ax.set_title("风险最高传感器")
    ax.set_xlabel("project_score")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sensor_health_barh.png"), dpi=160)
    plt.close(fig)


def run_single_bridge_task(
    task: BridgeAnalysisTask,
    output_root: str,
    analysis_mode: str,
    start_time: Optional[pd.Timestamp],
    end_time: Optional[pd.Timestamp],
    sample_days: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    del sample_days
    df = _read_input(task.csv_path)
    df = _filter_df(df, start_time=start_time, end_time=end_time, sensors=task.selected_sensors)
    if len(df) < 2:
        raise ValueError(f"桥梁 {task.bridge_name} 在所选时段内数据不足。")

    if analysis_mode == "普通预处理分析":
        outputs = _basic_preprocess_no_torch(df, task.bridge_name)
    else:
        # 仅在高级模式时才导入 torch 依赖链
        from .core import (
            BridgeSHMUnsupervisedPreprocessor,
            infer_sampling_minutes,
            infer_time_column,
            print_chinese_anomaly_summary,
            set_seed,
        )

        set_seed(42)
        time_axis, has_dt = infer_time_column(df)
        if has_dt:
            sampling_minutes = infer_sampling_minutes(pd.to_datetime(time_axis, errors="coerce"))
            window_size = int(max(4, round(24 * 60 * 1.0 / max(1, sampling_minutes))))
        else:
            window_size = 12
        pre = BridgeSHMUnsupervisedPreprocessor(window_size=window_size, stride=1, latent_dim=32, epochs=60)
        pre.fit(df)
        outputs = pre.transform(df, bridge_name=task.bridge_name)
        print_chinese_anomaly_summary(outputs)

    bridge_out = os.path.join(output_root, task.bridge_name)
    os.makedirs(bridge_out, exist_ok=True)
    for name, out_df in outputs.items():
        out_df.to_csv(os.path.join(bridge_out, f"{name}.csv"), index=False, encoding="utf-8-sig")

    if analysis_mode == "普通预处理分析":
        _save_basic_visualizations(df, outputs, bridge_out)
    else:
        # 高级模式下优先输出 core 的完整图件（热力图/全通道状态图/潜空间等）
        try:
            pre.save_visualizations(df, outputs, bridge_out, plot_top_k=4)  # type: ignore[name-defined]
        except Exception:
            # 若完整图件生成失败，退化到基础图件，保证流程不中断
            _save_basic_visualizations(df, outputs, bridge_out)

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
        bridge = _infer_bridge_name_from_path(p)
        tasks.append(
            BridgeAnalysisTask(
                bridge_name=bridge,
                source_type="offline",
                csv_path=p,
                selected_sensors=bridge_sensor_select.get(bridge),
            )
        )
    return tasks
