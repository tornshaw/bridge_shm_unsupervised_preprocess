#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桥梁健康监测系统无监督数据预处理原型
-----------------------------------
方法名称：
    基于时空图自监督解耦与多置信融合的无监督数据预处理方法

功能：
1. 多源监测数据对时、稳健标准化、缺失掩码生成
2. 基于空间位置和统计相关性的自适应传感器图构建
3. 基于掩码去噪自编码器的无监督表征学习
4. 融合重构误差、时序连续性、图一致性、潜在空间密度的质量评分
5. 自动识别缺失、突刺、漂移、卡滞(stuck)等异常
6. 基于解码结果 + 时序插值 + 图邻域估计进行置信融合修复
7. 生成训练曲线、异常热力图、传感器健康条形图、原始/清洗对比图、邻接矩阵图、潜空间散点图

依赖：
    numpy, pandas, scikit-learn, torch, matplotlib

输入：
    - csv 文件：第一列可为时间戳，其余列为传感器数值
    - 可选 positions csv：包含 [sensor, x, y, z] 或 [sensor, x, y]
输出：
    - cleaned_data.csv             修复后的数据
    - score_data.csv               综合质量评分
    - label_data.csv               异常标签
    - sensor_health_summary.csv    传感器健康摘要
    - bridge_test_metrics.csv      桥级测试指标
    - bridge_event_summary.csv     桥级事件摘要
    - point_status.csv             时刻级状态摘要
    - *.png                        可视化图件
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


KNOWN_IOT_START = dt.datetime(2026, 1, 25)
KNOWN_IOT_END = dt.datetime(2026, 2, 5, 23, 59, 59)
KNOWN_DB_START = dt.datetime(2026, 2, 20)
KNOWN_DB_END = dt.datetime(2026, 3, 10, 23, 59, 59)
STARTUP_START = dt.datetime(2025, 12, 1)
STARTUP_END = dt.datetime(2026, 1, 10, 23, 59, 59)


# -----------------------------
# 工具函数
# -----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def robust_center_scale(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(x, axis=0)
    q75 = np.nanpercentile(x, 75, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr < eps, 1.0, iqr)
    x_scaled = (x - median) / iqr
    return x_scaled, median, iqr


def inverse_scale(x_scaled: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    return x_scaled * iqr + median


def moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def rolling_std_1d(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=1, center=True).std().fillna(0.0).to_numpy()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_mad(x: np.ndarray, eps: float = 1e-6) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return mad + eps


def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (mad + eps)


def infer_threshold_gmm(score: np.ndarray) -> float:
    """使用双高斯混合自动确定异常阈值。若失败则退化为 median + 3*MAD。"""
    score = np.asarray(score, dtype=float).reshape(-1, 1)
    finite_mask = np.isfinite(score[:, 0])
    score = score[finite_mask]
    if len(score) == 0:
        return 3.0

    try:
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        gmm.fit(score)
        means = gmm.means_.flatten()
        covs = np.sqrt(gmm.covariances_.reshape(-1))
        weights = gmm.weights_.flatten()

        idx = np.argsort(means)
        m1, m2 = means[idx]
        s1, s2 = covs[idx]
        w1, w2 = weights[idx]

        xs = np.linspace(float(score.min()), float(score.max()), 2000).reshape(-1, 1)
        p1 = w1 * (1 / (np.sqrt(2 * np.pi) * max(s1, 1e-6))) * np.exp(-0.5 * ((xs - m1) / max(s1, 1e-6)) ** 2)
        p2 = w2 * (1 / (np.sqrt(2 * np.pi) * max(s2, 1e-6))) * np.exp(-0.5 * ((xs - m2) / max(s2, 1e-6)) ** 2)
        cross_idx = int(np.argmin(np.abs(p1 - p2)))
        tau = float(xs[cross_idx, 0])
        if not np.isfinite(tau):
            raise ValueError("invalid tau")
        return tau
    except Exception:
        s = score.flatten()
        return float(np.median(s) + 3.0 * safe_mad(s))


def _try_parse_numeric_timestamp(series: pd.Series) -> pd.Series:
    """尝试把纯数字时间戳解析成 datetime。优先秒，再毫秒。"""
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna().mean()
    if valid < 0.7:
        return pd.Series([pd.NaT] * len(series))

    parsed_s = pd.to_datetime(s, unit="s", errors="coerce")
    if parsed_s.notna().mean() > 0.7:
        return parsed_s

    parsed_ms = pd.to_datetime(s, unit="ms", errors="coerce")
    if parsed_ms.notna().mean() > 0.7:
        return parsed_ms

    return pd.Series([pd.NaT] * len(series))


def infer_time_column(df: pd.DataFrame) -> Tuple[pd.Series, bool]:
    """推断时间列；若不存在，则返回样本序号。"""
    if df.empty:
        return pd.Series(dtype=float), False

    first_name = str(df.columns[0]).strip().lower()
    first_col = df.iloc[:, 0]
    timestamp_like_names = {"timestamp", "time", "datetime", "date", "时间", "日期"}

    # 优先按“列名像时间列”处理
    if first_name in timestamp_like_names:
        if pd.api.types.is_numeric_dtype(first_col):
            parsed = _try_parse_numeric_timestamp(first_col)
        else:
            parsed = pd.to_datetime(first_col, errors="coerce")
        if parsed.notna().mean() > 0.7:
            return parsed, True
        return first_col.astype(str), False

    # 非数值列，也尝试解析为时间
    if not pd.api.types.is_numeric_dtype(first_col):
        parsed = pd.to_datetime(first_col, errors="coerce")
        if parsed.notna().mean() > 0.7:
            return parsed, True
        return first_col.astype(str), False

    return pd.Series(np.arange(len(df))), False


def get_sensor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """提取传感器数值列；若首列被判定为时间列，则从第二列开始选。"""
    _, has_datetime = infer_time_column(df)

    if df.shape[1] >= 2:
        if has_datetime:
            candidate = df.iloc[:, 1:]
        else:
            candidate = df
    else:
        candidate = df

    sensor_df = candidate.select_dtypes(include=[np.number]).copy()
    if sensor_df.empty:
        raise ValueError("未识别到传感器数值列，请检查输入 CSV 格式。")
    return sensor_df


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_figure(fig: plt.Figure, path: str, dpi: int = 160) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def style_time_axis(ax: plt.Axes, time_axis: pd.Series) -> None:
    if pd.api.types.is_datetime64_any_dtype(time_axis):
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)


def infer_sampling_minutes(time_axis: pd.Series, fallback: int = 10) -> int:
    """估计采样周期（分钟），用于补齐掉线造成的时间缺口。"""
    if not pd.api.types.is_datetime64_any_dtype(time_axis):
        return fallback
    deltas = pd.Series(time_axis).sort_values().diff().dropna().dt.total_seconds() / 60.0
    deltas = deltas[(deltas > 0) & np.isfinite(deltas)]
    if deltas.empty:
        return fallback
    return int(max(1, round(float(deltas.median()))))


def align_to_regular_timeline(df: pd.DataFrame, sampling_minutes: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    将数据重采样到规则时间轴并显式插入缺失行。
    返回：(对齐后的 DataFrame, 插入行掩码)。
    """
    time_axis, has_datetime = infer_time_column(df)
    if not has_datetime:
        return df.copy(), pd.Series(False, index=np.arange(len(df)))

    work = df.copy()
    work.iloc[:, 0] = pd.to_datetime(work.iloc[:, 0], errors="coerce")
    work = work.dropna(subset=[work.columns[0]]).drop_duplicates(subset=[work.columns[0]], keep="last")
    work = work.sort_values(work.columns[0]).reset_index(drop=True)

    if work.empty:
        return df.copy(), pd.Series(False, index=np.arange(len(df)))

    freq_min = sampling_minutes or infer_sampling_minutes(work.iloc[:, 0])
    full_time = pd.date_range(work.iloc[0, 0], work.iloc[-1, 0], freq=f"{freq_min}min")
    aligned = work.set_index(work.columns[0]).reindex(full_time).reset_index()
    aligned = aligned.rename(columns={"index": work.columns[0]})
    inserted_mask = aligned.drop(columns=[work.columns[0]]).isna().all(axis=1)
    return aligned, inserted_mask


def build_known_event_mask(time_axis: pd.Series) -> pd.Series:
    """
    构造已知运维事件窗口掩码：
    - 2026-01-25~2026-02-05：物联网卡掉线窗口
    - 2026-02-20~2026-03-10：数据库扩容/迁移窗口
    """
    if not pd.api.types.is_datetime64_any_dtype(time_axis):
        return pd.Series(False, index=np.arange(len(time_axis)))
    t = pd.to_datetime(time_axis, errors="coerce")
    iot_gap = (t >= pd.Timestamp("2026-01-25")) & (t <= pd.Timestamp("2026-02-05 23:59:59"))
    db_gap = (t >= pd.Timestamp("2026-02-20")) & (t <= pd.Timestamp("2026-03-10 23:59:59"))
    return (iot_gap | db_gap).fillna(False)


def build_startup_mask(time_axis: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(time_axis):
        return pd.Series(False, index=np.arange(len(time_axis)))
    t = pd.to_datetime(time_axis, errors="coerce")
    return ((t >= pd.Timestamp(STARTUP_START)) & (t <= pd.Timestamp(STARTUP_END))).fillna(False)


def split_sensor_name(sensor: str) -> Tuple[str, str]:
    if "_" in sensor:
        sid, stype = sensor.rsplit("_", 1)
        return sid, stype
    return sensor, "unknown"


def calc_health_scores(labels: Sequence[str]) -> Tuple[float, float, float, str, Counter]:
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
    avail_weights = {"known_system_gap": 1.0, "bridge_wide_gap": 1.2}
    d_penalty = sum(c.get(k, 0) * w for k, w in device_weights.items())
    a_penalty = sum(c.get(k, 0) * w for k, w in avail_weights.items())
    device_health = max(0.0, min(100.0, 100.0 * (1.0 - d_penalty / n)))
    availability = max(0.0, min(100.0, 100.0 * (1.0 - a_penalty / n)))
    project_score = 0.7 * device_health + 0.3 * availability
    dominant_issue = c.most_common(1)[0][0] if c else "normal"
    if dominant_issue == "normal" and len(c) > 1:
        dominant_issue = max((k for k in c if k != "normal"), key=lambda x: c[x], default="normal")
    return device_health, availability, project_score, dominant_issue, c


# -----------------------------
# 模型
# -----------------------------
class MaskedDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (256, 128)):
        super().__init__()
        h1, h2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


@dataclass
class FitArtifacts:
    median: np.ndarray
    iqr: np.ndarray
    adjacency: np.ndarray
    window_size: int
    stride: int
    latent_windows: np.ndarray
    latent_window_score: np.ndarray
    sensor_names: List[str]
    threshold: float
    training_loss_history: List[float]
    sampling_minutes: int = 10


# -----------------------------
# 主类
# -----------------------------
def sanitize_bridge_name(name: Optional[str]) -> str:
    if name is None:
        return "single_bridge"
    s = str(name).strip()
    if not s:
        return "single_bridge"
    return s


def infer_bridge_name_from_path(path: Optional[str]) -> str:
    if not path:
        return "single_bridge"
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    stem = re.sub(r"_(\d{8})_(\d{8})$", "", stem)
    return sanitize_bridge_name(stem)


def print_chinese_anomaly_summary(outputs: Dict[str, pd.DataFrame], top_k: int = 8) -> None:
    label_df = outputs.get("label_data")
    health_df = outputs.get("sensor_health_summary")
    metrics_df = outputs.get("bridge_test_metrics")
    event_df = outputs.get("bridge_event_summary")
    if label_df is None or health_df is None or metrics_df is None or event_df is None:
        return

    bridge_name = str(metrics_df.iloc[0].get("bridge_name", "single_bridge")) if len(metrics_df) else "single_bridge"

    label_cols = [c for c in label_df.columns if c != "timestamp"]
    counter: Counter = Counter()
    for c in label_cols:
        counter.update(label_df[c].astype(str).tolist())
    total_points = sum(counter.values())
    abnormal_points = total_points - counter.get("normal", 0)
    abnormal_ratio = abnormal_points / max(1, total_points)

    zh_map = {
        "normal": "正常",
        "device_gap": "设备缺测",
        "bridge_wide_gap": "全桥缺测",
        "known_system_gap": "系统已知离线",
        "spike": "突刺",
        "noise": "噪声",
        "drift": "漂移",
        "stuck": "卡滞",
        "step_change": "阶跃",
        "startup_jump": "启动跳变",
        "cross_sensor_conflict": "跨传感器冲突",
    }

    print(f"\n========== 桥梁异常结果摘要（{bridge_name}） ==========")
    print(f"总点数：{total_points}，异常点数：{abnormal_points}，异常占比：{abnormal_ratio:.2%}")
    print("异常类型统计（中文）：")
    for key, cnt in counter.most_common():
        if key == "normal":
            continue
        print(f"  - {zh_map.get(key, key)} ({key})：{cnt}")

    if len(health_df) > 0 and "sensor_name" in health_df.columns and "project_score" in health_df.columns:
        print(f"风险最高 Top{min(top_k, len(health_df))} 传感器：")
        worst = health_df.nsmallest(min(top_k, len(health_df)), "project_score")
        for _, row in worst.iterrows():
            dom = str(row.get("dominant_issue", "normal"))
            print(
                f"  - {row.get('sensor_name', '')}: 综合分={row.get('project_score', 0):.2f}, "
                f"主导问题={zh_map.get(dom, dom)}"
            )

    if len(event_df) > 0:
        erow = event_df.iloc[0].to_dict()
        print("事件级统计（便于报告阅读）：")
        print(
            "  - 启动跳变点数={startup}，系统缺测传感器小时={sys_h}，设备缺测传感器小时={dev_h}".format(
                startup=erow.get("startup_jump_point_count", 0),
                sys_h=erow.get("known_system_gap_sensor_hours", 0),
                dev_h=erow.get("device_gap_sensor_hours", 0),
            )
        )
        print(
            "  - 卡滞点数={stuck}，漂移点数={drift}，阶跃点数={step}，突刺/噪声点数={spike}".format(
                stuck=erow.get("stuck_point_count", 0),
                drift=erow.get("drift_point_count", 0),
                step=erow.get("step_change_point_count", 0),
                spike=erow.get("spike_noise_point_count", 0),
            )
        )
    print("======================================================\n")


class BridgeSHMUnsupervisedPreprocessor:
    def __init__(
        self,
        window_size: int = 12,
        stride: int = 1,
        latent_dim: int = 32,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        mask_ratio: float = 0.15,
        device: Optional[str] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.mask_ratio = mask_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[MaskedDenoisingAutoencoder] = None
        self.artifacts: Optional[FitArtifacts] = None

    def _build_graph(
        self,
        x_scaled: np.ndarray,
        sensor_names: List[str],
        positions: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        m = x_scaled.shape[1]
        corr = np.nan_to_num(np.corrcoef(np.nan_to_num(x_scaled, nan=0.0), rowvar=False), nan=0.0)
        corr = np.abs(corr)
        np.fill_diagonal(corr, 1.0)

        if positions is not None and {"sensor", "x", "y"}.issubset(set(positions.columns)):
            pos = positions.set_index("sensor")
            coords = []
            for s in sensor_names:
                if s in pos.index:
                    row = pos.loc[s]
                    coord = [row["x"], row["y"], row["z"] if "z" in pos.columns else 0.0]
                else:
                    coord = [0.0, 0.0, 0.0]
                coords.append(coord)
            coords = np.array(coords, dtype=float)
            dist = pairwise_distances(coords)
            sigma = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
            spatial = np.exp(-(dist ** 2) / (sigma ** 2 + 1e-6))
        else:
            spatial = np.ones((m, m), dtype=float)

        adjacency = 0.55 * corr + 0.45 * spatial
        adjacency = np.nan_to_num(adjacency, nan=0.0)
        adjacency = adjacency - np.diag(np.diag(adjacency)) + np.eye(m)
        row_sum = adjacency.sum(axis=1, keepdims=True)
        adjacency = adjacency / np.clip(row_sum, 1e-6, None)
        return adjacency

    def _graph_smooth(self, x: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        return x @ adjacency.T

    def _make_windows(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        windows = []
        spans = []
        for start in range(0, x.shape[0] - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(x[start:end].reshape(-1))
            spans.append((start, end))
        if len(windows) == 0:
            return np.empty((0, x.shape[1] * self.window_size), dtype=np.float32), []
        return np.array(windows, dtype=np.float32), spans

    def _prepare_model_input(
        self,
        x_scaled: np.ndarray,
        adjacency: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        trend_window = max(3, self.window_size // 2)
        x_trend = np.apply_along_axis(moving_average_1d, 0, x_scaled, trend_window)
        x_res = x_scaled - x_trend
        x_graph = self._graph_smooth(x_res, adjacency)
        x_input = np.concatenate([x_res, x_graph], axis=1)
        return x_trend, x_res, x_graph, x_input

    def _encode_windows(self, windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("模型未训练。")
        self.model.eval()
        with torch.no_grad():
            windows_t = torch.from_numpy(windows).to(self.device)
            recon_windows_t, latent_t = self.model(windows_t)
        return recon_windows_t.cpu().numpy(), latent_t.cpu().numpy()

    def fit(self, df: pd.DataFrame, positions: Optional[pd.DataFrame] = None) -> "BridgeSHMUnsupervisedPreprocessor":
        aligned_df, _ = align_to_regular_timeline(df)
        sensor_df = get_sensor_frame(aligned_df)
        sensor_names = list(sensor_df.columns)
        x_raw = sensor_df.to_numpy(dtype=float)

        x_filled = pd.DataFrame(x_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy()
        x_scaled, median, iqr = robust_center_scale(x_filled)
        adjacency = self._build_graph(x_scaled, sensor_names, positions)
        _, _, _, x_input = self._prepare_model_input(x_scaled, adjacency)

        windows, spans = self._make_windows(x_input)
        if len(windows) == 0:
            raise ValueError("样本长度小于 window_size，无法构造滑动窗口。")
        _ = spans
        input_dim = windows.shape[1]

        self.model = MaskedDenoisingAutoencoder(input_dim=input_dim, latent_dim=self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = TensorDataset(torch.from_numpy(windows))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        loss_history: List[float] = []
        for _ in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                noise = 0.02 * torch.randn_like(batch)
                mask = (torch.rand_like(batch) > self.mask_ratio).float()

                # 仅对被遮蔽部分注入噪声，避免未遮蔽部分也被整体污染
                masked_batch = batch * mask + noise * (1.0 - mask)

                recon, z = self.model(masked_batch)

                loss_rec = torch.mean((recon - batch) ** 2)
                aug = batch + 0.01 * torch.randn_like(batch)
                _, z_aug = self.model(aug)
                loss_con = torch.mean((z - z_aug) ** 2)
                loss_latent = torch.mean(z ** 2)

                loss = loss_rec + 0.15 * loss_con + 0.001 * loss_latent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1

            loss_history.append(epoch_loss / max(batch_count, 1))

        recon_windows, latent_windows = self._encode_windows(windows)
        _ = recon_windows

        nn_k = min(8, len(latent_windows))
        nbrs = NearestNeighbors(n_neighbors=nn_k)
        nbrs.fit(latent_windows)
        dists, _ = nbrs.kneighbors(latent_windows)
        latent_window_score = dists[:, -1]

        dummy_score = np.zeros(len(aligned_df), dtype=float)
        counts = np.zeros(len(aligned_df), dtype=float)
        for idx, (s, e) in enumerate(spans):
            dummy_score[s:e] += latent_window_score[idx]
            counts[s:e] += 1
        latent_point_score = dummy_score / np.clip(counts, 1, None)
        threshold = infer_threshold_gmm(latent_point_score)

        aligned_time, _ = infer_time_column(aligned_df)
        self.artifacts = FitArtifacts(
            median=median,
            iqr=iqr,
            adjacency=adjacency,
            window_size=self.window_size,
            stride=self.stride,
            latent_windows=latent_windows,
            latent_window_score=latent_window_score,
            sensor_names=sensor_names,
            threshold=threshold,
            training_loss_history=loss_history,
            sampling_minutes=infer_sampling_minutes(pd.to_datetime(aligned_time, errors="coerce")),
        )
        return self

    def transform(self, df: pd.DataFrame, bridge_name: str = "single_bridge") -> Dict[str, pd.DataFrame]:
        if self.model is None or self.artifacts is None:
            raise RuntimeError("请先调用 fit()")

        bridge_name = sanitize_bridge_name(bridge_name)
        aligned_df, inserted_gap_rows = align_to_regular_timeline(df, sampling_minutes=self.artifacts.sampling_minutes)
        sensor_df = aligned_df[self.artifacts.sensor_names].copy()
        timestamps, _ = infer_time_column(aligned_df)
        if len(timestamps) != len(sensor_df):
            timestamps = pd.Series(np.arange(len(sensor_df)))

        known_event_mask = build_known_event_mask(timestamps)
        startup_mask = build_startup_mask(timestamps)

        x_raw = sensor_df.to_numpy(dtype=float)
        raw_missing = np.isnan(x_raw)

        x_interp = pd.DataFrame(x_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy()
        x_scaled = (x_interp - self.artifacts.median) / self.artifacts.iqr
        x_trend, _, _, x_input = self._prepare_model_input(x_scaled, self.artifacts.adjacency)

        windows, spans = self._make_windows(x_input)
        if len(windows) == 0:
            raise ValueError("transform 输入长度小于 window_size，无法构造滑动窗口。")

        recon_windows, latent = self._encode_windows(windows)

        recon_full = np.zeros_like(x_input)
        counts = np.zeros_like(x_input)
        window_scores = np.zeros(x_input.shape[0], dtype=float)
        score_counts = np.zeros(x_input.shape[0], dtype=float)

        nbrs = NearestNeighbors(n_neighbors=min(8, len(latent)))
        nbrs.fit(latent)
        dists, _ = nbrs.kneighbors(latent)
        latent_density_score = dists[:, -1]

        for idx, (s, e) in enumerate(spans):
            recon_part = recon_windows[idx].reshape(self.window_size, -1)
            recon_full[s:e] += recon_part
            counts[s:e] += 1
            window_scores[s:e] += latent_density_score[idx]
            score_counts[s:e] += 1

        recon_full = recon_full / np.clip(counts, 1, None)
        latent_point_score = window_scores / np.clip(score_counts, 1, None)

        recon_res = recon_full[:, : x_scaled.shape[1]]
        recon_scaled = recon_res + x_trend

        rec_err = np.abs(recon_scaled - x_scaled)
        neighbor_avg = np.zeros_like(x_scaled)
        neighbor_avg[0] = x_scaled[0]
        neighbor_avg[-1] = x_scaled[-1]
        if x_scaled.shape[0] > 2:
            neighbor_avg[1:-1] = 0.5 * (x_scaled[:-2] + x_scaled[2:])
        temp_err = np.abs(x_scaled - neighbor_avg)
        graph_ref = self._graph_smooth(x_scaled, self.artifacts.adjacency)
        graph_err = np.abs(x_scaled - graph_ref)
        density_err = np.repeat(latent_point_score.reshape(-1, 1), x_scaled.shape[1], axis=1)

        def cv_score(a: np.ndarray) -> float:
            return float(np.std(a) / (np.mean(a) + 1e-6))

        w_rec = cv_score(rec_err)
        w_temp = cv_score(temp_err)
        w_graph = cv_score(graph_err)
        w_den = cv_score(density_err)
        w_sum = w_rec + w_temp + w_graph + w_den + 1e-6
        weights = np.array([w_rec, w_temp, w_graph, w_den]) / w_sum

        total_score = (
            weights[0] * robust_zscore(rec_err.flatten()).reshape(rec_err.shape)
            + weights[1] * robust_zscore(temp_err.flatten()).reshape(temp_err.shape)
            + weights[2] * robust_zscore(graph_err.flatten()).reshape(graph_err.shape)
            + weights[3] * robust_zscore(density_err.flatten()).reshape(density_err.shape)
        )

        tau = infer_threshold_gmm(total_score.flatten())

        labels = np.full(x_scaled.shape, "normal", dtype=object)
        system_gap_mask = np.repeat(inserted_gap_rows.to_numpy().reshape(-1, 1), x_scaled.shape[1], axis=1)
        bridge_wide_missing = np.repeat((raw_missing.mean(axis=1) >= 0.8).reshape(-1, 1), x_scaled.shape[1], axis=1)
        known_event_2d = np.repeat(known_event_mask.to_numpy().reshape(-1, 1), x_scaled.shape[1], axis=1)

        labels[raw_missing & (~bridge_wide_missing)] = "device_gap"
        labels[(raw_missing & bridge_wide_missing) | system_gap_mask] = "bridge_wide_gap"

        # 已知窗口中的缺失统一归为 known_system_gap
        labels[known_event_2d & (raw_missing | system_gap_mask | bridge_wide_missing)] = "known_system_gap"

        first_diff = np.abs(np.diff(x_scaled, axis=0, prepend=x_scaled[[0], :]))
        diff_thr = np.nanpercentile(first_diff, 95)
        spike_mask = (total_score > tau) & (first_diff > diff_thr)
        startup_jump_mask = np.repeat(startup_mask.to_numpy().reshape(-1, 1), x_scaled.shape[1], axis=1) & spike_mask
        step_mask = (first_diff > np.nanpercentile(first_diff, 98)) & (total_score > tau * 0.8)

        labels[spike_mask & (~raw_missing) & (labels == "normal")] = "spike"
        labels[startup_jump_mask & (~raw_missing)] = "startup_jump"
        labels[step_mask & (~startup_jump_mask) & (~raw_missing) & (labels == "normal")] = "step_change"

        drift_labels = np.zeros_like(x_scaled, dtype=bool)
        run = max(5, self.window_size)
        for j in range(x_scaled.shape[1]):
            resid = x_scaled[:, j] - recon_scaled[:, j]
            bias = pd.Series(resid).rolling(run, min_periods=run // 2).mean().fillna(0.0).to_numpy()
            sign_consistency = (
                pd.Series(np.sign(bias))
                .rolling(run, min_periods=run // 2)
                .apply(lambda v: np.abs(np.mean(v)), raw=True)
                .fillna(0.0)
                .to_numpy()
            )
            bias_thr = np.nanpercentile(np.abs(bias), 90)
            drift_labels[:, j] = (np.abs(bias) > bias_thr) & (sign_consistency > 0.8) & (total_score[:, j] > tau * 0.8)

        labels[drift_labels & (~raw_missing) & (labels == "normal")] = "drift"

        stuck_labels = np.zeros_like(x_scaled, dtype=bool)
        for j in range(x_scaled.shape[1]):
            rs = rolling_std_1d(x_scaled[:, j], max(5, self.window_size))
            thr = np.nanpercentile(rs, 10)
            flat = rs <= max(thr, 1e-4)
            stuck_labels[:, j] = flat & (total_score[:, j] > tau * 0.6)

        labels[stuck_labels & (~raw_missing) & (labels == "normal")] = "stuck"

        labels[(total_score > tau) & (labels == "normal")] = "cross_sensor_conflict"
        noise_mask = (total_score > tau * 0.9) & (first_diff <= diff_thr) & (labels == "normal")
        labels[noise_mask] = "noise"

        time_interp = pd.DataFrame(x_scaled).interpolate(limit_direction="both").bfill().ffill().to_numpy()
        graph_est = graph_ref
        rec_est = recon_scaled

        rec_conf = sigmoid(-robust_zscore(rec_err.flatten())).reshape(rec_err.shape)
        temp_conf = sigmoid(-robust_zscore(temp_err.flatten())).reshape(temp_err.shape)
        graph_conf = sigmoid(-robust_zscore(graph_err.flatten())).reshape(graph_err.shape)

        w1 = rec_conf
        w2 = temp_conf
        w3 = graph_conf
        wsum = w1 + w2 + w3 + 1e-6
        fused_scaled = (w1 * rec_est + w2 * time_interp + w3 * graph_est) / wsum

        cleaned_scaled = x_scaled.copy()
        replace_mask = np.isin(labels, ["device_gap", "spike", "drift", "stuck", "cross_sensor_conflict", "noise"])
        cleaned_scaled[replace_mask] = fused_scaled[replace_mask]
        cleaned = inverse_scale(cleaned_scaled, self.artifacts.median, self.artifacts.iqr)

        label_df = pd.DataFrame(labels, columns=self.artifacts.sensor_names)
        score_df = pd.DataFrame(total_score, columns=self.artifacts.sensor_names)
        point_status_df = pd.DataFrame(
            {
                "timestamp": timestamps.values,
                "abnormal_count": (label_df != "normal").sum(axis=1).values,
                "point_score_mean": score_df.mean(axis=1).values,
                "point_score_max": score_df.max(axis=1).values,
            }
        )
        point_status_df["point_status"] = np.where(point_status_df["abnormal_count"] > 0, "abnormal", "normal")

        health_rows = []
        bridge_event_counter: Counter = Counter()
        for col in self.artifacts.sensor_names:
            sensor_labels = label_df[col].astype(str).tolist()
            device_health, availability, project_score, dominant_issue, c = calc_health_scores(sensor_labels)
            bridge_event_counter.update(c)
            sensor_id, sensor_type = split_sensor_name(col)
            missing_ratio = (
                c.get("device_gap", 0) + c.get("bridge_wide_gap", 0) + c.get("known_system_gap", 0)
            ) / max(1, len(sensor_labels))
            health_rows.append(
                {
                    "sensor_name": col,
                    "bridge_name": bridge_name,
                    "sensor_id": sensor_id,
                    "sensor_type": sensor_type,
                    "device_health": round(device_health, 3),
                    "availability": round(availability, 3),
                    "project_score": round(project_score, 3),
                    "dominant_issue": dominant_issue,
                    "missing_ratio": round(missing_ratio, 4),
                    "stuck_ratio": round(c.get("stuck", 0) / max(1, len(sensor_labels)), 4),
                    "drift_ratio": round(c.get("drift", 0) / max(1, len(sensor_labels)), 4),
                    "spike_ratio": round((c.get("spike", 0) + c.get("noise", 0)) / max(1, len(sensor_labels)), 4),
                }
            )

        cleaned_df = pd.DataFrame(cleaned, columns=self.artifacts.sensor_names)
        cleaned_df.insert(0, "timestamp", timestamps.values)
        score_df.insert(0, "timestamp", timestamps.values)
        label_df.insert(0, "timestamp", timestamps.values)
        health_df = pd.DataFrame(health_rows).sort_values(by=["project_score"], ascending=[True])

        bridge_device_health = float(health_df["device_health"].mean()) if not health_df.empty else 100.0
        bridge_availability = float(health_df["availability"].mean()) if not health_df.empty else 100.0
        bridge_project_score = 0.7 * bridge_device_health + 0.3 * bridge_availability
        total_points = max(1, raw_missing.size)
        system_missing = bridge_event_counter.get("known_system_gap", 0) + bridge_event_counter.get("bridge_wide_gap", 0)
        device_missing = bridge_event_counter.get("device_gap", 0)

        bridge_metrics_df = pd.DataFrame(
            [
                {
                    "bridge_name": bridge_name,
                    "samples": len(sensor_df),
                    "sensor_count": len(self.artifacts.sensor_names),
                    "total_missing_ratio": round((system_missing + device_missing) / total_points, 4),
                    "system_missing_ratio": round(system_missing / total_points, 4),
                    "device_missing_ratio": round(device_missing / total_points, 4),
                    "avg_device_health": round(bridge_device_health, 3),
                    "avg_availability": round(bridge_availability, 3),
                    "bridge_project_score": round(bridge_project_score, 3),
                }
            ]
        )

        # 这里更诚实地命名：这些都是“按传感器点累计”的数量/时长
        bridge_event_df = pd.DataFrame(
            [
                {
                    "bridge_name": bridge_name,
                    "startup_jump_point_count": int(bridge_event_counter.get("startup_jump", 0)),
                    "known_system_gap_sensor_hours": round(
                        bridge_event_counter.get("known_system_gap", 0) * self.artifacts.sampling_minutes / 60.0, 2
                    ),
                    "device_gap_sensor_hours": round(
                        device_missing * self.artifacts.sampling_minutes / 60.0, 2
                    ),
                    "stuck_point_count": int(bridge_event_counter.get("stuck", 0)),
                    "drift_point_count": int(bridge_event_counter.get("drift", 0)),
                    "step_change_point_count": int(bridge_event_counter.get("step_change", 0)),
                    "spike_noise_point_count": int(
                        bridge_event_counter.get("spike", 0) + bridge_event_counter.get("noise", 0)
                    ),
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

    def save_visualizations(
        self,
        original_df: pd.DataFrame,
        outputs: Dict[str, pd.DataFrame],
        output_dir: str,
        plot_top_k: int = 4,
    ) -> List[str]:
        if self.artifacts is None:
            raise RuntimeError("请先调用 fit()")

        ensure_dir(output_dir)
        saved_paths: List[str] = []

        # 关键修复：可视化必须使用与 transform 同样规则补齐后的时间轴
        aligned_original, _ = align_to_regular_timeline(
            original_df, sampling_minutes=self.artifacts.sampling_minutes
        )
        timestamps, _ = infer_time_column(aligned_original)
        if len(timestamps) != len(aligned_original):
            timestamps = pd.Series(np.arange(len(aligned_original)))

        sensor_df = aligned_original[self.artifacts.sensor_names].copy()
        cleaned_df = outputs["cleaned_data"].copy()
        score_df = outputs["score_data"].copy()
        label_df = outputs["label_data"].copy()
        health_df = outputs["sensor_health_summary"].copy()
        point_status_df = outputs["point_status"].copy()

        def save_and_track(fig: plt.Figure, filename: str) -> None:
            path = os.path.join(output_dir, filename)
            save_figure(fig, path)
            saved_paths.append(path)

        # 1) 训练损失曲线
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(np.arange(1, len(self.artifacts.training_loss_history) + 1), self.artifacts.training_loss_history, linewidth=2.0)
        ax.set_title("Training Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        save_and_track(fig, "training_loss_curve.png")

        # 2) 邻接矩阵热力图（增强对比度 + 高连接标注）
        adj = np.asarray(self.artifacts.adjacency, dtype=float)
        p_low, p_high = np.nanpercentile(adj, [5, 99])
        if not np.isfinite(p_low):
            p_low = float(np.nanmin(adj))
        if not np.isfinite(p_high) or p_high <= p_low:
            p_high = float(np.nanmax(adj) + 1e-6)

        fig, ax = plt.subplots(figsize=(8.6, 7.2))
        im = ax.imshow(adj, aspect="auto", cmap="magma", vmin=p_low, vmax=p_high)
        ax.set_title("Sensor Adjacency Heatmap")
        ax.set_xlabel("Sensor Index")
        ax.set_ylabel("Sensor Index")
        top_thr = float(np.nanpercentile(adj, 99.5))
        idx = np.argwhere(adj >= top_thr)
        if len(idx) > 0:
            idx = idx[: min(80, len(idx))]
            ax.scatter(idx[:, 1], idx[:, 0], s=10, c="cyan", alpha=0.6, label="Top edges")
            ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Adjacency Strength")
        save_and_track(fig, "sensor_adjacency_heatmap.png")

        # 3) 异常评分热力图（分位裁剪 + 等值线）
        score_values = score_df.drop(columns=["timestamp"]).to_numpy(dtype=float).T
        s_low, s_high = np.nanpercentile(score_values, [5, 99])
        if not np.isfinite(s_low):
            s_low = float(np.nanmin(score_values))
        if not np.isfinite(s_high) or s_high <= s_low:
            s_high = float(np.nanmax(score_values) + 1e-6)
        fig, ax = plt.subplots(figsize=(12.5, 6.2))
        im = ax.imshow(score_values, aspect="auto", cmap="turbo", vmin=s_low, vmax=s_high)
        ax.set_title("Anomaly Score Heatmap")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Sensor")
        ax.set_yticks(np.arange(len(self.artifacts.sensor_names)))
        ax.set_yticklabels(self.artifacts.sensor_names)
        lv = np.nanpercentile(score_values, [90, 95, 99])
        lv = np.unique(lv[np.isfinite(lv)])
        if len(lv) > 0:
            ax.contour(score_values, levels=lv, colors="white", linewidths=0.4, alpha=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Score")
        save_and_track(fig, "anomaly_score_heatmap.png")

        # 4) 传感器健康指数条形图
        worst_health = health_df.nsmallest(min(12, len(health_df)), "project_score")
        fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.35 * len(worst_health) + 1.2)))
        sensor_name = worst_health["sensor_name"]
        ax.barh(sensor_name, worst_health["project_score"])
        ax.invert_yaxis()
        ax.set_title("Worst Sensor Project Score")
        ax.set_xlabel("Project Score")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=0.25)
        save_and_track(fig, "sensor_health_barh.png")

        # 5) 逐时刻异常计数曲线
        fig, ax1 = plt.subplots(figsize=(11, 4.8))
        ax1.plot(timestamps, point_status_df["abnormal_count"], linewidth=1.8, label="Abnormal count")
        ax1.set_title("Abnormal Count Over Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Abnormal Sensor Count")
        ax1.grid(alpha=0.25)
        style_time_axis(ax1, timestamps)

        ax2 = ax1.twinx()
        ax2.plot(timestamps, point_status_df["point_score_mean"], linestyle="--", linewidth=1.4, label="Mean score")
        ax2.set_ylabel("Mean Score")
        save_and_track(fig, "abnormal_count_over_time.png")

        # 6) 原始-清洗对比图
        top_sensors = health_df["sensor_name"].head(min(plot_top_k, len(health_df))).tolist()
        nrows = len(top_sensors)
        if nrows == 0:
            return saved_paths

        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, max(3.2 * nrows, 3.5)), sharex=True)
        if nrows == 1:
            axes = [axes]

        label_colors = {
            "device_gap": "black",
            "bridge_wide_gap": "gray",
            "known_system_gap": "steelblue",
            "spike": "red",
            "noise": "pink",
            "drift": "orange",
            "stuck": "purple",
            "step_change": "green",
            "startup_jump": "cyan",
            "cross_sensor_conflict": "brown",
        }

        ts_array = np.array(timestamps)

        for ax, sensor in zip(axes, top_sensors):
            raw_series = sensor_df[sensor].to_numpy(dtype=float)
            cleaned_series = cleaned_df[sensor].to_numpy(dtype=float)
            sensor_labels = label_df[sensor].astype(str).to_numpy()

            ax.plot(timestamps, raw_series, linewidth=1.0, alpha=0.75, label="Raw")
            ax.plot(timestamps, cleaned_series, linewidth=1.6, alpha=0.95, label="Cleaned")

            for lb, color in label_colors.items():
                mask = sensor_labels == lb
                if np.any(mask):
                    ax.scatter(ts_array[mask], cleaned_series[mask], s=14, c=color, label=lb, alpha=0.85)

            ax.set_title(f"{sensor}: Raw vs Cleaned")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.22)

        style_time_axis(axes[-1], timestamps)
        handles, labels_ = axes[0].get_legend_handles_labels()
        uniq = dict(zip(labels_, handles))
        axes[0].legend(uniq.values(), uniq.keys(), ncol=min(6, len(uniq)), fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Time")
        save_and_track(fig, "raw_vs_cleaned_top_sensors.png")

        # 7) 全通道时程图 + 异常状态矩阵（借鉴论文风格的上下对照图）
        all_sensors = self.artifacts.sensor_names
        n_all = len(all_sensors)
        if n_all > 0:
            fig_h = max(9.0, 1.15 * n_all + 4.8)
            fig = plt.figure(figsize=(16, fig_h))
            gs = fig.add_gridspec(2, 1, height_ratios=[max(1.6, n_all * 0.2), 1.0], hspace=0.12)
            gs_top = gs[0].subgridspec(n_all, 1, hspace=0.03)
            top_axes = [fig.add_subplot(gs_top[i, 0]) for i in range(n_all)]
            ax_bottom = fig.add_subplot(gs[1], sharex=top_axes[-1])

            time_num = mdates.date2num(pd.to_datetime(timestamps))
            if np.any(np.isfinite(time_num)):
                xmin, xmax = float(np.nanmin(time_num)), float(np.nanmax(time_num))
            else:
                xmin, xmax = 0.0, float(len(timestamps) - 1)
                time_num = np.arange(len(timestamps), dtype=float)

            # (a) 顶图：每个通道单独绘制，按通道上下排列（非叠加）
            for i, sensor in enumerate(all_sensors):
                raw_series = sensor_df[sensor].to_numpy(dtype=float)
                clean_series = cleaned_df[sensor].to_numpy(dtype=float)
                ax = top_axes[i]
                ax.plot_date(time_num, raw_series, "-", lw=0.55, alpha=0.35, color="#4c72b0", label="Raw")
                ax.plot_date(time_num, clean_series, "-", lw=0.8, alpha=0.95, color="#2f2f2f", label="Cleaned")
                ax.grid(True, axis="x", alpha=0.2, linestyle="--")
                ax.grid(True, axis="y", alpha=0.12)
                ax.set_ylabel(sensor, rotation=0, labelpad=28, fontsize=7)
                for sp in ax.spines.values():
                    sp.set_linewidth(0.9)
                if i < n_all - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                if i == 0:
                    ax.set_title("All-channel Time Series (one panel per sensor, Raw/Cleaned)")
                    ax.legend(loc="upper right", fontsize=7, ncol=2)

            # (b) 底图：离散异常状态矩阵
            label_matrix = label_df[all_sensors].astype(str).to_numpy(dtype=object).T
            status_order = [
                "normal",
                "device_gap",
                "bridge_wide_gap",
                "known_system_gap",
                "spike",
                "noise",
                "drift",
                "stuck",
                "step_change",
                "startup_jump",
                "cross_sensor_conflict",
            ]
            status_colors = {
                "normal": "#6DBE45",
                "device_gap": "#F5F5F5",
                "bridge_wide_gap": "#E8E8E8",
                "known_system_gap": "#D5EEF7",
                "spike": "#FF1A1A",
                "noise": "#69B9C9",
                "drift": "#3E59A8",
                "stuck": "#B455A0",
                "step_change": "#F39C12",
                "startup_jump": "#8E44AD",
                "cross_sensor_conflict": "#7F8C8D",
            }
            status_zh = {
                "normal": "正常",
                "device_gap": "设备缺测",
                "bridge_wide_gap": "全桥缺测",
                "known_system_gap": "已知系统离线",
                "spike": "突刺",
                "noise": "噪声",
                "drift": "漂移",
                "stuck": "卡滞",
                "step_change": "阶跃",
                "startup_jump": "启动跳变",
                "cross_sensor_conflict": "跨传感器冲突",
            }
            status_to_code = {s: i for i, s in enumerate(status_order)}
            S = np.vectorize(lambda x: status_to_code.get(str(x), status_to_code["cross_sensor_conflict"]))(label_matrix)

            cmap = mcolors.ListedColormap([status_colors[s] for s in status_order])
            norm = mcolors.BoundaryNorm(np.arange(-0.5, len(status_order) + 0.5, 1), cmap.N)
            ax_bottom.imshow(
                S,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
                extent=[xmin, xmax, 0.5, n_all + 0.5],
                origin="lower",
            )
            ax_bottom.set_yticks(np.arange(1, n_all + 1))
            ax_bottom.set_yticklabels(all_sensors, fontsize=7)
            ax_bottom.set_ylabel("Sensor")
            ax_bottom.set_title("Sensor Status Matrix")
            ax_bottom.grid(True, axis="x", alpha=0.22, linestyle="--")
            for sp in ax_bottom.spines.values():
                sp.set_linewidth(1.0)

            handles = [
                plt.Line2D([0], [0], color=status_colors[s], lw=7, label=f"{status_zh[s]}({s})")
                for s in status_order
            ]
            ax_bottom.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.22),
                ncol=4,
                frameon=True,
                fontsize=8,
                columnspacing=0.8,
                handlelength=1.8,
            )

            style_time_axis(top_axes[-1], timestamps)
            style_time_axis(ax_bottom, timestamps)
            top_axes[-1].set_xlim(xmin, xmax)
            ax_bottom.set_xlabel("Time")
            top_axes[0].text(-0.06, 1.08, "(a)", transform=top_axes[0].transAxes, fontsize=12, fontweight="bold")
            ax_bottom.text(-0.06, -0.18, "(b)", transform=ax_bottom.transAxes, fontsize=12, fontweight="bold")

            save_and_track(fig, "all_channels_status_overview.png")

        # 8) 潜空间散点图
        x_raw = sensor_df.to_numpy(dtype=float)
        x_interp = pd.DataFrame(x_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy()
        x_scaled = (x_interp - self.artifacts.median) / self.artifacts.iqr
        _, _, _, x_input = self._prepare_model_input(x_scaled, self.artifacts.adjacency)
        windows, spans = self._make_windows(x_input)
        if len(windows) == 0:
            return saved_paths

        _, latent = self._encode_windows(windows)

        if latent.shape[1] > 2:
            latent_2d = PCA(n_components=2, random_state=42).fit_transform(latent)
        elif latent.shape[1] == 2:
            latent_2d = latent
        else:
            latent_2d = np.column_stack([latent[:, 0], np.zeros(len(latent))])

        window_abnormal = []
        point_abnormal_mask = (label_df.drop(columns=["timestamp"]) != "normal").to_numpy(dtype=bool)
        for s, e in spans:
            window_abnormal.append(point_abnormal_mask[s:e].any())
        window_abnormal = np.array(window_abnormal, dtype=bool)

        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        sc = ax.scatter(
            latent_2d[:, 0],
            latent_2d[:, 1],
            c=self.artifacts.latent_window_score,
            s=np.where(window_abnormal, 42, 18),
            alpha=0.85,
        )
        if np.any(window_abnormal):
            ax.scatter(
                latent_2d[window_abnormal, 0],
                latent_2d[window_abnormal, 1],
                s=56,
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
                label="Abnormal window",
            )
            ax.legend(loc="best")
        ax.set_title("Latent Window Distribution")
        ax.set_xlabel("Latent Axis 1")
        ax.set_ylabel("Latent Axis 2")
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Window density score")
        save_and_track(fig, "latent_window_distribution.png")

        return saved_paths


# -----------------------------
# Demo 数据
# -----------------------------
def make_demo_data(n_steps: int = 400, n_sensors: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2025-01-01", periods=n_steps, freq="5min")

    temperature = 15 + 8 * np.sin(np.linspace(0, 4 * np.pi, n_steps))
    wind = 3 + np.sin(np.linspace(0, 7 * np.pi, n_steps))

    data = {}
    positions = []
    for i in range(n_sensors):
        base = 100 + 0.5 * i
        signal = base + 0.15 * temperature + 0.35 * np.sin(np.linspace(0, 8 * np.pi, n_steps) + i / 3.0)
        signal += 0.08 * wind + rng.normal(scale=0.08 + i * 0.005, size=n_steps)
        data[f"S{i+1:02d}"] = signal
        positions.append({"sensor": f"S{i+1:02d}", "x": i * 8.0, "y": 0.0, "z": 0.0})

    df = pd.DataFrame(data)
    df.insert(0, "timestamp", ts)

    # 注入异常：缺失、突刺、漂移、卡滞
    df.loc[50:60, "S02"] = np.nan
    df.loc[120, "S04"] += 8.0
    df.loc[121, "S04"] -= 6.0
    df.loc[180:240, "S06"] = df.loc[180:240, "S06"] + np.linspace(0, 2.5, 61)
    df.loc[280:320, "S08"] = float(df.loc[279, "S08"])
    df.loc[330:335, "S10"] = np.nan

    pos_df = pd.DataFrame(positions)
    return df, pos_df


def read_input(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] > 1:
        first_col = df.columns[0]
        try:
            df[first_col] = pd.to_datetime(df[first_col], errors="ignore")
        except Exception:
            pass
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="桥梁健康监测系统无监督数据预处理原型")
    parser.add_argument("--input_csv", type=str, default=None, help="输入监测数据 csv")
    parser.add_argument("--positions_csv", type=str, default=None, help="传感器位置 csv，可选")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录")
    parser.add_argument("--demo", action="store_true", help="使用内置模拟数据演示")
    parser.add_argument("--window_size", type=int, default=12, help="滑动窗口长度")
    parser.add_argument("--stride", type=int, default=1, help="滑动步长")
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数")
    parser.add_argument("--latent_dim", type=int, default=32, help="潜变量维数")
    parser.add_argument("--plot_top_k", type=int, default=4, help="原始/清洗对比图中展示的传感器数量")
    parser.add_argument("--skip_plots", action="store_true", help="仅导出 csv，不生成 png 图件")
    args = parser.parse_args()

    set_seed(42)
    ensure_dir(args.output_dir)

    if args.demo or args.input_csv is None:
        df, pos_df = make_demo_data()
        bridge_name = "demo_bridge"
    else:
        df = read_input(args.input_csv)
        pos_df = pd.read_csv(args.positions_csv) if args.positions_csv else None
        bridge_name = infer_bridge_name_from_path(args.input_csv)

    pre = BridgeSHMUnsupervisedPreprocessor(
        window_size=args.window_size,
        stride=args.stride,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
    )
    pre.fit(df, positions=pos_df)
    outputs = pre.transform(df, bridge_name=bridge_name)
    print_chinese_anomaly_summary(outputs)

    for name, out_df in outputs.items():
        out_path = os.path.join(args.output_dir, f"{name}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    plot_files: List[str] = []
    if not args.skip_plots:
        plot_files = pre.save_visualizations(df, outputs, args.output_dir, plot_top_k=args.plot_top_k)

    print("处理完成。输出文件：")
    for name in outputs:
        print(f" - {os.path.join(args.output_dir, f'{name}.csv')}")
    for path in plot_files:
        print(f" - {path}")


if __name__ == "__main__":
    main()
