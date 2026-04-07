
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

依赖：
    numpy, pandas, scikit-learn, torch

输入：
    - csv 文件：第一列可为时间戳，其余列为传感器数值
    - 可选 positions csv：包含 [sensor, x, y, z] 或 [sensor, x, y]
输出：
    - cleaned_data.csv        修复后的数据
    - score_data.csv          综合质量评分
    - label_data.csv          异常标签
    - sensor_health.csv       传感器健康摘要
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# 工具函数
# -----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    score = score.reshape(-1, 1)
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

        xs = np.linspace(score.min(), score.max(), 2000).reshape(-1, 1)
        p1 = w1 * (1 / (np.sqrt(2 * np.pi) * s1)) * np.exp(-0.5 * ((xs - m1) / s1) ** 2)
        p2 = w2 * (1 / (np.sqrt(2 * np.pi) * s2)) * np.exp(-0.5 * ((xs - m2) / s2) ** 2)
        cross_idx = np.argmin(np.abs(p1 - p2))
        tau = float(xs[cross_idx, 0])
        if not np.isfinite(tau):
            raise ValueError("invalid tau")
        return tau
    except Exception:
        s = score.flatten()
        return float(np.median(s) + 3.0 * safe_mad(s))


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



class FitArtifacts:
    def __init__(
        self,
        median: np.ndarray,
        iqr: np.ndarray,
        adjacency: np.ndarray,
        window_size: int,
        stride: int,
        latent_windows: np.ndarray,
        latent_window_score: np.ndarray,
        sensor_names: List[str],
        threshold: float,
    ):
        self.median = median
        self.iqr = iqr
        self.adjacency = adjacency
        self.window_size = window_size
        self.stride = stride
        self.latent_windows = latent_windows
        self.latent_window_score = latent_window_score
        self.sensor_names = sensor_names
        self.threshold = threshold

# -----------------------------
# 主类
# -----------------------------
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
        # x shape: [T, M]
        return x @ adjacency.T

    def _make_windows(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        windows = []
        spans = []
        for start in range(0, x.shape[0] - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(x[start:end].reshape(-1))
            spans.append((start, end))
        return np.array(windows, dtype=np.float32), spans

    def fit(self, df: pd.DataFrame, positions: Optional[pd.DataFrame] = None) -> "BridgeSHMUnsupervisedPreprocessor":
        sensor_df = df.select_dtypes(include=[np.number]).copy()
        sensor_names = list(sensor_df.columns)
        x_raw = sensor_df.to_numpy(dtype=float)
        missing_mask = np.isnan(x_raw)

        # 初始插值，保留缺失掩码
        x_filled = pd.DataFrame(x_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy()
        x_scaled, median, iqr = robust_center_scale(x_filled)
        adjacency = self._build_graph(x_scaled, sensor_names, positions)

        x_trend = np.apply_along_axis(moving_average_1d, 0, x_scaled, max(3, self.window_size // 2))
        x_res = x_scaled - x_trend
        x_graph = self._graph_smooth(x_res, adjacency)
        x_input = np.concatenate([x_res, x_graph], axis=1)

        windows, spans = self._make_windows(x_input)
        input_dim = windows.shape[1]

        self.model = MaskedDenoisingAutoencoder(input_dim=input_dim, latent_dim=self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = TensorDataset(torch.from_numpy(windows))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        adjacency_tensor = torch.from_numpy(adjacency.astype(np.float32)).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                noise = 0.02 * torch.randn_like(batch)
                mask = (torch.rand_like(batch) > self.mask_ratio).float()
                masked_batch = batch * mask + noise

                recon, z = self.model(masked_batch)

                # 重构损失
                loss_rec = torch.mean((recon - batch) ** 2)

                # 一致性损失：增强前后潜在表示接近
                aug = batch + 0.01 * torch.randn_like(batch)
                _, z_aug = self.model(aug)
                loss_con = torch.mean((z - z_aug) ** 2)

                # 平滑损失：鼓励潜在空间稳定
                loss_latent = torch.mean(z[:, 1:] ** 2) if z.shape[1] > 1 else torch.mean(z ** 2)

                loss = loss_rec + 0.15 * loss_con + 0.001 * loss_latent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

        # 计算潜在窗口评分
        self.model.eval()
        with torch.no_grad():
            windows_tensor = torch.from_numpy(windows).to(self.device)
            recon_windows, latent_windows_t = self.model(windows_tensor)
            latent_windows = latent_windows_t.cpu().numpy()
            recon_windows = recon_windows.cpu().numpy()

        nn_k = min(8, len(latent_windows))
        nbrs = NearestNeighbors(n_neighbors=nn_k)
        nbrs.fit(latent_windows)
        dists, _ = nbrs.kneighbors(latent_windows)
        latent_window_score = dists[:, -1]

        dummy_score = np.zeros(len(df), dtype=float)
        counts = np.zeros(len(df), dtype=float)
        for idx, (s, e) in enumerate(spans):
            dummy_score[s:e] += latent_window_score[idx]
            counts[s:e] += 1
        latent_point_score = dummy_score / np.clip(counts, 1, None)
        threshold = infer_threshold_gmm(latent_point_score)

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
        )
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if self.model is None or self.artifacts is None:
            raise RuntimeError("请先调用 fit()")

        sensor_df = df[self.artifacts.sensor_names].copy()
        timestamps = df.iloc[:, 0].copy() if not np.issubdtype(df.iloc[:, 0].dtype, np.number) else pd.Series(np.arange(len(df)))
        x_raw = sensor_df.to_numpy(dtype=float)
        raw_missing = np.isnan(x_raw)

        x_interp = pd.DataFrame(x_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy()
        x_scaled = (x_interp - self.artifacts.median) / self.artifacts.iqr
        x_trend = np.apply_along_axis(moving_average_1d, 0, x_scaled, max(3, self.window_size // 2))
        x_res = x_scaled - x_trend
        x_graph = self._graph_smooth(x_res, self.artifacts.adjacency)
        x_input = np.concatenate([x_res, x_graph], axis=1)

        windows, spans = self._make_windows(x_input)
        with torch.no_grad():
            windows_t = torch.from_numpy(windows).to(self.device)
            recon_windows_t, latent_t = self.model(windows_t)
            recon_windows = recon_windows_t.cpu().numpy()
            latent = latent_t.cpu().numpy()

        # 窗口重构聚合回逐点
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

        # 四类点级评分
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

        # 自适应权重：由评分离散度决定
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

        # 异常分类
        labels = np.full(x_scaled.shape, "normal", dtype=object)
        labels[raw_missing] = "missing"

        # 突刺 spike
        first_diff = np.abs(np.diff(x_scaled, axis=0, prepend=x_scaled[[0], :]))
        diff_thr = np.nanpercentile(first_diff, 95)
        spike_mask = (total_score > tau) & (first_diff > diff_thr)
        labels[spike_mask & (~raw_missing)] = "spike"

        # 漂移 drift
        drift_labels = np.zeros_like(x_scaled, dtype=bool)
        run = max(5, self.window_size)
        for j in range(x_scaled.shape[1]):
            resid = x_scaled[:, j] - recon_scaled[:, j]
            bias = pd.Series(resid).rolling(run, min_periods=run // 2).mean().fillna(0.0).to_numpy()
            sign_consistency = pd.Series(np.sign(bias)).rolling(run, min_periods=run // 2).apply(lambda v: np.abs(np.mean(v)), raw=True).fillna(0.0).to_numpy()
            bias_thr = np.nanpercentile(np.abs(bias), 90)
            drift_labels[:, j] = (np.abs(bias) > bias_thr) & (sign_consistency > 0.8) & (total_score[:, j] > tau * 0.8)
        labels[drift_labels & (~raw_missing) & (labels == "normal")] = "drift"

        # 卡滞 stuck
        stuck_labels = np.zeros_like(x_scaled, dtype=bool)
        for j in range(x_scaled.shape[1]):
            rs = rolling_std_1d(x_scaled[:, j], max(5, self.window_size))
            thr = np.nanpercentile(rs, 10)
            flat = rs <= max(thr, 1e-4)
            stuck_labels[:, j] = flat & (total_score[:, j] > tau * 0.6)
        labels[stuck_labels & (~raw_missing) & (labels == "normal")] = "stuck"

        # 剩余异常归为 graph_inconsistent
        labels[(total_score > tau) & (labels == "normal")] = "graph_inconsistent"

        # 修复
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
        replace_mask = np.isin(labels, ["missing", "spike", "drift", "stuck", "graph_inconsistent"])
        cleaned_scaled[replace_mask] = fused_scaled[replace_mask]
        cleaned = inverse_scale(cleaned_scaled, self.artifacts.median, self.artifacts.iqr)

        # 健康摘要
        label_df = pd.DataFrame(labels, columns=self.artifacts.sensor_names)
        score_df = pd.DataFrame(total_score, columns=self.artifacts.sensor_names)
        health_rows = []
        n = len(label_df)
        for col in self.artifacts.sensor_names:
            abnormal_ratio = float((label_df[col] != "normal").mean())
            health_index = max(0.0, 100.0 * (1.0 - abnormal_ratio))
            top_label = label_df[col].value_counts().idxmax()
            health_rows.append(
                {
                    "sensor": col,
                    "health_index": round(health_index, 3),
                    "abnormal_ratio": round(abnormal_ratio, 4),
                    "dominant_label": top_label,
                    "mean_score": round(float(score_df[col].mean()), 6),
                }
            )

        cleaned_df = pd.DataFrame(cleaned, columns=self.artifacts.sensor_names)
        cleaned_df.insert(0, "timestamp", timestamps.values)
        score_df.insert(0, "timestamp", timestamps.values)
        label_df.insert(0, "timestamp", timestamps.values)
        health_df = pd.DataFrame(health_rows).sort_values(by=["health_index", "abnormal_ratio"], ascending=[True, False])

        return {
            "cleaned_data": cleaned_df,
            "score_data": score_df,
            "label_data": label_df,
            "sensor_health": health_df,
        }


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
    # 尝试解析第一列时间
    if df.shape[1] > 1:
        first_col = df.columns[0]
        try:
            parsed = pd.to_datetime(df[first_col])
            df[first_col] = parsed.astype(str)
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
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.demo or args.input_csv is None:
        df, pos_df = make_demo_data()
    else:
        df = read_input(args.input_csv)
        pos_df = pd.read_csv(args.positions_csv) if args.positions_csv else None

    pre = BridgeSHMUnsupervisedPreprocessor(
        window_size=args.window_size,
        stride=args.stride,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
    )
    pre.fit(df, positions=pos_df)
    outputs = pre.transform(df)

    for name, out_df in outputs.items():
        out_path = os.path.join(args.output_dir, f"{name}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("处理完成。输出文件：")
    for name in outputs:
        print(f" - {os.path.join(args.output_dir, f'{name}.csv')}")


if __name__ == "__main__":
    main()
