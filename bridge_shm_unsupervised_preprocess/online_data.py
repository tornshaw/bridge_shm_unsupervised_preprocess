from __future__ import annotations

import glob
import os
import socket
import subprocess
from typing import Dict, List, Optional

import pandas as pd


def _pick_bridge_col(df: pd.DataFrame) -> str:
    for col in ["bridge_name", "桥梁名称", "bridge", "桥名", "object_name", "OBJECT_NAME", "结构物名称"]:
        if col in df.columns:
            return col
    # 兜底：优先选择值中“桥”占比高的列（避免选到 object_id）
    best_col = df.columns[0]
    best_ratio = -1.0
    for c in df.columns:
        vals = [str(x).strip() for x in df[c].dropna().tolist() if str(x).strip()]
        if not vals:
            continue
        ratio = sum("桥" in v for v in vals) / len(vals)
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = c
    return best_col


def find_mapping_file(mapping_dir: str = "mapping") -> Optional[str]:
    cands = sorted(glob.glob(os.path.join(mapping_dir, "*.xls*")))
    return cands[0] if cands else None


def load_bridge_names_from_mapping(mapping_file: str) -> List[str]:
    df = pd.read_excel(mapping_file)
    bridge_col = _pick_bridge_col(df)
    return sorted({str(x).strip() for x in df[bridge_col].dropna().tolist() if str(x).strip()})


def load_bridge_sensors_from_mapping(mapping_file: str) -> Dict[str, List[str]]:
    """读取 mapping 表中的桥梁-测点映射，返回 {bridge_name: [sensor, ...]}."""
    df = pd.read_excel(mapping_file)
    bridge_col = _pick_bridge_col(df)
    sensor_col = next(
        (c for c in ["point_code", "测点编号", "sensor_name", "point_name", "测点名称"] if c in df.columns),
        None,
    )
    if sensor_col is None:
        return {}

    out: Dict[str, List[str]] = {}
    for _, row in df[[bridge_col, sensor_col]].dropna().iterrows():
        b = str(row[bridge_col]).strip()
        s = str(row[sensor_col]).strip()
        if not b or not s:
            continue
        out.setdefault(b, [])
        if s not in out[b]:
            out[b].append(s)
    return out


def test_doris_connection(host: str, port: int, user: str, password: str, database: str) -> tuple[bool, str]:
    """测试 Doris 数据库连接状态."""
    try:
        import pymysql  # type: ignore
    except Exception as e:
        # 无驱动时，至少做一次端口联通性测试，避免 GUI 完全不可用
        try:
            with socket.create_connection((host, int(port)), timeout=5):
                return True, f"端口可达({host}:{port})，但缺少 pymysql 依赖: {e}"
        except Exception:
            return False, f"缺少 pymysql 依赖且端口不可达: {e}"

    conn = None
    try:
        conn = pymysql.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=database,
            connect_timeout=5,
            charset="utf8mb4",
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return True, "数据库连接成功"
    except Exception as e:
        return False, f"数据库连接失败: {e}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def fetch_bridge_data_with_export_script(
    script_path: str,
    bridge_name: str,
    start_time: str,
    end_time: str,
    output_csv: str,
) -> str:
    cmd = [
        "python3",
        script_path,
        "--bridge",
        bridge_name,
        "--start",
        start_time,
        "--end",
        end_time,
        "--output",
        output_csv,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"导出失败: {proc.stderr or proc.stdout}")
    if not os.path.exists(output_csv):
        raise FileNotFoundError(f"导出脚本执行成功但未找到输出文件: {output_csv}")
    return output_csv


def fetch_bridge_data_online(
    bridge_name: str,
    start_time: str,
    end_time: str,
    output_csv: str,
    export_script: Optional[str] = None,
) -> str:
    """
    在线取数统一入口：
    1) 优先使用 data-export-csv.py
    2) 失败则回退到 data/ 下同名离线 CSV（便于先跑通流程）
    """
    if export_script and os.path.exists(export_script):
        try:
            return fetch_bridge_data_with_export_script(
                script_path=export_script,
                bridge_name=bridge_name,
                start_time=start_time,
                end_time=end_time,
                output_csv=output_csv,
            )
        except Exception:
            pass

    local_cands = sorted(glob.glob(os.path.join("data", f"{bridge_name}*.csv")))
    if not local_cands:
        raise RuntimeError(
            f"在线导出失败且未找到本地兜底数据: data/{bridge_name}*.csv，请检查导出脚本参数。"
        )

    df = pd.read_csv(local_cands[0], encoding="utf-8-sig")
    t = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    mask = (t >= pd.Timestamp(start_time)) & (t <= pd.Timestamp(end_time))
    out = df.loc[mask].copy()
    if out.empty:
        out = df.copy()
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv
