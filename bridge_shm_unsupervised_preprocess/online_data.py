from __future__ import annotations

import glob
import os
import subprocess
from typing import List, Optional

import pandas as pd


def find_mapping_file(mapping_dir: str = "mapping") -> Optional[str]:
    cands = sorted(glob.glob(os.path.join(mapping_dir, "*.xls*")))
    return cands[0] if cands else None


def load_bridge_names_from_mapping(mapping_file: str) -> List[str]:
    df = pd.read_excel(mapping_file)
    for col in ["bridge_name", "桥梁名称", "bridge", "桥名"]:
        if col in df.columns:
            return sorted({str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()})
    # fallback: 第一列
    first = df.columns[0]
    return sorted({str(x).strip() for x in df[first].dropna().tolist() if str(x).strip()})


def fetch_bridge_data_with_export_script(
    script_path: str,
    bridge_name: str,
    start_time: str,
    end_time: str,
    output_csv: str,
) -> str:
    """
    调用 data-export-csv.py 导出 Doris 数据。
    约定脚本支持参数：--bridge --start --end --output
    """
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
