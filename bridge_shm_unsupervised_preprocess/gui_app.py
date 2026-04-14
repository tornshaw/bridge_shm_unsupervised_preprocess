from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import pandas as pd

from .online_data import (
    fetch_bridge_data_online,
    find_mapping_file,
    load_bridge_names_from_mapping,
)

APP_NAME = "桥梁健康监测系统传感器健康状态无监督可视化分析软件"


class BridgeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1250x760")

        self.source_mode = tk.StringVar(value="offline")
        self.analysis_mode = tk.StringVar(value="时空图+多置信融合分析")
        self.period_mode = tk.StringVar(value="7d")
        self.sample_days_var = tk.StringVar(value="1")
        self.start_var = tk.StringVar(value="2026-03-01 00:00:00")
        self.end_var = tk.StringVar(value="2026-03-07 23:59:59")
        self.output_dir_var = tk.StringVar(value="outputs/gui_analysis")
        self.export_script_var = tk.StringVar(value="data-export-csv.py")

        self.offline_csvs: List[str] = []
        self.online_bridge_names: List[str] = []
        self.bridge_sensor_select: Dict[str, List[str]] = {}
        self.bridge_sensors_all: Dict[str, List[str]] = {}
        self.current_bridge_name: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(top, text=APP_NAME, font=("Microsoft YaHei", 14, "bold")).pack(side=tk.LEFT)

        cfg = ttk.LabelFrame(self, text="数据源与分析配置")
        cfg.pack(fill=tk.X, padx=10, pady=6)

        ttk.Radiobutton(cfg, text="离线CSV", variable=self.source_mode, value="offline").grid(row=0, column=0, padx=8, pady=6)
        ttk.Radiobutton(cfg, text="在线Doris", variable=self.source_mode, value="online").grid(row=0, column=1, padx=8, pady=6)

        ttk.Button(cfg, text="加载离线CSV(可多选)", command=self._add_offline_csvs).grid(row=0, column=2, padx=8, pady=6)
        ttk.Button(cfg, text="读取mapping桥名", command=self._load_online_bridges).grid(row=0, column=3, padx=8, pady=6)

        ttk.Label(cfg, text="导出脚本:").grid(row=0, column=4, padx=6, pady=6, sticky="e")
        ttk.Entry(cfg, textvariable=self.export_script_var, width=28).grid(row=0, column=5, padx=6, pady=6)

        ttk.Label(cfg, text="分析方式:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        mode_box = ttk.Combobox(cfg, textvariable=self.analysis_mode, values=["普通预处理分析", "时空图+多置信融合分析"], width=24, state="readonly")
        mode_box.grid(row=1, column=1, padx=6, pady=6)

        ttk.Label(cfg, text="样本窗口(天):").grid(row=1, column=2, padx=6, pady=6, sticky="e")
        ttk.Entry(cfg, textvariable=self.sample_days_var, width=8).grid(row=1, column=3, padx=6, pady=6, sticky="w")

        ttk.Label(cfg, text="输出目录:").grid(row=1, column=4, padx=6, pady=6, sticky="e")
        ttk.Entry(cfg, textvariable=self.output_dir_var, width=28).grid(row=1, column=5, padx=6, pady=6)

        period = ttk.LabelFrame(self, text="分析时段（多桥统一时段）")
        period.pack(fill=tk.X, padx=10, pady=6)

        ttk.Radiobutton(period, text="最近1天", variable=self.period_mode, value="1d", command=self._apply_period_preset).grid(row=0, column=0, padx=8, pady=6)
        ttk.Radiobutton(period, text="最近7天", variable=self.period_mode, value="7d", command=self._apply_period_preset).grid(row=0, column=1, padx=8, pady=6)
        ttk.Radiobutton(period, text="自选", variable=self.period_mode, value="custom").grid(row=0, column=2, padx=8, pady=6)

        ttk.Label(period, text="开始:").grid(row=0, column=3, padx=6, pady=6, sticky="e")
        ttk.Entry(period, textvariable=self.start_var, width=20).grid(row=0, column=4, padx=6, pady=6)
        ttk.Label(period, text="结束:").grid(row=0, column=5, padx=6, pady=6, sticky="e")
        ttk.Entry(period, textvariable=self.end_var, width=20).grid(row=0, column=6, padx=6, pady=6)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        left = ttk.LabelFrame(body, text="桥梁列表（多选）")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.bridge_list = tk.Listbox(left, selectmode=tk.EXTENDED, width=42, height=24)
        self.bridge_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.bridge_list.bind("<<ListboxSelect>>", self._on_bridge_select)

        right = ttk.LabelFrame(body, text="传感器选择（当前桥）")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.current_bridge_label = ttk.Label(right, text="当前桥: -")
        self.current_bridge_label.pack(anchor="w", padx=6, pady=4)

        btn_bar = ttk.Frame(right)
        btn_bar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(btn_bar, text="全选", command=self._select_all_sensors).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="清空", command=self._clear_sensors).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="保存当前桥传感器选择", command=self._save_sensor_selection).pack(side=tk.LEFT, padx=8)

        self.sensor_list = tk.Listbox(right, selectmode=tk.MULTIPLE, width=70, height=22)
        self.sensor_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        run_bar = ttk.Frame(self)
        run_bar.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(run_bar, text="开始分析", command=self._run).pack(side=tk.RIGHT, padx=6)

    def _apply_period_preset(self) -> None:
        now = pd.Timestamp.now()
        if self.period_mode.get() == "1d":
            start = now - pd.Timedelta(days=1)
        elif self.period_mode.get() == "7d":
            start = now - pd.Timedelta(days=7)
        else:
            return
        self.start_var.set(start.strftime("%Y-%m-%d %H:%M:%S"))
        self.end_var.set(now.strftime("%Y-%m-%d %H:%M:%S"))

    def _add_offline_csvs(self) -> None:
        files = filedialog.askopenfilenames(filetypes=[("CSV", "*.csv")])
        if not files:
            return
        for f in files:
            if f not in self.offline_csvs:
                self.offline_csvs.append(f)
        self._refresh_bridge_list()

    def _load_online_bridges(self) -> None:
        mapping = find_mapping_file("mapping")
        if not mapping:
            messagebox.showwarning("提示", "未找到 mapping/*.xls* 文件。")
            return
        try:
            self.online_bridge_names = load_bridge_names_from_mapping(mapping)
            self._refresh_bridge_list()
            messagebox.showinfo("成功", f"已加载在线桥梁数量: {len(self.online_bridge_names)}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def _refresh_bridge_list(self) -> None:
        self.bridge_list.delete(0, tk.END)
        if self.source_mode.get() == "offline":
            for p in self.offline_csvs:
                self.bridge_list.insert(tk.END, os.path.basename(p))
        else:
            for b in self.online_bridge_names:
                self.bridge_list.insert(tk.END, b)

    def _on_bridge_select(self, _event=None) -> None:
        sel = self.bridge_list.curselection()
        if not sel:
            return
        idx = sel[0]
        bridge = self.bridge_list.get(idx)
        self.current_bridge_name = bridge
        self.current_bridge_label.config(text=f"当前桥: {bridge}")

        sensors = self.bridge_sensors_all.get(bridge)
        if sensors is None:
            sensors = self._load_sensors_for_bridge(bridge)
            self.bridge_sensors_all[bridge] = sensors

        self.sensor_list.delete(0, tk.END)
        for s in sensors:
            self.sensor_list.insert(tk.END, s)

        selected = set(self.bridge_sensor_select.get(bridge, sensors))
        for i, s in enumerate(sensors):
            if s in selected:
                self.sensor_list.selection_set(i)

    def _load_sensors_for_bridge(self, bridge: str) -> List[str]:
        if self.source_mode.get() == "offline":
            path = next((p for p in self.offline_csvs if os.path.basename(p) == bridge), None)
            if not path:
                return []
            df = pd.read_csv(path, nrows=5)
            return list(df.columns[1:])
        return []

    def _select_all_sensors(self) -> None:
        self.sensor_list.selection_set(0, tk.END)

    def _clear_sensors(self) -> None:
        self.sensor_list.selection_clear(0, tk.END)

    def _save_sensor_selection(self) -> None:
        bridge = self.current_bridge_name
        if not bridge:
            messagebox.showwarning("提示", "请先选中一个桥")
            return
        idxs = self.sensor_list.curselection()
        sensors = [self.sensor_list.get(i) for i in idxs]
        self.bridge_sensor_select[bridge] = sensors
        messagebox.showinfo("成功", f"已保存 {bridge} 传感器选择: {len(sensors)} 个")

    def _run(self) -> None:
        try:
            start = pd.to_datetime(self.start_var.get())
            end = pd.to_datetime(self.end_var.get())
            sample_days = float(self.sample_days_var.get())
            if end < start:
                raise ValueError("结束时间不能早于开始时间")
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        selected_idx = self.bridge_list.curselection()
        if not selected_idx:
            messagebox.showwarning("提示", "请至少选择一座桥")
            return

        output_root = self.output_dir_var.get().strip() or "outputs/gui_analysis"
        os.makedirs(output_root, exist_ok=True)

        try:
            from .app_service import build_offline_tasks, run_multi_bridge_tasks

            if self.source_mode.get() == "offline":
                selected_files = [self.offline_csvs[i] for i in selected_idx]
                tasks = build_offline_tasks(selected_files, self.bridge_sensor_select)
            else:
                export_script = self.export_script_var.get().strip()
                tasks = []
                for i in selected_idx:
                    bridge = self.online_bridge_names[i]
                    csv_out = os.path.join(output_root, f"{bridge}_online.csv")
                    fetch_bridge_data_online(
                        bridge_name=bridge,
                        start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
                        end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
                        output_csv=csv_out,
                        export_script=export_script,
                    )
                    tasks.extend(build_offline_tasks([csv_out], self.bridge_sensor_select))

            summary = run_multi_bridge_tasks(
                tasks=tasks,
                output_root=output_root,
                analysis_mode=self.analysis_mode.get(),
                start_time=start,
                end_time=end,
                sample_days=sample_days,
            )
            messagebox.showinfo("完成", f"分析完成。桥梁数={len(summary)}\n输出目录: {output_root}")
        except Exception as e:
            messagebox.showerror("分析失败", str(e))


def main() -> None:
    app = BridgeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
