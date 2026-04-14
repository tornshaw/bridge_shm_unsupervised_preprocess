from __future__ import annotations

import os
import re
import shutil
import glob
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import pandas as pd
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None

from .online_data import (
    fetch_bridge_data_online,
    find_mapping_file,
    load_bridge_names_from_mapping,
    load_bridge_sensors_from_mapping,
    test_doris_connection,
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
        self.db_host_var = tk.StringVar(value="localhost")
        self.db_port_var = tk.StringVar(value="9030")
        self.db_user_var = tk.StringVar(value="root")
        self.db_password_var = tk.StringVar(value="")
        self.db_name_var = tk.StringVar(value="")
        self.db_status_var = tk.StringVar(value="未测试")
        self.online_extract_start_var = tk.StringVar(value="2026-03-01 00:00:00")
        self.online_extract_end_var = tk.StringVar(value="2026-03-07 23:59:59")

        self.offline_csvs: List[str] = []
        self.online_bridge_names: List[str] = []
        self.online_bridge_sensors: Dict[str, List[str]] = {}
        self.bridge_sensor_select: Dict[str, List[str]] = {}
        self.bridge_sensors_all: Dict[str, List[str]] = {}
        self.bridge_items: List[Dict[str, str]] = []
        self.selected_bridge_indices: List[int] = []
        self.current_bridge_name: Optional[str] = None
        self.tree_to_bridge: Dict[str, str] = {}
        self.result_images: List[str] = []
        self.result_image_index: int = 0
        self.result_photo = None
        self.current_image_path: Optional[str] = None
        self.node_checked: Dict[str, bool] = {}
        self.node_raw_text: Dict[str, str] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(top, text=APP_NAME, font=("Microsoft YaHei", 14, "bold")).pack(side=tk.LEFT)

        cfg = ttk.LabelFrame(self, text="数据源与分析配置")
        cfg.pack(fill=tk.X, padx=10, pady=6)

        ttk.Radiobutton(cfg, text="离线CSV", variable=self.source_mode, value="offline", command=self._on_mode_change).grid(row=0, column=0, padx=8, pady=6)
        ttk.Radiobutton(cfg, text="在线Doris", variable=self.source_mode, value="online", command=self._on_mode_change).grid(row=0, column=1, padx=8, pady=6)

        self.btn_add_offline = ttk.Button(cfg, text="加载离线CSV(可多选)", command=self._add_offline_csvs)
        self.btn_add_offline.grid(row=0, column=2, padx=8, pady=6)
        self.btn_load_online = ttk.Button(cfg, text="读取mapping桥名", command=self._load_online_bridges)
        self.btn_load_online.grid(row=0, column=3, padx=8, pady=6)
        ttk.Button(cfg, text="清除已选数据", command=self._clear_selected_data).grid(row=0, column=4, padx=8, pady=6)

        ttk.Label(cfg, text="导出脚本:").grid(row=0, column=5, padx=6, pady=6, sticky="e")
        self.entry_export = ttk.Entry(cfg, textvariable=self.export_script_var, width=26)
        self.entry_export.grid(row=0, column=6, padx=6, pady=6)

        ttk.Label(cfg, text="分析方式:").grid(row=1, column=0, padx=6, pady=6, sticky="e")
        mode_box = ttk.Combobox(cfg, textvariable=self.analysis_mode, values=["普通预处理分析", "时空图+多置信融合分析"], width=24, state="readonly")
        mode_box.grid(row=1, column=1, padx=6, pady=6)

        ttk.Label(cfg, text="样本窗口(天):").grid(row=1, column=2, padx=6, pady=6, sticky="e")
        ttk.Entry(cfg, textvariable=self.sample_days_var, width=8).grid(row=1, column=3, padx=6, pady=6, sticky="w")

        ttk.Label(cfg, text="输出目录:").grid(row=1, column=4, padx=6, pady=6, sticky="e")
        ttk.Entry(cfg, textvariable=self.output_dir_var, width=26).grid(row=1, column=5, padx=6, pady=6)

        online_cfg = ttk.LabelFrame(self, text="在线 Doris 连接（在线模式生效）")
        online_cfg.pack(fill=tk.X, padx=10, pady=6)
        ttk.Label(online_cfg, text="Host").grid(row=0, column=0, padx=5, pady=4, sticky="e")
        self.entry_host = ttk.Entry(online_cfg, textvariable=self.db_host_var, width=18)
        self.entry_host.grid(row=0, column=1, padx=5, pady=4)
        ttk.Label(online_cfg, text="Port").grid(row=0, column=2, padx=5, pady=4, sticky="e")
        self.entry_port = ttk.Entry(online_cfg, textvariable=self.db_port_var, width=8)
        self.entry_port.grid(row=0, column=3, padx=5, pady=4)
        ttk.Label(online_cfg, text="User").grid(row=0, column=4, padx=5, pady=4, sticky="e")
        self.entry_user = ttk.Entry(online_cfg, textvariable=self.db_user_var, width=12)
        self.entry_user.grid(row=0, column=5, padx=5, pady=4)
        ttk.Label(online_cfg, text="Password").grid(row=0, column=6, padx=5, pady=4, sticky="e")
        self.entry_pwd = ttk.Entry(online_cfg, textvariable=self.db_password_var, width=14, show="*")
        self.entry_pwd.grid(row=0, column=7, padx=5, pady=4)
        ttk.Label(online_cfg, text="DB").grid(row=0, column=8, padx=5, pady=4, sticky="e")
        self.entry_db = ttk.Entry(online_cfg, textvariable=self.db_name_var, width=14)
        self.entry_db.grid(row=0, column=9, padx=5, pady=4)
        self.btn_test_conn = ttk.Button(online_cfg, text="数据库连接", command=self._test_db_connection)
        self.btn_test_conn.grid(row=0, column=10, padx=8, pady=4)
        ttk.Label(online_cfg, textvariable=self.db_status_var).grid(row=0, column=11, padx=8, pady=4, sticky="w")
        ttk.Label(online_cfg, text="提取开始").grid(row=1, column=0, padx=5, pady=4, sticky="e")
        self.online_extract_start_entry = ttk.Entry(online_cfg, textvariable=self.online_extract_start_var, width=20)
        self.online_extract_start_entry.grid(row=1, column=1, padx=5, pady=4)
        ttk.Label(online_cfg, text="提取结束").grid(row=1, column=2, padx=5, pady=4, sticky="e")
        self.online_extract_end_entry = ttk.Entry(online_cfg, textvariable=self.online_extract_end_var, width=20)
        self.online_extract_end_entry.grid(row=1, column=3, padx=5, pady=4)
        self.btn_extract_online = ttk.Button(online_cfg, text="提取数据", command=self._extract_online_data)
        self.btn_extract_online.grid(row=1, column=4, padx=8, pady=4)

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

        left = ttk.LabelFrame(body, text="桥梁-测点树（左）")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.bridge_tree = ttk.Treeview(left, columns=("type",), show="tree", height=26, selectmode="extended")
        self.bridge_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.bridge_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        mid = ttk.LabelFrame(body, text="当前选择信息")
        mid.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=8)

        self.current_bridge_label = ttk.Label(mid, text="当前桥: -")
        self.current_bridge_label.pack(anchor="w", padx=6, pady=4)
        self.current_range_label = ttk.Label(mid, text="当前数据时间范围: -")
        self.current_range_label.pack(anchor="w", padx=6, pady=2)
        ttk.Label(mid, text="提示：在左侧树中可直接多选桥梁与测点。").pack(anchor="w", padx=6, pady=6)
        self.selected_info_label = ttk.Label(mid, text="已选设备信息：暂无")
        self.selected_info_label.pack(anchor="w", padx=6, pady=4)

        right = ttk.LabelFrame(body, text="分析图片（右）")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        nav = ttk.Frame(right)
        nav.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(nav, text="普通预处理分析", command=lambda: self._run("普通预处理分析")).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="时空图+多置信融合分析", command=lambda: self._run("时空图+多置信融合分析")).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="清除分析结果", command=self._clear_analysis_results).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="上一张", command=lambda: self._switch_image(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="下一张", command=lambda: self._switch_image(1)).pack(side=tk.LEFT, padx=4)
        self.image_title = ttk.Label(nav, text="暂无图片")
        self.image_title.pack(side=tk.LEFT, padx=8)
        self.image_label = ttk.Label(right)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.image_label.bind("<Configure>", lambda _e: self._render_current_image())
        self.bridge_tree.bind("<Button-1>", self._on_tree_click, add="+")

        bottom = ttk.LabelFrame(self, text="分析结果表格与日志（下）")
        bottom.pack(fill=tk.BOTH, padx=10, pady=6)
        self.summary_table = ttk.Treeview(bottom, show="headings", height=6)
        self.summary_table.pack(fill=tk.X, padx=6, pady=4)
        self.result_text = tk.Text(bottom, height=6)
        self.result_text.pack(fill=tk.BOTH, padx=6, pady=4)
        self.result_text.insert(tk.END, "等待分析...\n")
        self.result_text.configure(state=tk.DISABLED)
        self._on_mode_change()

    def _set_result(self, text: str) -> None:
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)

    def _show_summary_table(self, summary: pd.DataFrame) -> None:
        for c in self.summary_table["columns"]:
            self.summary_table.heading(c, text="")
        self.summary_table.delete(*self.summary_table.get_children())
        if summary is None or summary.empty:
            self.summary_table["columns"] = ()
            return
        cols = list(summary.columns)
        self.summary_table["columns"] = cols
        for c in cols:
            self.summary_table.heading(c, text=c)
            self.summary_table.column(c, width=120, anchor="center")
        for _, row in summary.iterrows():
            self.summary_table.insert("", tk.END, values=[row.get(c, "") for c in cols])

    def _load_result_images(self, output_root: str, bridge_name: Optional[str] = None) -> None:
        target_dir = output_root
        if bridge_name:
            cand = os.path.join(output_root, bridge_name)
            if os.path.isdir(cand):
                target_dir = cand
        self.result_images = sorted(glob.glob(os.path.join(target_dir, "*.png")))
        self.result_image_index = 0
        self._render_current_image()

    def _render_current_image(self) -> None:
        if not self.result_images:
            self.image_title.config(text="暂无图片")
            self.image_label.config(image="", text="暂无图片")
            self.result_photo = None
            self.current_image_path = None
            return
        path = self.result_images[self.result_image_index]
        self.current_image_path = path
        self.image_title.config(text=os.path.basename(path))
        try:
            if Image is not None and ImageTk is not None:
                im = Image.open(path)
                w = max(200, self.image_label.winfo_width() - 10)
                h = max(160, self.image_label.winfo_height() - 10)
                im.thumbnail((w, h))
                self.result_photo = ImageTk.PhotoImage(im)
            else:
                self.result_photo = tk.PhotoImage(file=path)
            self.image_label.config(image=self.result_photo, text="")
        except Exception:
            self.image_label.config(image="", text=f"无法显示图片: {path}")

    def _switch_image(self, step: int) -> None:
        if not self.result_images:
            return
        self.result_image_index = (self.result_image_index + step) % len(self.result_images)
        self._render_current_image()

    def _on_mode_change(self) -> None:
        offline = self.source_mode.get() == "offline"
        self.btn_add_offline.config(state=tk.NORMAL if offline else tk.DISABLED)
        for w in [
            self.btn_load_online,
            self.entry_export,
            self.btn_test_conn,
            self.entry_host,
            self.entry_port,
            self.entry_user,
            self.entry_pwd,
            self.entry_db,
            self.online_extract_start_entry,
            self.online_extract_end_entry,
            self.btn_extract_online,
        ]:
            w.config(state=tk.DISABLED if offline else tk.NORMAL)
        self._refresh_bridge_list()

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
            self.online_bridge_sensors = load_bridge_sensors_from_mapping(mapping)
            self._refresh_bridge_list()
            messagebox.showinfo("成功", f"已加载在线桥梁数量: {len(self.online_bridge_names)}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def _test_db_connection(self) -> None:
        ok, msg = test_doris_connection(
            host=self.db_host_var.get().strip(),
            port=int(self.db_port_var.get().strip() or "9030"),
            user=self.db_user_var.get().strip(),
            password=self.db_password_var.get(),
            database=self.db_name_var.get().strip(),
        )
        self.db_status_var.set(msg)
        if ok:
            messagebox.showinfo("连接测试", msg)
        else:
            messagebox.showwarning("连接测试", msg)

    def _extract_online_data(self) -> None:
        if ("成功" not in self.db_status_var.get()) and ("端口可达" not in self.db_status_var.get()):
            messagebox.showwarning("提示", "请先点击“数据库连接”。")
            return
        if not self.online_bridge_names:
            messagebox.showwarning("提示", "请先点击“读取mapping桥名”。")
            return
        selected_idx, _ = self._collect_selection_from_tree()
        if not selected_idx:
            messagebox.showwarning("提示", "请先在左侧树中选择桥梁。")
            return
        try:
            start = pd.to_datetime(self.online_extract_start_var.get())
            end = pd.to_datetime(self.online_extract_end_var.get())
        except Exception as e:
            messagebox.showerror("参数错误", f"提取时间区间无效: {e}")
            return
        if end < start:
            messagebox.showerror("参数错误", "提取结束时间不能早于开始时间")
            return

        output_root = self.output_dir_var.get().strip() or "outputs/gui_analysis"
        export_script = self.export_script_var.get().strip()
        extracted_items: List[Dict[str, str]] = []
        for i in selected_idx:
            bridge = self.bridge_items[i]["bridge_name"]
            csv_out = os.path.join(output_root, f"{bridge}_online.csv")
            fetch_bridge_data_online(
                bridge_name=bridge,
                start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
                output_csv=csv_out,
                export_script=export_script,
            )
            extracted_items.append(
                {
                    "bridge_name": bridge,
                    "date_range": f"{start.strftime('%Y%m%d')}~{end.strftime('%Y%m%d')}",
                    "csv_path": csv_out,
                }
            )
            # 提取后用实际 CSV 列刷新测点
            self.bridge_sensors_all[bridge] = self._load_sensors_from_csv(csv_out)
        self.bridge_items = extracted_items
        self._refresh_bridge_tree_from_items()
        messagebox.showinfo("成功", f"在线提取完成，桥梁数={len(extracted_items)}")

    def _clear_selected_data(self) -> None:
        self.offline_csvs.clear()
        self.online_bridge_names.clear()
        self.bridge_sensors_all.clear()
        self.bridge_sensor_select.clear()
        self.bridge_items.clear()
        self.selected_bridge_indices.clear()
        self.current_bridge_name = None
        self.current_bridge_label.config(text="当前桥: -")
        self.current_range_label.config(text="当前数据时间范围: -")
        for item in self.bridge_tree.get_children():
            self.bridge_tree.delete(item)
        self._refresh_bridge_list()

    @staticmethod
    def _parse_offline_bridge_info(path: str) -> Dict[str, str]:
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        m = re.match(r"^(?P<bridge>.+?)_(?P<start>\d{8})_(?P<end>\d{8})$", stem)
        if m:
            return {
                "bridge_name": m.group("bridge"),
                "date_range": f"{m.group('start')}~{m.group('end')}",
                "csv_path": path,
            }
        return {"bridge_name": stem, "date_range": "-", "csv_path": path}

    @staticmethod
    def _load_sensors_from_csv(path: str) -> List[str]:
        df = pd.read_csv(path, nrows=5)
        return list(df.columns[1:])

    def _refresh_bridge_tree_from_items(self) -> None:
        for item in self.bridge_tree.get_children():
            self.bridge_tree.delete(item)
        self.selected_bridge_indices = []
        self.tree_to_bridge = {}
        self.node_checked = {}
        self.node_raw_text = {}
        for info in self.bridge_items:
            item_id = self.bridge_tree.insert("", tk.END, text=f"☐ {info['bridge_name']}", open=True)
            self.tree_to_bridge[item_id] = info["bridge_name"]
            self.node_raw_text[item_id] = info["bridge_name"]
            self.node_checked[item_id] = False
        self._refresh_tree_sensors()
        self._update_selected_info()

    def _refresh_bridge_list(self) -> None:
        self.bridge_items = []
        if self.source_mode.get() == "offline":
            for p in self.offline_csvs:
                info = self._parse_offline_bridge_info(p)
                self.bridge_items.append(info)
        else:
            for b in self.online_bridge_names:
                self.bridge_items.append({"bridge_name": b, "date_range": "-", "csv_path": ""})
        self._refresh_bridge_tree_from_items()

    def _refresh_tree_sensors(self) -> None:
        for node_id, bridge in list(self.tree_to_bridge.items()):
            # 清理已有子节点
            for ch in self.bridge_tree.get_children(node_id):
                self.bridge_tree.delete(ch)
            sensors = self.bridge_sensors_all.get(bridge)
            if sensors is None:
                sensors = self._load_sensors_for_bridge(bridge)
                self.bridge_sensors_all[bridge] = sensors
            for s in sensors:
                ch = self.bridge_tree.insert(node_id, tk.END, text=f"☐ {s}")
                self.node_raw_text[ch] = s
                self.node_checked[ch] = False

    def _set_node_checked(self, node: str, checked: bool) -> None:
        self.node_checked[node] = checked
        raw = self.node_raw_text.get(node, self.bridge_tree.item(node, "text").replace("☐", "").replace("☑", "").strip())
        prefix = "☑" if checked else "☐"
        self.bridge_tree.item(node, text=f"{prefix} {raw}")

    def _on_tree_click(self, event) -> None:
        node = self.bridge_tree.identify_row(event.y)
        if not node:
            return
        checked = not self.node_checked.get(node, False)
        self._set_node_checked(node, checked)
        parent = self.bridge_tree.parent(node)
        if parent == "":
            for ch in self.bridge_tree.get_children(node):
                self._set_node_checked(ch, checked)
        else:
            siblings = self.bridge_tree.get_children(parent)
            all_checked = all(self.node_checked.get(ch, False) for ch in siblings)
            self._set_node_checked(parent, all_checked)
        self.bridge_tree.selection_set(node)
        self._update_selected_info()

    def _update_selected_info(self) -> None:
        lines = []
        total_bridges = 0
        total_sensors = 0
        for node, bridge in self.tree_to_bridge.items():
            if not self.node_checked.get(node, False):
                continue
            total_bridges += 1
            chs = self.bridge_tree.get_children(node)
            n = sum(1 for ch in chs if self.node_checked.get(ch, False))
            total_sensors += n
            lines.append(f"{bridge}: {n} 个设备")
        if not lines:
            self.selected_info_label.config(text="已选设备信息：暂无")
        else:
            summary = f"已选设备信息：桥梁 {total_bridges} 座，设备 {total_sensors} 个\n" + "；".join(lines[:4])
            self.selected_info_label.config(text=summary)

    def _on_tree_select(self, _event=None) -> None:
        sel_nodes = self.bridge_tree.selection()
        roots = []
        for node in sel_nodes:
            root = node if self.bridge_tree.parent(node) == "" else self.bridge_tree.parent(node)
            if root not in roots:
                roots.append(root)
        if not roots:
            return
        self.selected_bridge_indices = []
        for rid in roots:
            bridge = self.tree_to_bridge.get(rid)
            if bridge:
                idx = next((i for i, it in enumerate(self.bridge_items) if it["bridge_name"] == bridge), -1)
                if idx >= 0:
                    self.selected_bridge_indices.append(idx)

        idx = self.selected_bridge_indices[0] if self.selected_bridge_indices else -1
        if idx < 0 or idx >= len(self.bridge_items):
            return
        bridge = self.bridge_items[idx]["bridge_name"]
        self.current_bridge_name = bridge
        self.current_bridge_label.config(text=f"当前桥: {bridge}")
        self.current_range_label.config(text=f"当前数据时间范围: {self.bridge_items[idx].get('date_range', '-')}")

        sensors = self.bridge_sensors_all.get(bridge)
        if sensors is None:
            sensors = self._load_sensors_for_bridge(bridge)
            self.bridge_sensors_all[bridge] = sensors

        # 传感器选择在树中完成，不再使用独立列表。

    def _load_sensors_for_bridge(self, bridge: str) -> List[str]:
        if self.source_mode.get() == "offline":
            path = next((it["csv_path"] for it in self.bridge_items if it["bridge_name"] == bridge and it.get("csv_path")), None)
            if not path:
                return []
            return self._load_sensors_from_csv(path)
        path = next((it["csv_path"] for it in self.bridge_items if it["bridge_name"] == bridge and it.get("csv_path")), None)
        if path and os.path.exists(path):
            return self._load_sensors_from_csv(path)
        return self.online_bridge_sensors.get(bridge, [])

    def _collect_selection_from_tree(self) -> tuple[List[int], Dict[str, List[str]]]:
        roots: Dict[str, str] = {}
        picked_sensors: Dict[str, List[str]] = {}
        for root, bridge in self.tree_to_bridge.items():
            if not self.node_checked.get(root, False):
                continue
            roots[root] = bridge
            for ch in self.bridge_tree.get_children(root):
                if self.node_checked.get(ch, False):
                    sensor_name = self.node_raw_text.get(ch, "").strip()
                    picked_sensors.setdefault(bridge, [])
                    if sensor_name and sensor_name not in picked_sensors[bridge]:
                        picked_sensors[bridge].append(sensor_name)

        idxs: List[int] = []
        for bridge in roots.values():
            idx = next((i for i, it in enumerate(self.bridge_items) if it["bridge_name"] == bridge), -1)
            if idx >= 0:
                idxs.append(idx)
                if bridge not in picked_sensors:
                    picked_sensors[bridge] = self.bridge_sensors_all.get(bridge, [])
        return idxs, picked_sensors

    def _clear_analysis_results(self) -> None:
        output_root = self.output_dir_var.get().strip() or "outputs/gui_analysis"
        if os.path.exists(output_root):
            shutil.rmtree(output_root, ignore_errors=True)
        self._show_summary_table(pd.DataFrame())
        self.result_images = []
        self._render_current_image()
        self._set_result("分析结果已清除。\n")
        messagebox.showinfo("成功", f"已清理输出目录: {output_root}")

    def _run(self, analysis_mode: str) -> None:
        try:
            start = pd.to_datetime(self.start_var.get())
            end = pd.to_datetime(self.end_var.get())
            sample_days = float(self.sample_days_var.get())
            if end < start:
                raise ValueError("结束时间不能早于开始时间")
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        selected_idx, picked_sensors = self._collect_selection_from_tree()
        selected_idx = tuple(selected_idx)
        if not selected_idx:
            messagebox.showwarning("提示", "请至少选择一座桥")
            return
        self.bridge_sensor_select.update(picked_sensors)

        output_root = self.output_dir_var.get().strip() or "outputs/gui_analysis"
        os.makedirs(output_root, exist_ok=True)

        try:
            from .app_service import build_offline_tasks, run_multi_bridge_tasks

            if self.source_mode.get() == "offline":
                selected_files = []
                for i in selected_idx:
                    if i < len(self.bridge_items):
                        p = self.bridge_items[i].get("csv_path")
                        if p:
                            selected_files.append(p)
                tasks = build_offline_tasks(selected_files, self.bridge_sensor_select)
            else:
                if ("成功" not in self.db_status_var.get()) and ("端口可达" not in self.db_status_var.get()):
                    raise ValueError("在线模式请先测试数据库连接成功，再执行分析。")
                export_script = self.export_script_var.get().strip()
                tasks = []
                for i in selected_idx:
                    bridge = self.bridge_items[i]["bridge_name"]
                    csv_out = os.path.join(output_root, f"{bridge}_online.csv")
                    fetch_bridge_data_online(
                        bridge_name=bridge,
                        start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
                        end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
                        output_csv=csv_out,
                        export_script=export_script,
                    )
                    tasks.extend(build_offline_tasks([csv_out], self.bridge_sensor_select))
            if not tasks:
                raise ValueError("当前选择未生成可分析任务，请检查桥梁/数据加载状态。")

            summary = run_multi_bridge_tasks(
                tasks=tasks,
                output_root=output_root,
                analysis_mode=analysis_mode,
                start_time=start,
                end_time=end,
                sample_days=sample_days,
            )
            self.analysis_mode.set(analysis_mode)
            show_cols = [c for c in ["bridge_name", "bridge_project_score", "avg_device_health", "avg_availability"] if c in summary.columns]
            result = f"分析完成。模式={analysis_mode}，桥梁数={len(summary)}\n输出目录: {output_root}\n\n"
            if show_cols:
                result += summary[show_cols].to_string(index=False)
            self._set_result(result)
            self._show_summary_table(summary)
            first_bridge = summary.iloc[0]["bridge_name"] if not summary.empty and "bridge_name" in summary.columns else None
            self._load_result_images(output_root=output_root, bridge_name=first_bridge)
            messagebox.showinfo("完成", f"分析完成。桥梁数={len(summary)}\n输出目录: {output_root}")
        except Exception as e:
            messagebox.showerror("分析失败", str(e))


def main() -> None:
    app = BridgeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
