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
    load_point_name_mapping,
    test_doris_connection,
)

APP_NAME = "桥梁健康监测系统传感器健康状态无监督可视化分析软件"
INDICATOR_MAP = {
    "WSD": "温湿度",
    "YB": "主梁应变",
    "QJ": "桥墩倾角",
    "LX": "梁墩相对位移",
    "BB": "主梁挠度",
    "ZL": "主梁竖向振动",
    "QD": "桥墩三向振动",
}


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
        self.db_host_var = tk.StringVar(value="10.172.121.139")
        self.db_port_var = tk.StringVar(value="9030")
        self.db_user_var = tk.StringVar(value="root")
        self.db_password_var = tk.StringVar(value="8j2q8nWs0u7ZogoDzMxa")
        self.db_name_var = tk.StringVar(value="CITYLL_DW_DWD")
        self.db_status_var = tk.StringVar(value="未测试")
        self.online_extract_start_var = tk.StringVar(value="2026-03-01 00:00:00")
        self.online_extract_end_var = tk.StringVar(value="2026-03-07 23:59:59")

        self.offline_csvs: List[str] = []
        self.online_bridge_names: List[str] = []
        self.online_bridge_sensors: Dict[str, List[str]] = {}
        self.point_name_map: Dict[str, str] = {}
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
        self.node_sensor_id: Dict[str, str] = {}
        self.image_zoom: float = 1.0
        self.last_analysis_output_dir: Optional[str] = None
        self.history_table_df = pd.DataFrame()

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
        self.btn_load_online = ttk.Button(online_cfg, text="读取mapping桥名", command=self._load_online_bridges)
        self.btn_load_online.grid(row=1, column=0, padx=8, pady=4)
        ttk.Label(online_cfg, text="提取开始").grid(row=1, column=1, padx=5, pady=4, sticky="e")
        self.online_extract_start_entry = ttk.Entry(online_cfg, textvariable=self.online_extract_start_var, width=20)
        self.online_extract_start_entry.grid(row=1, column=2, padx=5, pady=4)
        ttk.Label(online_cfg, text="提取结束").grid(row=1, column=3, padx=5, pady=4, sticky="e")
        self.online_extract_end_entry = ttk.Entry(online_cfg, textvariable=self.online_extract_end_var, width=20)
        self.online_extract_end_entry.grid(row=1, column=4, padx=5, pady=4)
        self.btn_extract_online = ttk.Button(online_cfg, text="提取数据", command=self._extract_online_data)
        self.btn_extract_online.grid(row=1, column=5, padx=8, pady=4)

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

        left = ttk.LabelFrame(body, text="桥梁-测点树")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        tree_wrap = ttk.Frame(left)
        tree_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.bridge_tree = ttk.Treeview(tree_wrap, columns=("type",), show="tree", height=26, selectmode="extended")
        tree_scroll_y = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.bridge_tree.yview)
        tree_scroll_x = ttk.Scrollbar(tree_wrap, orient=tk.HORIZONTAL, command=self.bridge_tree.xview)
        self.bridge_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.bridge_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        tree_wrap.rowconfigure(0, weight=1)
        tree_wrap.columnconfigure(0, weight=1)
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

        right = ttk.LabelFrame(body, text="分析图片")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        nav = ttk.Frame(right)
        nav.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(nav, text="普通预处理分析", command=lambda: self._run("普通预处理分析")).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="时空图+多置信融合分析", command=lambda: self._run("时空图+多置信融合分析")).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="清除分析结果", command=self._clear_analysis_results).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="上一张", command=lambda: self._switch_image(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="下一张", command=lambda: self._switch_image(1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav, text="新窗口打开", command=self._open_image_window).pack(side=tk.LEFT, padx=4)
        self.image_title = ttk.Label(nav, text="暂无图片")
        self.image_title.pack(side=tk.LEFT, padx=8)
        img_wrap = ttk.Frame(right)
        img_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.image_canvas = tk.Canvas(img_wrap, bg="white")
        img_scroll_y = ttk.Scrollbar(img_wrap, orient=tk.VERTICAL, command=self.image_canvas.yview)
        img_scroll_x = ttk.Scrollbar(img_wrap, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        self.image_canvas.configure(yscrollcommand=img_scroll_y.set, xscrollcommand=img_scroll_x.set)
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        img_scroll_y.grid(row=0, column=1, sticky="ns")
        img_scroll_x.grid(row=1, column=0, sticky="ew")
        img_wrap.rowconfigure(0, weight=1)
        img_wrap.columnconfigure(0, weight=1)
        self.image_label = ttk.Label(self.image_canvas)
        self.image_canvas_window = self.image_canvas.create_window((0, 0), window=self.image_label, anchor="nw")
        self.image_label.bind("<Configure>", lambda _e: self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all")))
        self.image_canvas.bind("<Configure>", lambda _e: self._render_current_image())
        self.image_canvas.bind("<Enter>", lambda _e: self.image_canvas.focus_set())
        self.image_label.bind("<Enter>", lambda _e: self.image_canvas.focus_set())
        self.bridge_tree.bind("<Button-1>", self._on_tree_click, add="+")
        self.image_canvas.bind("<Control-MouseWheel>", self._on_image_zoom_wheel)
        self.image_label.bind("<Control-MouseWheel>", self._on_image_zoom_wheel)
        self._bind_mousewheel(self.bridge_tree, self.bridge_tree)
        self._bind_mousewheel(self.image_canvas, self.image_canvas)

        bottom = ttk.LabelFrame(self, text="分析结果表格与日志")
        bottom.pack(fill=tk.BOTH, padx=10, pady=6)
        table_wrap = ttk.Frame(bottom)
        table_wrap.pack(fill=tk.X, padx=6, pady=4)
        self.summary_table = ttk.Treeview(table_wrap, show="headings", height=6)
        table_scroll_y = ttk.Scrollbar(table_wrap, orient=tk.VERTICAL, command=self.summary_table.yview)
        table_scroll_x = ttk.Scrollbar(table_wrap, orient=tk.HORIZONTAL, command=self.summary_table.xview)
        self.summary_table.configure(yscrollcommand=table_scroll_y.set, xscrollcommand=table_scroll_x.set)
        self.summary_table.grid(row=0, column=0, sticky="nsew")
        table_scroll_y.grid(row=0, column=1, sticky="ns")
        table_scroll_x.grid(row=1, column=0, sticky="ew")
        table_wrap.rowconfigure(0, weight=1)
        table_wrap.columnconfigure(0, weight=1)
        log_wrap = ttk.Frame(bottom)
        log_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.result_text = tk.Text(log_wrap, height=6)
        log_scroll_y = ttk.Scrollbar(log_wrap, orient=tk.VERTICAL, command=self.result_text.yview)
        log_scroll_x = ttk.Scrollbar(log_wrap, orient=tk.HORIZONTAL, command=self.result_text.xview)
        self.result_text.configure(yscrollcommand=log_scroll_y.set, xscrollcommand=log_scroll_x.set, wrap=tk.NONE)
        self.result_text.grid(row=0, column=0, sticky="nsew")
        log_scroll_y.grid(row=0, column=1, sticky="ns")
        log_scroll_x.grid(row=1, column=0, sticky="ew")
        log_wrap.rowconfigure(0, weight=1)
        log_wrap.columnconfigure(0, weight=1)
        self.result_text.insert(tk.END, "等待分析...\n")
        self.result_text.configure(state=tk.DISABLED)
        self._bind_mousewheel(self.summary_table, self.summary_table)
        self._bind_mousewheel(self.result_text, self.result_text)
        self._on_mode_change()

    def _set_result(self, text: str, append: bool = False) -> None:
        self.result_text.configure(state=tk.NORMAL)
        if not append:
            self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)

    def _bind_mousewheel(self, widget, target) -> None:
        def _on_wheel(event):
            delta = int(-1 * (event.delta / 120))
            try:
                target.yview_scroll(delta, "units")
            except Exception:
                pass
            return "break"

        def _on_shift_wheel(event):
            delta = int(-1 * (event.delta / 120))
            try:
                target.xview_scroll(delta, "units")
            except Exception:
                pass
            return "break"

        widget.bind("<MouseWheel>", _on_wheel, add="+")
        widget.bind("<Shift-MouseWheel>", _on_shift_wheel, add="+")

    def _on_image_zoom_wheel(self, event) -> str:
        if event.delta > 0:
            self.image_zoom = min(3.0, self.image_zoom * 1.1)
        else:
            self.image_zoom = max(0.3, self.image_zoom / 1.1)
        self._render_current_image()
        return "break"

    @staticmethod
    def _to_chinese_summary(summary: pd.DataFrame, analysis_time: str, method: str) -> pd.DataFrame:
        rename_map = {
            "bridge_name": "桥名",
            "samples": "样本数",
            "sensor_count": "传感器数量",
            "total_missing_ratio": "总缺失比例",
            "system_missing_ratio": "系统缺失比例",
            "device_missing_ratio": "设备缺失比例",
            "avg_device_health": "平均设备健康指数",
            "avg_availability": "平均系统可用性",
            "bridge_project_score": "桥梁传感器总分数",
        }
        out = summary.copy()
        out.insert(0, "分析时间", analysis_time)
        out.insert(1, "分析方法", method)
        out = out.rename(columns=rename_map)
        return out

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
                w = max(200, self.image_canvas.winfo_width() - 20)
                h = max(160, self.image_canvas.winfo_height() - 20)
                im.thumbnail((w, h))
                if self.image_zoom != 1.0:
                    zw = max(1, int(im.width * self.image_zoom))
                    zh = max(1, int(im.height * self.image_zoom))
                    im = im.resize((zw, zh))
                self.result_photo = ImageTk.PhotoImage(im)
            else:
                self.result_photo = tk.PhotoImage(file=path)
            self.image_label.config(image=self.result_photo, text="")
            self.image_canvas.itemconfigure(self.image_canvas_window, width=self.result_photo.width(), height=self.result_photo.height())
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
        except Exception:
            self.image_label.config(image="", text=f"无法显示图片: {path}")

    def _switch_image(self, step: int) -> None:
        if not self.result_images:
            return
        self.result_image_index = (self.result_image_index + step) % len(self.result_images)
        self._render_current_image()

    def _open_image_window(self) -> None:
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            messagebox.showwarning("提示", "当前没有可打开的图片。")
            return
        win = tk.Toplevel(self)
        win.title(os.path.basename(self.current_image_path))
        win.geometry("1200x800")
        scale_var = tk.DoubleVar(value=1.0)
        top = ttk.Frame(win)
        top.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(top, text="缩放").pack(side=tk.LEFT)
        ttk.Scale(top, from_=0.2, to=3.0, variable=scale_var, orient=tk.HORIZONTAL, length=280).pack(side=tk.LEFT, padx=6)
        canvas = tk.Canvas(win, bg="white")
        sy = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
        sx = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sy.pack(fill=tk.Y, side=tk.RIGHT)
        sx.pack(fill=tk.X, side=tk.BOTTOM)
        lbl = ttk.Label(canvas)
        cwin = canvas.create_window((0, 0), window=lbl, anchor="nw")

        def redraw(*_):
            try:
                if Image is None or ImageTk is None:
                    photo = tk.PhotoImage(file=self.current_image_path)
                else:
                    im = Image.open(self.current_image_path)
                    w, h = im.size
                    ratio = max(0.2, float(scale_var.get()))
                    im = im.resize((max(1, int(w * ratio)), max(1, int(h * ratio))))
                    photo = ImageTk.PhotoImage(im)
                lbl.image = photo
                lbl.config(image=photo)
                canvas.itemconfig(cwin, width=photo.width(), height=photo.height())
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception as e:
                messagebox.showerror("错误", f"打开图片失败: {e}")

        def on_ctrl_wheel(event):
            if event.delta > 0:
                scale_var.set(min(3.0, float(scale_var.get()) * 1.1))
            else:
                scale_var.set(max(0.2, float(scale_var.get()) / 1.1))
            return "break"

        self._bind_mousewheel(canvas, canvas)
        canvas.bind("<Enter>", lambda _e: canvas.focus_set())
        canvas.bind("<Control-MouseWheel>", on_ctrl_wheel, add="+")
        scale_var.trace_add("write", redraw)
        redraw()

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
            self.point_name_map = load_point_name_mapping(mapping)
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

    @staticmethod
    def _parse_sensor_base(sensor_name: str) -> tuple[str, str]:
        base = sensor_name.split("_", 1)[0]
        suffix = sensor_name.split("_", 1)[1] if "_" in sensor_name else ""
        base5 = "-".join(base.split("-")[:5]) if "-" in base else base
        return base5, suffix

    def _sensor_indicator_code(self, sensor_name: str) -> str:
        base5, _ = self._parse_sensor_base(sensor_name)
        parts = base5.split("-")
        return parts[3] if len(parts) >= 4 else "OTHER"

    def _sensor_display_name(self, sensor_name: str) -> str:
        base5, suffix = self._parse_sensor_base(sensor_name)
        point_cn = self.point_name_map.get(base5, base5)
        return f"{point_cn}_{suffix}" if suffix else point_cn

    def _refresh_bridge_tree_from_items(self) -> None:
        for item in self.bridge_tree.get_children():
            self.bridge_tree.delete(item)
        self.selected_bridge_indices = []
        self.tree_to_bridge = {}
        self.node_checked = {}
        self.node_raw_text = {}
        self.node_sensor_id = {}
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
            group: Dict[str, List[str]] = {}
            for s in sensors:
                code = self._sensor_indicator_code(s)
                group.setdefault(code, []).append(s)
            for code, sensor_list in sorted(group.items(), key=lambda x: x[0]):
                ind_name = INDICATOR_MAP.get(code, code)
                ind_node = self.bridge_tree.insert(node_id, tk.END, text=f"☐ {ind_name}({code})", open=True)
                self.node_raw_text[ind_node] = f"{ind_name}({code})"
                self.node_checked[ind_node] = False
                for s in sensor_list:
                    display = self._sensor_display_name(s)
                    leaf = self.bridge_tree.insert(ind_node, tk.END, text=f"☐ {display}")
                    self.node_raw_text[leaf] = display
                    self.node_sensor_id[leaf] = s
                    self.node_checked[leaf] = False

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
        # 下行联动：父节点勾选影响全部子孙节点
        def set_descendants(nid: str, val: bool) -> None:
            for ch in self.bridge_tree.get_children(nid):
                self._set_node_checked(ch, val)
                set_descendants(ch, val)

        set_descendants(node, checked)

        # 上行联动：子节点变化反推父节点状态
        parent = self.bridge_tree.parent(node)
        while parent:
            siblings = self.bridge_tree.get_children(parent)
            all_checked = all(self.node_checked.get(ch, False) for ch in siblings) if siblings else False
            self._set_node_checked(parent, all_checked)
            parent = self.bridge_tree.parent(parent)
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
            n = 0
            for ind in self.bridge_tree.get_children(node):
                for leaf in self.bridge_tree.get_children(ind):
                    if self.node_checked.get(leaf, False):
                        n += 1
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
            root = node
            while self.bridge_tree.parent(root) != "":
                root = self.bridge_tree.parent(root)
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
            for ind in self.bridge_tree.get_children(root):
                for leaf in self.bridge_tree.get_children(ind):
                    if self.node_checked.get(leaf, False):
                        sensor_name = self.node_sensor_id.get(leaf, "").strip()
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
        target = self.last_analysis_output_dir
        if target and os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True)
        self.history_table_df = pd.DataFrame()
        self._show_summary_table(self.history_table_df)
        self.result_images = []
        self.image_zoom = 1.0
        self._render_current_image()
        self._set_result("分析结果已清除。\n", append=False)
        messagebox.showinfo("成功", f"已清理当次分析结果目录: {target or '无'}")

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
        run_tag = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        session_output_root = os.path.join(output_root, f"run_{run_tag}")
        os.makedirs(session_output_root, exist_ok=True)
        self.image_zoom = 1.0

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
                    csv_out = os.path.join(session_output_root, f"{bridge}_online.csv")
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
                output_root=session_output_root,
                analysis_mode=analysis_mode,
                start_time=start,
                end_time=end,
                sample_days=sample_days,
            )
            self.analysis_mode.set(analysis_mode)
            now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            cn_summary = self._to_chinese_summary(summary, analysis_time=now_str, method=analysis_mode)
            self.history_table_df = pd.concat([self.history_table_df, cn_summary], ignore_index=True)
            result = (
                f"分析完成时间: {now_str}\n"
                f"分析方法: {analysis_mode}\n"
                f"桥梁数: {len(summary)}\n"
                f"输出目录: {session_output_root}\n\n"
            )
            result += cn_summary.to_string(index=False)
            result += "\n\n" + ("-" * 80) + "\n"
            self._set_result(result, append=True)
            self._show_summary_table(self.history_table_df)
            first_bridge = summary.iloc[0]["bridge_name"] if not summary.empty and "bridge_name" in summary.columns else None
            self._load_result_images(output_root=session_output_root, bridge_name=first_bridge)
            self.last_analysis_output_dir = session_output_root
            messagebox.showinfo("完成", f"分析完成。桥梁数={len(summary)}\n输出目录: {session_output_root}")
        except Exception as e:
            messagebox.showerror("分析失败", str(e))


def main() -> None:
    app = BridgeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
