# EXE 打包说明

软件名：**桥梁健康监测系统传感器健康状态无监督可视化分析软件**

## 1. 安装依赖

```bash
pip install pyinstaller pandas numpy matplotlib scikit-learn torch openpyxl
```

## 2. 打包命令

```bash
pyinstaller --noconfirm --clean --windowed \
  --name "桥梁健康监测系统传感器健康状态无监督可视化分析软件" \
  run_gui_app.py
```

## 3. 运行

打包完成后运行：

- `dist/桥梁健康监测系统传感器健康状态无监督可视化分析软件/桥梁健康监测系统传感器健康状态无监督可视化分析软件.exe`

## 4. 功能说明

- 离线分析：加载多个 CSV，统一时段分析，多桥对比。
- 在线分析：读取 `mapping/*.xls*` 的桥名，并调用 `data-export-csv.py` 导出 Doris 数据后分析。
- 支持分析时段：1 天 / 7 天 / 自定义。
- 支持按桥分别选择传感器。
- 支持两种分析方式：普通预处理分析、时空图+多置信融合分析。
