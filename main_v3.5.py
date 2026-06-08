# -*- coding: utf-8 -*-
"""
混沌置乱表循环阶分析与图像置乱加密演示系统
技术栈：CustomTkinter + NumPy + Pandas + Matplotlib + Pillow

运行方式：
    conda activate chaos_perm
    python main.py

说明：
    1. 本程序使用混沌映射生成长度为 N 的实数序列；
    2. 将序列排序，得到“原位置 -> 排序后位置”的置乱表；
    3. 将置乱表看作一个排列，分解循环圈并计算排列阶；
    4. 支持图像像素位置置乱与逆置乱恢复；
    5. 支持导出 CSV、PNG 图表和实验结果文件。
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import customtkinter as ctk
import numpy as np
import pandas as pd
from PIL import Image
from tkinter import filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


# =========================
# 1. 全局配置
# =========================

APP_NAME = "基于混沌映射的置乱表循环阶分析与图像置乱系统"
EPS = 1e-12

DEFAULT_PARAMS = {
    "Logistic": 3.99,
    "Tent": 1.999,
    "Sine": 0.99,
    "Chebyshev": 4.0,
}

MAP_DESCRIPTIONS = {
    "Logistic": "x(n+1)= μ·x(n)·(1-x(n))，常用混沌区间：3.57 < μ < 4。",
    "Tent": "x(n+1)= μ·x(n) 或 μ·(1-x(n))，常用参数：接近 2。",
    "Sine": "x(n+1)= a·sin(πx(n))，常用参数：接近 1。",
    "Chebyshev": "x(n+1)=(cos(k·arccos(2x(n)-1))+1)/2，常用 k>1。",
}

# 让 Matplotlib 尽量正常显示中文和负号
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 2. 工具函数
# =========================

def get_base_dir() -> Path:
    """返回 main.py 所在目录。"""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def get_results_dir() -> Path:
    """创建并返回 results 目录。"""
    results_dir = get_base_dir() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp_unit_interval(x: float) -> float:
    """避免数值落在 0 或 1 的退化点。"""
    if not np.isfinite(x):
        raise ValueError("混沌迭代出现非有限数值，请检查参数。")
    if x <= 0:
        return EPS
    if x >= 1:
        return 1 - EPS
    return float(x)


def huge_int_log10(n: int) -> float:
    """计算超大整数的 log10，避免直接转 float 溢出。"""
    if n <= 0:
        return float("nan")
    bit_len = n.bit_length()
    if bit_len < 1024:
        return math.log10(n)
    shift = bit_len - 53
    mantissa = n >> shift
    return math.log10(mantissa) + shift * math.log10(2)


def huge_int_display(n: int, max_chars: int = 120) -> str:
    """显示超大整数，过长时只显示位数和首尾片段。"""
    digits_est = int(huge_int_log10(n)) + 1
    if digits_est <= max_chars:
        return str(n)
    s = str(n)
    return f"{digits_est} 位整数，首 40 位：{s[:40]} ... 末 20 位：{s[-20:]}"


def safe_int(value: str, name: str, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        result = int(value)
    except Exception:
        raise ValueError(f"{name} 必须是整数。")
    if min_value is not None and result < min_value:
        raise ValueError(f"{name} 不能小于 {min_value}。")
    if max_value is not None and result > max_value:
        raise ValueError(f"{name} 不能大于 {max_value}。")
    return result


def safe_float(value: str, name: str) -> float:
    try:
        result = float(value)
    except Exception:
        raise ValueError(f"{name} 必须是数字。")
    if not np.isfinite(result):
        raise ValueError(f"{name} 必须是有限数字。")
    return result


# =========================
# 3. 混沌映射与置乱表生成
# =========================

def chaos_next(map_name: str, x: float, param: float) -> float:
    """计算一次混沌映射迭代。"""
    x = clamp_unit_interval(x)

    if map_name == "Logistic":
        if not (0 < param <= 4):
            raise ValueError("Logistic 映射参数 μ 建议满足 0 < μ ≤ 4，混沌区间常取 3.57 < μ < 4。")
        return clamp_unit_interval(param * x * (1 - x))

    if map_name == "Tent":
        if not (0 < param <= 2):
            raise ValueError("Tent 映射参数 μ 建议满足 0 < μ ≤ 2，混沌区间常取接近 2。")
        if x < 0.5:
            return clamp_unit_interval(param * x)
        return clamp_unit_interval(param * (1 - x))

    if map_name == "Sine":
        if not (0 < param <= 1):
            raise ValueError("Sine 映射参数 a 建议满足 0 < a ≤ 1，常取接近 1。")
        return clamp_unit_interval(param * math.sin(math.pi * x))

    if map_name == "Chebyshev":
        if param <= 1:
            raise ValueError("Chebyshev 映射参数 k 建议满足 k > 1，常取 3 或 4。")
        t = max(-1.0, min(1.0, 2 * x - 1))
        y = math.cos(param * math.acos(t))
        return clamp_unit_interval((y + 1) / 2)

    raise ValueError(f"未知映射类型：{map_name}")


def generate_chaos_sequence(
    map_name: str,
    param: float,
    x0: float,
    warmup: int,
    n: int,
) -> np.ndarray:
    """先迭代 warmup 轮去除暂态，再生成长度为 n 的混沌序列。"""
    if not (0 < x0 < 1):
        raise ValueError("初始值 x0 必须满足 0 < x0 < 1。")
    if warmup < 0:
        raise ValueError("预迭代轮数 M 不能为负数。")
    if n <= 1:
        raise ValueError("N 必须大于 1。")

    x = clamp_unit_interval(x0)

    for _ in range(warmup):
        x = chaos_next(map_name, x, param)

    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = chaos_next(map_name, x, param)
        seq[i] = x

    return seq


def sequence_to_permutation(seq: np.ndarray) -> np.ndarray:
    """
    将混沌序列转换为置乱表。

    题目描述：若 x_i 被排序后排在第 j 位，则置乱表中将第 i 个数移至第 j 位。
    因此本函数返回 perm，其中 perm[i] = j。
    内部使用 0 基索引；导出 CSV 时会同时给出 1 基索引。
    """
    n = len(seq)
    original_indices = np.arange(n)
    sorted_indices = np.lexsort((original_indices, seq))  # 先按 seq 排序，若相等按原索引排序
    perm = np.empty(n, dtype=np.int64)
    perm[sorted_indices] = np.arange(n, dtype=np.int64)
    return perm


def validate_permutation(perm: np.ndarray) -> bool:
    n = len(perm)
    return np.array_equal(np.sort(perm), np.arange(n))


# =========================
# 4. 循环圈分析
# =========================

def decompose_cycles(perm: np.ndarray) -> List[List[int]]:
    """将排列 perm 分解为循环圈。perm[i] 表示 i 被映射到的位置。"""
    n = len(perm)
    visited = np.zeros(n, dtype=bool)
    cycles: List[List[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        current = start
        cycle: List[int] = []
        while not visited[current]:
            visited[current] = True
            cycle.append(int(current))
            current = int(perm[current])
        cycles.append(cycle)

    return cycles


def analyze_cycles(perm: np.ndarray) -> Tuple[List[List[int]], Dict[str, object]]:
    """统计循环圈长度分布、最大循环长度和排列阶。"""
    if not validate_permutation(perm):
        raise ValueError("输入不是合法置乱表，无法进行循环圈分析。")

    cycles = decompose_cycles(perm)
    lengths = [len(c) for c in cycles]
    length_counter = Counter(lengths)

    order = 1
    for length in lengths:
        order = math.lcm(order, length)

    stats: Dict[str, object] = {
        "N": len(perm),
        "cycle_count": len(cycles),
        "different_length_count": len(length_counter),
        "max_cycle_length": max(lengths),
        "min_cycle_length": min(lengths),
        "fixed_point_count": length_counter.get(1, 0),
        "order": order,
        "order_log10": huge_int_log10(order),
        "order_digits": int(huge_int_log10(order)) + 1,
        "length_counter": dict(sorted(length_counter.items())),
    }
    return cycles, stats


def format_stats_text(map_name: str, param: float, x0: float, warmup: int, stats: Dict[str, object], elapsed: float) -> str:
    length_counter = stats["length_counter"]
    top_items = sorted(length_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    length_lines = "\n".join([f"  长度 {k}: {v} 个" for k, v in top_items])

    return (
        f"映射类型：{map_name}\n"
        f"参数：{param}\n"
        f"初始值 x0：{x0}\n"
        f"预迭代轮数 M：{warmup}\n"
        f"置乱规模 N：{stats['N']}\n"
        f"\n"
        f"循环圈总数：{stats['cycle_count']}\n"
        f"不同循环长度种类数：{stats['different_length_count']}\n"
        f"最大循环长度：{stats['max_cycle_length']}\n"
        f"最小循环长度：{stats['min_cycle_length']}\n"
        f"不动点数量：{stats['fixed_point_count']}\n"
        f"排列总阶：{huge_int_display(int(stats['order']))}\n"
        f"log10(总阶)：{stats['order_log10']:.4f}\n"
        f"总阶位数：{stats['order_digits']}\n"
        f"运行耗时：{elapsed:.4f} 秒\n"
        f"\n"
        f"循环长度分布 Top 15：\n{length_lines}\n"
    )


# =========================
# 5. 图像置乱与恢复
# =========================

def scramble_image_by_perm(image: Image.Image, perm: np.ndarray) -> Image.Image:
    """按照 perm 执行图像像素位置置乱：原像素 i 移动到 perm[i]。"""
    img = image.convert("RGB")
    arr = np.array(img)
    h, w, c = arr.shape
    n = h * w
    if len(perm) != n:
        raise ValueError(f"置乱表长度 {len(perm)} 与图像像素数 {n} 不一致。")

    flat = arr.reshape(n, c)
    scrambled = np.empty_like(flat)
    scrambled[perm] = flat
    return Image.fromarray(scrambled.reshape(h, w, c), mode="RGB")


def recover_image_by_perm(scrambled_image: Image.Image, perm: np.ndarray) -> Image.Image:
    """按照相同 perm 执行逆置乱恢复：原像素 i = 置乱图像中的 perm[i]。"""
    img = scrambled_image.convert("RGB")
    arr = np.array(img)
    h, w, c = arr.shape
    n = h * w
    if len(perm) != n:
        raise ValueError(f"置乱表长度 {len(perm)} 与图像像素数 {n} 不一致。")

    flat = arr.reshape(n, c)
    recovered = flat[perm]
    return Image.fromarray(recovered.reshape(h, w, c), mode="RGB")


# =========================
# 6. 导出函数
# =========================

def export_permutation_csv(path: Path, seq: np.ndarray, perm: np.ndarray) -> None:
    df = pd.DataFrame({
        "original_index_0_based": np.arange(len(seq)),
        "original_index_1_based": np.arange(1, len(seq) + 1),
        "chaotic_value": seq,
        "scrambled_position_0_based": perm,
        "scrambled_position_1_based": perm + 1,
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")


def export_cycle_stats_csv(path: Path, stats: Dict[str, object]) -> None:
    counter = stats["length_counter"]
    df = pd.DataFrame({
        "cycle_length": list(counter.keys()),
        "cycle_count": list(counter.values()),
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")


def export_summary_txt(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# =========================
# 7. 绘图函数
# =========================

def make_cycle_distribution_figure(stats: Dict[str, object]) -> Figure:
    counter: Dict[int, int] = stats["length_counter"]  # type: ignore
    items = sorted(counter.items(), key=lambda kv: kv[0])

    # 长度种类过多时，只展示数量最多的前 25 类，避免图表过密
    if len(items) > 25:
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:25]
        items = sorted(items, key=lambda kv: kv[0])
        title = "循环长度分布（数量最多的前 25 类）"
    else:
        title = "循环长度分布"

    x_labels = [str(k) for k, _ in items]
    y_values = [v for _, v in items]

    fig = Figure(figsize=(7.2, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(x_labels, y_values)
    ax.set_title(title)
    ax.set_xlabel("循环圈长度")
    ax.set_ylabel("循环圈个数")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def make_n_curve_figure(df: pd.DataFrame, map_name: str) -> Figure:
    fig = Figure(figsize=(7.2, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(df["N"], df["avg_log10_order"], marker="o")
    ax.set_title(f"{map_name} 映射：平均 log10(阶)-N 曲线")
    ax.set_xlabel("N")
    ax.set_ylabel("平均 log10(排列阶)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_map_comparison_figure(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7.2, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(df["map_name"], df["avg_log10_order"])
    ax.set_title("四种混沌映射平均 log10(阶) 对比")
    ax.set_xlabel("混沌映射")
    ax.set_ylabel("平均 log10(排列阶)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def make_four_maps_n_curve_figure(df: pd.DataFrame) -> Figure:
    """绘制四种混沌映射在不同 N 下的平均 log10(阶) 对比折线图。"""
    fig = Figure(figsize=(7.2, 4.2), dpi=100)
    ax = fig.add_subplot(111)

    for map_name in DEFAULT_PARAMS.keys():
        sub = df[df["map_name"] == map_name].sort_values("N")
        if not sub.empty:
            ax.plot(sub["N"], sub["avg_log10_order"], marker="o", label=map_name)

    ax.set_title("四种混沌映射：平均 log10(阶)-N 对比曲线")
    ax.set_xlabel("N")
    ax.set_ylabel("平均 log10(排列阶)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# =========================
# 8. GUI 主程序
# =========================

class ChaosPermutationApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title(APP_NAME)
        self.geometry("1280x820")
        self.minsize(1120, 720)

        self.results_dir = get_results_dir()

        self.last_seq: Optional[np.ndarray] = None
        self.last_perm: Optional[np.ndarray] = None
        self.last_cycles: Optional[List[List[int]]] = None
        self.last_stats: Optional[Dict[str, object]] = None
        self.last_stats_text: str = ""
        self.last_cycle_fig: Optional[Figure] = None

        self.original_image: Optional[Image.Image] = None
        self.scrambled_image: Optional[Image.Image] = None
        self.recovered_image: Optional[Image.Image] = None
        self.image_perm: Optional[np.ndarray] = None

        self._build_ui()

    # ---------- UI 构建 ----------
    def _build_ui(self) -> None:
        """构建最终展示版界面：浅色蓝灰、侧边导航、仪表盘布局。"""
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        self.font_family = "Microsoft YaHei UI"
        self.colors = {
            "window": "#EAF2FB",
            "sidebar": "#052659",
            "sidebar_hover": "#2F6FA8",
            "content": "#F5F8FC",
            "card": "#FFFFFF",
            "card_soft": "#EEF5FC",
            "border": "#C7DAEE",
            "text": "#1F2937",
            "muted": "#64748B",
            "primary": "#5483B3",
            "primary_hover": "#2F6FA8",
            "success": "#0F766E",
        }
        self.configure(fg_color=self.colors["window"])

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self.sidebar = ctk.CTkFrame(self, width=230, corner_radius=0, fg_color=self.colors["sidebar"])
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)

        brand = ctk.CTkLabel(
            self.sidebar,
            text="Chaos Permutation\nAnalysis System",
            font=ctk.CTkFont(family=self.font_family, size=20, weight="bold"),
            text_color="#FFFFFF",
            justify="left",
        )
        brand.grid(row=0, column=0, padx=22, pady=(26, 8), sticky="w")
        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="混沌映射 · 循环阶 · 图像置乱",
            font=ctk.CTkFont(family=self.font_family, size=12),
            text_color="#D9EAFB",
        )
        subtitle.grid(row=1, column=0, padx=22, pady=(0, 24), sticky="w")

        self.nav_buttons: Dict[str, ctk.CTkButton] = {}
        nav_items = [
            ("dashboard", "01  首页仪表盘"),
            ("single", "02  单次置乱分析"),
            ("batch", "03  批量实验"),
            ("image", "04  图像置乱"),
            ("results", "05  结果管理"),
            ("help", "06  关于系统"),
        ]
        for i, (key, label) in enumerate(nav_items, start=2):
            btn = ctk.CTkButton(
                self.sidebar,
                text=label,
                height=42,
                corner_radius=12,
                fg_color="transparent",
                hover_color=self.colors["sidebar_hover"],
                text_color="#EAF6FF",
                anchor="w",
                font=ctk.CTkFont(family=self.font_family, size=14),
                command=lambda k=key: self.show_page(k),
            )
            btn.grid(row=i, column=0, padx=14, pady=5, sticky="ew")
            self.nav_buttons[key] = btn

        ctk.CTkLabel(
            self.sidebar,
            text="Version 1.0",
            font=ctk.CTkFont(family=self.font_family, size=12),
            text_color="#B8D9F5",
        ).grid(row=20, column=0, padx=22, pady=(20, 24), sticky="sw")
        self.sidebar.grid_rowconfigure(19, weight=1)

        self.main_area = ctk.CTkFrame(self, fg_color=self.colors["content"], corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew")
        self.main_area.grid_columnconfigure(0, weight=1)
        self.main_area.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self.main_area, fg_color=self.colors["content"], corner_radius=0)
        self.header.grid(row=0, column=0, padx=28, pady=(22, 6), sticky="ew")
        self.header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.header,
            text="基于混沌映射的置乱表循环阶分析与图像置乱系统",
            font=ctk.CTkFont(family=self.font_family, size=24, weight="bold"),
            text_color="#052659",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            self.header,
            text="Chaos-Based Permutation Order Analysis System · 循环圈分解 · 平均阶实验 · 图像像素置乱恢复",
            font=ctk.CTkFont(family=self.font_family, size=13),
            text_color=self.colors["muted"],
        ).grid(row=1, column=0, pady=(4, 0), sticky="w")

        self.page_container = ctk.CTkFrame(self.main_area, fg_color=self.colors["content"], corner_radius=0)
        self.page_container.grid(row=1, column=0, padx=24, pady=14, sticky="nsew")
        self.page_container.grid_columnconfigure(0, weight=1)
        self.page_container.grid_rowconfigure(0, weight=1)

        self.pages: Dict[str, ctk.CTkFrame] = {}
        for key in ["dashboard", "single", "batch", "image", "results", "help"]:
            frame = ctk.CTkFrame(self.page_container, fg_color="transparent", corner_radius=0)
            frame.grid(row=0, column=0, sticky="nsew")
            frame.grid_remove()
            self.pages[key] = frame

        self._build_dashboard_page()
        self._build_single_tab()
        self._build_batch_tab()
        self._build_image_tab()
        self._build_results_page()
        self._build_help_tab()

        self.status_var = ctk.StringVar(value=f"结果目录：{self.results_dir}")
        status_bar = ctk.CTkFrame(self, fg_color="#DCEBFA", corner_radius=0, height=34)
        status_bar.grid(row=1, column=1, sticky="ew")
        status_bar.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            status_bar,
            textvariable=self.status_var,
            anchor="w",
            font=ctk.CTkFont(family=self.font_family, size=12),
            text_color=self.colors["muted"],
        ).grid(row=0, column=0, padx=20, pady=6, sticky="ew")

        self.show_page("dashboard")

    def _card(self, parent, fg: Optional[str] = None) -> ctk.CTkFrame:
        return ctk.CTkFrame(
            parent,
            fg_color=fg or self.colors["card"],
            corner_radius=20,
            border_width=1,
            border_color=self.colors["border"],
        )

    def _section_title(self, parent, text: str, row: int = 0, column: int = 0, columnspan: int = 1) -> None:
        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(family=self.font_family, size=17, weight="bold"),
            text_color=self.colors["text"],
        ).grid(row=row, column=column, columnspan=columnspan, padx=16, pady=(16, 8), sticky="w")

    def _primary_button(self, parent, text: str, command) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=38,
            corner_radius=12,
            fg_color=self.colors["primary"],
            hover_color=self.colors["primary_hover"],
            font=ctk.CTkFont(family=self.font_family, size=13, weight="bold"),
        )

    def _secondary_button(self, parent, text: str, command) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=36,
            corner_radius=12,
            fg_color="#E6F1FB",
            hover_color="#D3E8F8",
            text_color=self.colors["text"],
            font=ctk.CTkFont(family=self.font_family, size=13),
        )

    def show_page(self, key: str) -> None:
        for page_key, frame in self.pages.items():
            if page_key == key:
                frame.grid()
            else:
                frame.grid_remove()
        for page_key, btn in self.nav_buttons.items():
            if page_key == key:
                btn.configure(fg_color="#5483B3", text_color="#FFFFFF")
            else:
                btn.configure(fg_color="transparent", text_color="#EAF6FF")
        if key == "results":
            self.refresh_results_list()

    def _build_dashboard_page(self) -> None:
        page = self.pages["dashboard"]
        page.grid_columnconfigure((0, 1, 2, 3), weight=1)
        page.grid_rowconfigure(1, weight=1)

        self.dashboard_cards = {}
        cards = [
            ("map", "当前映射", "Logistic"),
            ("n", "置乱规模", "N=1000"),
            ("max_cycle", "最大循环长度", "—"),
            ("order", "log10(阶)", "—"),
        ]
        for i, (key, title, value) in enumerate(cards):
            card = self._card(page, fg="#F8FCFF")
            card.grid(row=0, column=i, padx=8, pady=(0, 14), sticky="nsew")
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(family=self.font_family, size=12), text_color=self.colors["muted"]).grid(row=0, column=0, padx=18, pady=(16, 2), sticky="w")
            value_var = ctk.StringVar(value=value)
            ctk.CTkLabel(card, textvariable=value_var, font=ctk.CTkFont(family=self.font_family, size=22, weight="bold"), text_color=self.colors["text"]).grid(row=1, column=0, padx=18, pady=(0, 16), sticky="w")
            self.dashboard_cards[key] = value_var

        center = self._card(page)
        center.grid(row=1, column=0, columnspan=4, padx=8, pady=8, sticky="nsew")
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(1, weight=1)
        self._section_title(center, "实验概览")
        self.dashboard_plot_frame = ctk.CTkFrame(center, fg_color="#FFFFFF", corner_radius=16, border_width=1, border_color=self.colors["border"])
        self.dashboard_plot_frame.grid(row=1, column=0, padx=16, pady=(4, 14), sticky="nsew")
        ctk.CTkLabel(self.dashboard_plot_frame, text="完成一次分析或批量实验后，核心图表会同步显示在这里。", text_color=self.colors["muted"], font=ctk.CTkFont(family=self.font_family, size=14)).pack(expand=True)

        actions = ctk.CTkFrame(page, fg_color="transparent")
        actions.grid(row=2, column=0, columnspan=4, padx=8, pady=(10, 0), sticky="ew")
        actions.grid_columnconfigure((0, 1, 2), weight=1)
        self._primary_button(actions, "开始单次分析", lambda: self.show_page("single")).grid(row=0, column=0, padx=8, pady=4, sticky="ew")
        self._primary_button(actions, "运行批量实验", lambda: self.show_page("batch")).grid(row=0, column=1, padx=8, pady=4, sticky="ew")
        self._primary_button(actions, "进入图像置乱", lambda: self.show_page("image")).grid(row=0, column=2, padx=8, pady=4, sticky="ew")

    def _build_common_param_panel(self, parent) -> ctk.CTkFrame:
        frame = self._card(parent, fg=self.colors["card_soft"])
        frame.grid_columnconfigure(1, weight=1)

        self.map_var = ctk.StringVar(value="Logistic")
        self.param_var = ctk.StringVar(value=str(DEFAULT_PARAMS["Logistic"]))
        self.x0_var = ctk.StringVar(value="0.3721")
        self.warmup_var = ctk.StringVar(value="1000")
        self.n_var = ctk.StringVar(value="1000")

        self._section_title(frame, "参数设置", row=0, column=0, columnspan=2)
        fields = [
            ("混沌映射", "menu"),
            ("参数 μ/a/k", self.param_var),
            ("初始值 x0", self.x0_var),
            ("预迭代 M", self.warmup_var),
            ("置乱规模 N", self.n_var),
        ]
        for row, (label, var) in enumerate(fields, start=1):
            ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(family=self.font_family, size=13), text_color=self.colors["text"]).grid(row=row, column=0, padx=16, pady=8, sticky="w")
            if var == "menu":
                widget = ctk.CTkOptionMenu(frame, variable=self.map_var, values=list(DEFAULT_PARAMS.keys()), command=self._on_map_change, fg_color=self.colors["primary"], button_color=self.colors["primary_hover"], button_hover_color="#1D4ED8")
            else:
                widget = ctk.CTkEntry(frame, textvariable=var, corner_radius=10, border_color=self.colors["border"], font=ctk.CTkFont(family=self.font_family, size=13))
            widget.grid(row=row, column=1, padx=16, pady=8, sticky="ew")

        self.map_desc_var = ctk.StringVar(value=MAP_DESCRIPTIONS["Logistic"])
        desc = ctk.CTkLabel(frame, textvariable=self.map_desc_var, wraplength=330, justify="left", text_color=self.colors["muted"], font=ctk.CTkFont(family=self.font_family, size=12))
        desc.grid(row=6, column=0, columnspan=2, padx=16, pady=(10, 16), sticky="w")
        return frame

    def _build_single_tab(self) -> None:
        page = self.pages["single"]
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(page, fg_color="transparent", width=370)
        left.grid(row=0, column=0, padx=(0, 14), pady=0, sticky="nsw")
        left.grid_columnconfigure(0, weight=1)
        left.grid_propagate(False)

        param_panel = self._build_common_param_panel(left)
        param_panel.grid(row=0, column=0, padx=8, pady=(0, 12), sticky="ew")

        btn_frame = self._card(left)
        btn_frame.grid(row=1, column=0, padx=8, pady=0, sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        self._section_title(btn_frame, "操作")
        self._primary_button(btn_frame, "生成置乱表并分析", self.run_single_analysis).grid(row=1, column=0, padx=16, pady=(4, 8), sticky="ew")
        self._secondary_button(btn_frame, "导出当前结果", self.export_current_result).grid(row=2, column=0, padx=16, pady=8, sticky="ew")
        self._secondary_button(btn_frame, "打开 results 目录", self.open_results_dir).grid(row=3, column=0, padx=16, pady=(8, 16), sticky="ew")

        note = ctk.CTkLabel(left, text="导出的 CSV 同时包含 0 基索引和 1 基索引；报告中建议使用 1 基索引描述。", wraplength=330, justify="left", text_color=self.colors["muted"], font=ctk.CTkFont(family=self.font_family, size=12))
        note.grid(row=2, column=0, padx=18, pady=12, sticky="w")

        right = self._card(page)
        right.grid(row=0, column=1, padx=(0, 0), pady=0, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)
        self._section_title(right, "循环结构分析结果")
        self.single_text = ctk.CTkTextbox(right, height=210, font=ctk.CTkFont(family=self.font_family, size=13), corner_radius=14, border_width=1, border_color=self.colors["border"])
        self.single_text.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.single_text.insert("1.0", "点击“生成置乱表并分析”后，这里会显示循环圈统计结果。")
        self.single_plot_frame = ctk.CTkFrame(right, fg_color="#FFFFFF", corner_radius=16, border_width=1, border_color=self.colors["border"])
        self.single_plot_frame.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")

    def _build_batch_tab(self) -> None:
        page = self.pages["batch"]
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(page, fg_color="transparent", width=370)
        left.grid(row=0, column=0, padx=(0, 14), pady=0, sticky="nsw")
        left.grid_columnconfigure(0, weight=1)
        left.grid_propagate(False)

        panel = self._card(left, fg=self.colors["card_soft"])
        panel.grid(row=0, column=0, padx=8, pady=(0, 12), sticky="ew")
        panel.grid_columnconfigure(1, weight=1)
        self.n_min_var = ctk.StringVar(value="500")
        self.n_max_var = ctk.StringVar(value="5000")
        self.n_step_var = ctk.StringVar(value="500")
        self.seed_count_var = ctk.StringVar(value="50")
        self._section_title(panel, "批量实验参数", row=0, column=0, columnspan=2)
        labels_vars = [("N 起始", self.n_min_var), ("N 结束", self.n_max_var), ("N 步长", self.n_step_var), ("种子数量", self.seed_count_var)]
        for i, (label, var) in enumerate(labels_vars, start=1):
            ctk.CTkLabel(panel, text=label, font=ctk.CTkFont(family=self.font_family, size=13), text_color=self.colors["text"]).grid(row=i, column=0, padx=16, pady=8, sticky="w")
            ctk.CTkEntry(panel, textvariable=var, corner_radius=10, border_color=self.colors["border"], font=ctk.CTkFont(family=self.font_family, size=13)).grid(row=i, column=1, padx=16, pady=8, sticky="ew")

        btns = self._card(left)
        btns.grid(row=1, column=0, padx=8, pady=0, sticky="ew")
        btns.grid_columnconfigure(0, weight=1)
        self._section_title(btns, "实验操作")
        self._primary_button(btns, "当前映射：平均阶-N 曲线", self.run_n_curve_experiment).grid(row=1, column=0, padx=16, pady=(4, 8), sticky="ew")
        self._primary_button(btns, "四种映射：随N变化曲线", self.run_four_maps_n_curve_experiment).grid(row=2, column=0, padx=16, pady=8, sticky="ew")
        self._secondary_button(btns, "四种映射：固定N柱状对比", self.run_map_comparison_experiment).grid(row=3, column=0, padx=16, pady=8, sticky="ew")
        self._secondary_button(btns, "一键生成报告素材", self.run_report_materials).grid(row=4, column=0, padx=16, pady=8, sticky="ew")
        self._secondary_button(btns, "打开 results 目录", self.open_results_dir).grid(row=5, column=0, padx=16, pady=(8, 16), sticky="ew")

        right = self._card(page)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)
        self._section_title(right, "批量实验输出")
        self.batch_text = ctk.CTkTextbox(right, height=200, font=ctk.CTkFont(family=self.font_family, size=13), corner_radius=14, border_width=1, border_color=self.colors["border"])
        self.batch_text.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.batch_text.insert("1.0", "批量实验结果会显示在这里，并自动导出 CSV 和 PNG 图表。")
        self.batch_plot_frame = ctk.CTkFrame(right, fg_color="#FFFFFF", corner_radius=16, border_width=1, border_color=self.colors["border"])
        self.batch_plot_frame.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")

    def _build_image_tab(self) -> None:
        page = self.pages["image"]
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(page, fg_color="transparent", width=370)
        left.grid(row=0, column=0, padx=(0, 14), pady=0, sticky="nsw")
        left.grid_columnconfigure(0, weight=1)
        left.grid_propagate(False)
        panel = self._card(left, fg=self.colors["card_soft"])
        panel.grid(row=0, column=0, padx=8, pady=(0, 12), sticky="ew")
        panel.grid_columnconfigure(1, weight=1)
        self._section_title(panel, "图像置乱参数", row=0, column=0, columnspan=2)
        self.max_side_var = ctk.StringVar(value="512")
        ctk.CTkLabel(panel, text="最大演示边长", font=ctk.CTkFont(family=self.font_family, size=13), text_color=self.colors["text"]).grid(row=1, column=0, padx=16, pady=8, sticky="w")
        ctk.CTkEntry(panel, textvariable=self.max_side_var, corner_radius=10, border_color=self.colors["border"]).grid(row=1, column=1, padx=16, pady=8, sticky="ew")
        self._primary_button(panel, "选择图像", self.choose_image).grid(row=2, column=0, columnspan=2, padx=16, pady=(14, 8), sticky="ew")
        self._primary_button(panel, "执行图像置乱", self.scramble_current_image).grid(row=3, column=0, columnspan=2, padx=16, pady=8, sticky="ew")
        self._secondary_button(panel, "执行逆置乱恢复", self.recover_current_image).grid(row=4, column=0, columnspan=2, padx=16, pady=8, sticky="ew")
        self._secondary_button(panel, "保存图像结果", self.save_image_results).grid(row=5, column=0, columnspan=2, padx=16, pady=8, sticky="ew")
        self._secondary_button(panel, "打开 results 目录", self.open_results_dir).grid(row=6, column=0, columnspan=2, padx=16, pady=(8, 16), sticky="ew")

        self.image_info_text = ctk.CTkTextbox(left, height=220, font=ctk.CTkFont(family=self.font_family, size=13), corner_radius=14, border_width=1, border_color=self.colors["border"])
        self.image_info_text.grid(row=1, column=0, padx=8, pady=0, sticky="ew")
        self.image_info_text.insert("1.0", "请选择图像后进行置乱。图像像素数会作为置乱规模 N。")

        right = self._card(page)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure((0, 1, 2), weight=1)
        right.grid_rowconfigure(1, weight=1)
        for col, title in enumerate(["原图", "置乱图", "恢复图"]):
            ctk.CTkLabel(right, text=title, font=ctk.CTkFont(family=self.font_family, size=16, weight="bold"), text_color=self.colors["text"]).grid(row=0, column=col, padx=8, pady=(16, 8))
        self.original_img_label = ctk.CTkLabel(right, text="未选择图像", fg_color="#F8FBFF", corner_radius=16)
        self.original_img_label.grid(row=1, column=0, padx=(16, 8), pady=(0, 16), sticky="nsew")
        self.scrambled_img_label = ctk.CTkLabel(right, text="未置乱", fg_color="#F8FBFF", corner_radius=16)
        self.scrambled_img_label.grid(row=1, column=1, padx=8, pady=(0, 16), sticky="nsew")
        self.recovered_img_label = ctk.CTkLabel(right, text="未恢复", fg_color="#F8FBFF", corner_radius=16)
        self.recovered_img_label.grid(row=1, column=2, padx=(8, 16), pady=(0, 16), sticky="nsew")

    def _build_results_page(self) -> None:
        page = self.pages["results"]
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        top = self._card(page)
        top.grid(row=0, column=0, padx=8, pady=(0, 14), sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        self._section_title(top, "结果文件管理")
        ctk.CTkLabel(top, text=f"当前结果目录：{self.results_dir}", text_color=self.colors["muted"], font=ctk.CTkFont(family=self.font_family, size=12)).grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")
        btnbar = ctk.CTkFrame(top, fg_color="transparent")
        btnbar.grid(row=0, column=1, rowspan=2, padx=16, pady=16, sticky="e")
        self._secondary_button(btnbar, "刷新列表", self.refresh_results_list).grid(row=0, column=0, padx=6)
        self._primary_button(btnbar, "打开目录", self.open_results_dir).grid(row=0, column=1, padx=6)

        self.results_list_frame = ctk.CTkScrollableFrame(page, fg_color="#FFFFFF", corner_radius=18, border_width=1, border_color=self.colors["border"])
        self.results_list_frame.grid(row=1, column=0, padx=8, pady=0, sticky="nsew")
        self.results_list_frame.grid_columnconfigure(0, weight=1)

    def refresh_results_list(self) -> None:
        if not hasattr(self, "results_list_frame"):
            return
        for child in self.results_list_frame.winfo_children():
            child.destroy()
        files = sorted(self.results_dir.glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        if not files:
            ctk.CTkLabel(self.results_list_frame, text="results 目录暂无文件。", text_color=self.colors["muted"], font=ctk.CTkFont(family=self.font_family, size=14)).grid(row=0, column=0, padx=16, pady=16, sticky="w")
            return
        for i, path in enumerate(files[:80]):
            row = ctk.CTkFrame(self.results_list_frame, fg_color="#F8FBFF" if i % 2 == 0 else "#EEF5FC", corner_radius=10)
            row.grid(row=i, column=0, padx=10, pady=5, sticky="ew")
            row.grid_columnconfigure(0, weight=1)
            size_kb = path.stat().st_size / 1024
            ctk.CTkLabel(row, text=path.name, anchor="w", font=ctk.CTkFont(family=self.font_family, size=13), text_color=self.colors["text"]).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="ew")
            ctk.CTkLabel(row, text=f"{path.suffix.upper().lstrip('.') or 'FILE'} · {size_kb:.1f} KB", anchor="w", font=ctk.CTkFont(family=self.font_family, size=11), text_color=self.colors["muted"]).grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
            self._secondary_button(row, "打开", lambda p=path: os.startfile(str(p))).grid(row=0, column=1, rowspan=2, padx=12, pady=8)

    def _build_help_tab(self) -> None:
        page = self.pages["help"]
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(0, weight=1)
        card = self._card(page)
        card.grid(row=0, column=0, padx=8, pady=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)
        self._section_title(card, "关于系统")
        text = ctk.CTkTextbox(card, font=ctk.CTkFont(family=self.font_family, size=14), corner_radius=14, border_width=1, border_color=self.colors["border"])
        text.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")
        help_content = f"""
【软件功能】
1. 支持 Logistic、Tent、Sine、Chebyshev 四种混沌映射。
2. 根据参数、初始值、预迭代轮数 M 和规模 N 生成置乱表。
3. 将置乱表分解为循环圈，统计循环圈数量、循环长度分布、最大循环长度和排列总阶。
4. 支持平均 log10(阶)-N 曲线实验、固定 N 映射对比和四映射随 N 变化对比。
5. 支持图像像素位置置乱和逆置乱恢复。
6. 支持导出 CSV 表格、PNG 图表、TXT 摘要和图像结果。

【推荐实验流程】
第一步：在“单次置乱分析”中设置 N=1000、M=1000，生成置乱表并导出结果。
第二步：在“批量实验”中运行“四种映射：随N变化曲线”。
第三步：在“图像置乱”中选择一张图片，完成置乱和恢复，并保存结果。
第四步：打开 results 目录，将 CSV、PNG 和 TXT 文件作为报告素材。

【索引说明】
程序内部使用 0 基索引，导出 CSV 同时提供 0 基和 1 基索引。写报告时建议使用 1 基索引，与题目描述更一致。

【结果目录】
{self.results_dir}

【注意】
排列阶可能非常大，因此批量曲线使用 log10(阶) 展示。该处理不会改变循环圈分解结果，只是为了让图表更易读。
"""
        text.insert("1.0", help_content.strip())
        text.configure(state="disabled")

    def _update_dashboard_after_single(self) -> None:
        if not self.last_stats:
            return
        self.dashboard_cards["map"].set(self.map_var.get())
        self.dashboard_cards["n"].set(f"N={self.last_stats.get('n', '—')}")
        self.dashboard_cards["max_cycle"].set(str(self.last_stats.get("max_cycle_length", "—")))
        order_log10 = self.last_stats.get("order_log10", None)
        self.dashboard_cards["order"].set(f"{float(order_log10):.4f}" if order_log10 is not None else "—")
        if self.last_cycle_fig is not None:
            self._render_figure(self.dashboard_plot_frame, self.last_cycle_fig)

    # ---------- 通用 UI 方法 ----------
    def _on_map_change(self, selected: str) -> None:
        self.param_var.set(str(DEFAULT_PARAMS[selected]))
        self.map_desc_var.set(MAP_DESCRIPTIONS[selected])

    def _get_params(self) -> Tuple[str, float, float, int, int]:
        map_name = self.map_var.get()
        param = safe_float(self.param_var.get(), "映射参数")
        x0 = safe_float(self.x0_var.get(), "初始值 x0")
        warmup = safe_int(self.warmup_var.get(), "预迭代轮数 M", 0, 1_000_000)
        n = safe_int(self.n_var.get(), "置乱规模 N", 2, 1_000_000)
        return map_name, param, x0, warmup, n

    def _set_text(self, textbox: ctk.CTkTextbox, content: str) -> None:
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("1.0", content)

    def _append_text(self, textbox: ctk.CTkTextbox, content: str) -> None:
        textbox.configure(state="normal")
        textbox.insert("end", content)
        textbox.see("end")

    def _render_figure(self, parent: ctk.CTkFrame, fig: Figure) -> None:
        for child in parent.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        parent._canvas = canvas  # 防止被垃圾回收

    def _show_pil_image(self, label: ctk.CTkLabel, image: Image.Image, max_size: Tuple[int, int] = (330, 430)) -> None:
        img = image.copy().convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        label.configure(image=ctk_img, text="")
        label._image_ref = ctk_img

    def set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.update_idletasks()

    def open_results_dir(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.results_dir))

    # ---------- 单次分析 ----------
    def run_single_analysis(self) -> None:
        try:
            map_name, param, x0, warmup, n = self._get_params()
            self.set_status("正在生成置乱表并分析循环圈...")
            t0 = time.perf_counter()
            seq = generate_chaos_sequence(map_name, param, x0, warmup, n)
            perm = sequence_to_permutation(seq)
            cycles, stats = analyze_cycles(perm)
            elapsed = time.perf_counter() - t0

            self.last_seq = seq
            self.last_perm = perm
            self.last_cycles = cycles
            self.last_stats = stats
            self.last_stats_text = format_stats_text(map_name, param, x0, warmup, stats, elapsed)
            self.last_cycle_fig = make_cycle_distribution_figure(stats)

            self._set_text(self.single_text, self.last_stats_text)
            self._render_figure(self.single_plot_frame, self.last_cycle_fig)
            self._update_dashboard_after_single()
            self.set_status("单次置乱分析完成。")
        except Exception as e:
            messagebox.showerror("错误", str(e))
            self.set_status("操作失败，请检查参数。")

    def export_current_result(self) -> None:
        try:
            if self.last_seq is None or self.last_perm is None or self.last_stats is None:
                messagebox.showwarning("提示", "请先生成置乱表并分析。")
                return

            tag = now_tag()
            map_name = self.map_var.get()
            prefix = f"{tag}_{map_name}_N{len(self.last_perm)}"

            perm_csv = self.results_dir / f"{prefix}_permutation_table.csv"
            cycle_csv = self.results_dir / f"{prefix}_cycle_statistics.csv"
            summary_txt = self.results_dir / f"{prefix}_summary.txt"
            fig_png = self.results_dir / f"{prefix}_cycle_distribution.png"

            export_permutation_csv(perm_csv, self.last_seq, self.last_perm)
            export_cycle_stats_csv(cycle_csv, self.last_stats)
            export_summary_txt(summary_txt, self.last_stats_text)
            if self.last_cycle_fig is not None:
                self.last_cycle_fig.savefig(fig_png, dpi=160, bbox_inches="tight")

            self.set_status(f"当前结果已导出：{self.results_dir}")
            messagebox.showinfo("导出成功", f"已导出：\n{perm_csv.name}\n{cycle_csv.name}\n{summary_txt.name}\n{fig_png.name}")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    # ---------- 批量实验 ----------
    def _get_batch_params(self) -> Tuple[int, int, int, int]:
        n_min = safe_int(self.n_min_var.get(), "N 起始", 2, 1_000_000)
        n_max = safe_int(self.n_max_var.get(), "N 结束", 2, 1_000_000)
        n_step = safe_int(self.n_step_var.get(), "N 步长", 1, 1_000_000)
        seed_count = safe_int(self.seed_count_var.get(), "种子数量", 1, 200)
        if n_min > n_max:
            raise ValueError("N 起始不能大于 N 结束。")
        return n_min, n_max, n_step, seed_count

    def _seed_values(self, seed_count: int) -> np.ndarray:
        # 固定随机种子，保证实验可复现
        rng = np.random.default_rng(2025)
        return rng.uniform(0.123456, 0.876543, size=seed_count)

    def _single_order_log10(self, map_name: str, param: float, x0: float, warmup: int, n: int) -> Tuple[float, int, int, int]:
        seq = generate_chaos_sequence(map_name, param, x0, warmup, n)
        perm = sequence_to_permutation(seq)
        _, stats = analyze_cycles(perm)
        return (
            float(stats["order_log10"]),
            int(stats["cycle_count"]),
            int(stats["different_length_count"]),
            int(stats["max_cycle_length"]),
        )

    def run_n_curve_experiment(self) -> Optional[pd.DataFrame]:
        try:
            map_name, param, _x0, warmup, _n = self._get_params()
            n_min, n_max, n_step, seed_count = self._get_batch_params()
            seeds = self._seed_values(seed_count)
            n_values = list(range(n_min, n_max + 1, n_step))

            self._set_text(self.batch_text, "正在运行平均阶-N 曲线实验...\n")
            self.set_status("正在运行批量实验，请稍候...")
            self.update_idletasks()

            rows = []
            t0 = time.perf_counter()
            for n in n_values:
                logs, cycle_counts, len_counts, max_lens = [], [], [], []
                for seed in seeds:
                    order_log10, cycle_count, different_len_count, max_len = self._single_order_log10(map_name, param, float(seed), warmup, n)
                    logs.append(order_log10)
                    cycle_counts.append(cycle_count)
                    len_counts.append(different_len_count)
                    max_lens.append(max_len)
                rows.append({
                    "map_name": map_name,
                    "param": param,
                    "N": n,
                    "seed_count": seed_count,
                    "avg_log10_order": float(np.mean(logs)),
                    "std_log10_order": float(np.std(logs)),
                    "avg_cycle_count": float(np.mean(cycle_counts)),
                    "avg_different_length_count": float(np.mean(len_counts)),
                    "avg_max_cycle_length": float(np.mean(max_lens)),
                })
                self._append_text(self.batch_text, f"N={n} 完成，平均 log10(阶)={rows[-1]['avg_log10_order']:.4f}\n")
                self.update_idletasks()

            elapsed = time.perf_counter() - t0
            df = pd.DataFrame(rows)
            fig = make_n_curve_figure(df, map_name)
            self._render_figure(self.batch_plot_frame, fig)
            self._render_figure(self.dashboard_plot_frame, fig)

            tag = now_tag()
            csv_path = self.results_dir / f"{tag}_{map_name}_avg_order_N_curve.csv"
            png_path = self.results_dir / f"{tag}_{map_name}_avg_order_N_curve.png"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            fig.savefig(png_path, dpi=160, bbox_inches="tight")

            summary = (
                f"平均阶-N 曲线实验完成。\n"
                f"映射：{map_name}\n参数：{param}\nM：{warmup}\n种子数量：{seed_count}\n"
                f"N 范围：{n_min} 到 {n_max}，步长 {n_step}\n"
                f"耗时：{elapsed:.4f} 秒\n"
                f"CSV：{csv_path.name}\nPNG：{png_path.name}\n"
            )
            self._append_text(self.batch_text, "\n" + summary)
            self.set_status("平均阶-N 曲线实验完成。")
            return df
        except Exception as e:
            messagebox.showerror("实验失败", str(e))
            self.set_status("批量实验失败，请检查参数。")
            return None

    def run_map_comparison_experiment(self) -> Optional[pd.DataFrame]:
        try:
            _map_name, _param, _x0, warmup, n = self._get_params()
            _n_min, _n_max, _n_step, seed_count = self._get_batch_params()
            seeds = self._seed_values(seed_count)

            self._set_text(self.batch_text, "正在运行四种映射平均阶对比实验...\n")
            self.set_status("正在运行四种映射对比实验，请稍候...")
            self.update_idletasks()

            rows = []
            t0 = time.perf_counter()
            for map_name in DEFAULT_PARAMS.keys():
                param = DEFAULT_PARAMS[map_name]
                logs, cycle_counts, len_counts, max_lens = [], [], [], []
                for seed in seeds:
                    order_log10, cycle_count, different_len_count, max_len = self._single_order_log10(map_name, param, float(seed), warmup, n)
                    logs.append(order_log10)
                    cycle_counts.append(cycle_count)
                    len_counts.append(different_len_count)
                    max_lens.append(max_len)
                rows.append({
                    "map_name": map_name,
                    "param": param,
                    "N": n,
                    "M": warmup,
                    "seed_count": seed_count,
                    "avg_log10_order": float(np.mean(logs)),
                    "std_log10_order": float(np.std(logs)),
                    "avg_cycle_count": float(np.mean(cycle_counts)),
                    "avg_different_length_count": float(np.mean(len_counts)),
                    "avg_max_cycle_length": float(np.mean(max_lens)),
                })
                self._append_text(self.batch_text, f"{map_name} 完成，平均 log10(阶)={rows[-1]['avg_log10_order']:.4f}\n")
                self.update_idletasks()

            elapsed = time.perf_counter() - t0
            df = pd.DataFrame(rows)
            fig = make_map_comparison_figure(df)
            self._render_figure(self.batch_plot_frame, fig)
            self._render_figure(self.dashboard_plot_frame, fig)

            tag = now_tag()
            csv_path = self.results_dir / f"{tag}_four_maps_comparison.csv"
            png_path = self.results_dir / f"{tag}_four_maps_comparison.png"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            fig.savefig(png_path, dpi=160, bbox_inches="tight")

            summary = (
                f"四种映射平均阶对比实验完成。\n"
                f"N：{n}\nM：{warmup}\n种子数量：{seed_count}\n耗时：{elapsed:.4f} 秒\n"
                f"CSV：{csv_path.name}\nPNG：{png_path.name}\n"
            )
            self._append_text(self.batch_text, "\n" + summary)
            self.set_status("四种映射对比实验完成。")
            return df
        except Exception as e:
            messagebox.showerror("实验失败", str(e))
            self.set_status("映射对比实验失败，请检查参数。")
            return None


    def run_four_maps_n_curve_experiment(self) -> Optional[pd.DataFrame]:
        """四种映射在多个 N 下的平均阶对比曲线。

        这是报告中最核心的批量实验之一：同一 N 范围、同一种子数量下，
        对 Logistic、Tent、Sine、Chebyshev 四种映射分别计算平均 log10(排列阶)。
        """
        try:
            _map_name, _param, _x0, warmup, _n = self._get_params()
            n_min, n_max, n_step, seed_count = self._get_batch_params()
            seeds = self._seed_values(seed_count)
            n_values = list(range(n_min, n_max + 1, n_step))

            self._set_text(self.batch_text, "正在运行四种映射随 N 变化对比实验...\n")
            self.set_status("正在运行四种映射随 N 变化对比实验，请稍候...")
            self.update_idletasks()

            rows = []
            t0 = time.perf_counter()
            for map_name in DEFAULT_PARAMS.keys():
                param = DEFAULT_PARAMS[map_name]
                self._append_text(self.batch_text, f"\n[{map_name}] 开始，参数={param}\n")
                for n in n_values:
                    logs, cycle_counts, len_counts, max_lens = [], [], [], []
                    for seed in seeds:
                        order_log10, cycle_count, different_len_count, max_len = self._single_order_log10(
                            map_name, param, float(seed), warmup, n
                        )
                        logs.append(order_log10)
                        cycle_counts.append(cycle_count)
                        len_counts.append(different_len_count)
                        max_lens.append(max_len)
                    rows.append({
                        "map_name": map_name,
                        "param": param,
                        "M": warmup,
                        "N": n,
                        "seed_count": seed_count,
                        "avg_log10_order": float(np.mean(logs)),
                        "std_log10_order": float(np.std(logs)),
                        "avg_cycle_count": float(np.mean(cycle_counts)),
                        "avg_different_length_count": float(np.mean(len_counts)),
                        "avg_max_cycle_length": float(np.mean(max_lens)),
                    })
                    self._append_text(
                        self.batch_text,
                        f"{map_name}  N={n} 完成，平均 log10(阶)={rows[-1]['avg_log10_order']:.4f}\n"
                    )
                    self.update_idletasks()

            elapsed = time.perf_counter() - t0
            df = pd.DataFrame(rows)
            fig = make_four_maps_n_curve_figure(df)
            self._render_figure(self.batch_plot_frame, fig)
            self._render_figure(self.dashboard_plot_frame, fig)

            tag = now_tag()
            csv_path = self.results_dir / f"{tag}_four_maps_N_curve_comparison.csv"
            png_path = self.results_dir / f"{tag}_four_maps_N_curve_comparison.png"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            fig.savefig(png_path, dpi=160, bbox_inches="tight")

            summary = (
                f"四种映射随 N 变化对比实验完成。\n"
                f"M：{warmup}\n种子数量：{seed_count}\n"
                f"N 范围：{n_min} 到 {n_max}，步长 {n_step}\n"
                f"耗时：{elapsed:.4f} 秒\n"
                f"CSV：{csv_path.name}\nPNG：{png_path.name}\n"
            )
            self._append_text(self.batch_text, "\n" + summary)
            self.set_status("四种映射随 N 变化对比实验完成。")
            return df
        except Exception as e:
            messagebox.showerror("实验失败", str(e))
            self.set_status("四种映射随 N 变化对比实验失败，请检查参数。")
            return None

    def run_report_materials(self) -> None:
        """一键生成报告常用素材：单次分析、N 曲线、四映射对比。"""
        try:
            self.run_single_analysis()
            self.export_current_result()
            self.run_n_curve_experiment()
            self.run_map_comparison_experiment()
            self.run_four_maps_n_curve_experiment()
            messagebox.showinfo("完成", "报告素材已生成。请打开 results 目录查看 CSV、PNG 和 TXT 文件。")
        except Exception as e:
            messagebox.showerror("失败", str(e))

    # ---------- 图像置乱 ----------
    def choose_image(self) -> None:
        try:
            path = filedialog.askopenfilename(
                title="选择图像",
                filetypes=[
                    ("图像文件", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                    ("所有文件", "*.*"),
                ],
            )
            if not path:
                return

            img = Image.open(path).convert("RGB")
            max_side = safe_int(self.max_side_var.get(), "最大演示边长", 0, 10000)
            original_size = img.size
            if max_side > 0 and max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

            self.original_image = img
            self.scrambled_image = None
            self.recovered_image = None
            self.image_perm = None

            self._show_pil_image(self.original_img_label, img)
            self.scrambled_img_label.configure(image=None, text="未置乱")
            self.recovered_img_label.configure(image=None, text="未恢复")

            info = (
                f"已选择图像：{Path(path).name}\n"
                f"原始尺寸：{original_size[0]} × {original_size[1]}\n"
                f"演示尺寸：{img.size[0]} × {img.size[1]}\n"
                f"像素数 N：{img.size[0] * img.size[1]}\n"
                f"当前映射参数将用于生成图像置乱表。"
            )
            self._set_text(self.image_info_text, info)
            self.set_status("图像已载入。")
        except Exception as e:
            messagebox.showerror("选择图像失败", str(e))

    def scramble_current_image(self) -> None:
        try:
            if self.original_image is None:
                messagebox.showwarning("提示", "请先选择图像。")
                return

            map_name, param, x0, warmup, _n = self._get_params()
            w, h = self.original_image.size
            n_pixels = w * h

            self.set_status("正在生成图像置乱表并执行置乱...")
            t0 = time.perf_counter()
            seq = generate_chaos_sequence(map_name, param, x0, warmup, n_pixels)
            perm = sequence_to_permutation(seq)
            scrambled = scramble_image_by_perm(self.original_image, perm)
            elapsed = time.perf_counter() - t0

            self.image_perm = perm
            self.scrambled_image = scrambled
            self.recovered_image = None
            self._show_pil_image(self.scrambled_img_label, scrambled)
            self.recovered_img_label.configure(image=None, text="未恢复")

            info = (
                f"图像置乱完成。\n"
                f"映射：{map_name}\n参数：{param}\nx0：{x0}\nM：{warmup}\n"
                f"图像尺寸：{w} × {h}\n像素数 N：{n_pixels}\n"
                f"耗时：{elapsed:.4f} 秒\n"
            )
            self._set_text(self.image_info_text, info)
            self.set_status("图像置乱完成。")
        except Exception as e:
            messagebox.showerror("图像置乱失败", str(e))
            self.set_status("图像置乱失败。")

    def recover_current_image(self) -> None:
        try:
            if self.scrambled_image is None or self.image_perm is None:
                messagebox.showwarning("提示", "请先执行图像置乱。")
                return

            t0 = time.perf_counter()
            recovered = recover_image_by_perm(self.scrambled_image, self.image_perm)
            elapsed = time.perf_counter() - t0
            self.recovered_image = recovered
            self._show_pil_image(self.recovered_img_label, recovered)

            is_same = False
            if self.original_image is not None:
                is_same = np.array_equal(np.array(self.original_image.convert("RGB")), np.array(recovered.convert("RGB")))

            self._append_text(
                self.image_info_text,
                f"\n逆置乱恢复完成。\n恢复耗时：{elapsed:.4f} 秒\n恢复校验：{'完全一致' if is_same else '不一致，请检查参数或置乱表'}\n",
            )
            self.set_status("图像恢复完成。")
        except Exception as e:
            messagebox.showerror("图像恢复失败", str(e))
            self.set_status("图像恢复失败。")

    def save_image_results(self) -> None:
        try:
            if self.original_image is None or self.scrambled_image is None:
                messagebox.showwarning("提示", "请至少完成图像选择和置乱。")
                return

            tag = now_tag()
            map_name = self.map_var.get()
            prefix = f"{tag}_{map_name}_image"

            original_path = self.results_dir / f"{prefix}_original.png"
            scrambled_path = self.results_dir / f"{prefix}_scrambled.png"
            recovered_path = self.results_dir / f"{prefix}_recovered.png"
            info_path = self.results_dir / f"{prefix}_info.txt"

            self.original_image.save(original_path)
            self.scrambled_image.save(scrambled_path)
            if self.recovered_image is not None:
                self.recovered_image.save(recovered_path)

            info = self.image_info_text.get("1.0", "end").strip()
            info_path.write_text(info, encoding="utf-8")

            messagebox.showinfo("保存成功", f"图像结果已保存到 results 目录。\n{original_path.name}\n{scrambled_path.name}")
            self.set_status("图像结果已保存。")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))


# =========================
# 9. 程序入口
# =========================

def main() -> None:
    app = ChaosPermutationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
