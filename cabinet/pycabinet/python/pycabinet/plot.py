# -*- coding: utf-8 -*-
"""可编程可视化工具：matplotlib 辅助函数，不依赖 Streamlit。

使用示例：
    import pycabinet
    from pycabinet.plot import plot_hsh_space, plot_drawer_heatmap

    mem = pycabinet.Memory(path="./demo.db")

    # 在 Jupyter 中直接画 HSH 编码空间
    codes = mem.encode_detail("用户明天下午3点开会")
    plot_hsh_space(codes)
    plt.show()

    # 画 Drawer 热力图
    stats = mem.get_drawer_stats(0x0)
    plot_drawer_heatmap(stats)
    plt.show()

    # 完全自定义子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_hsh_space(codes, ax=axes[0])
    plot_drawer_heatmap(stats, ax=axes[1])
    plt.tight_layout()
    plt.savefig("my_report.png", dpi=300)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# 延迟导入 matplotlib，避免硬依赖
_imported = False

def _ensure_matplotlib():
    global _imported
    if _imported:
        return
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib 未安装。请运行：pip install matplotlib\n"
            "或使用：pip install pycabinet[plot]"
        ) from exc
    _imported = True


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

FEAT_NAMES = {
    0x0: "名词", 0x1: "动词", 0x2: "形容词", 0x3: "副词",
    0x4: "代词", 0x5: "介词", 0x6: "连词", 0x7: "助词",
    0x8: "数词", 0x9: "量词", 0xA: "时间词", 0xB: "方位词",
    0xC: "标点", 0xD: "字符串", 0xE: "常用词", 0xF: "兜底",
}

LEVEL_COLORS = {4: "#E74C3C", 3: "#E67E22", 2: "#F1C40F", 1: "#9B59B6"}
LEVEL_NAMES = {4: "精确", 3: "同簇", 2: "同类", 1: "关联"}


def _try_setup_cjk_font():
    """尝试设置 matplotlib 中文字体，失败时静默忽略。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        candidates = [
            "Microsoft YaHei", "SimHei", "SimSun",
            "WenQuanYi Micro Hei", "Noto Sans CJK SC",
            "PingFang SC", "Heiti SC", "Arial Unicode MS",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        selected = next((f for f in candidates if f in available), None)
        if selected:
            plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
        else:
            cjk_fonts = [
                f.name for f in fm.fontManager.ttflist
                if "CJK" in f.name or "Hei" in f.name or "YaHei" in f.name or "Sim" in f.name
            ]
            if cjk_fonts:
                plt.rcParams["font.sans-serif"] = [cjk_fonts[0], "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# 首次导入时尝试设置字体
_try_setup_cjk_font()


# ---------------------------------------------------------------------------
# 绘图函数
# ---------------------------------------------------------------------------

def plot_hsh_space(
    codes,  # List[Tuple[str, str, HSHCode]] 来自 encode_detail()
    figsize: Tuple[float, float] = (8, 8),
    ax=None,
):
    """绘制 HSH 编码空间散点图。

    Args:
        codes: [(word, pos_tag, HSHCode), ...] 来自 ``encode_detail()``。
        figsize: 当 *ax* 为 None 时创建新图形的尺寸。
        ax: 可选的 matplotlib Axes，用于嵌入子图。

    Returns:
        matplotlib Axes 对象。
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # 按 feat 分组着色
    groups = {}
    for word, pos, hsh in codes:
        feat = hsh.feat
        sim = hsh.sim
        abs = hsh.abs
        groups.setdefault(feat, []).append((word, sim, abs))

    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
    for idx, (feat, pts) in enumerate(sorted(groups.items())):
        sims = [p[1] for p in pts]
        abss = [p[2] for p in pts]
        label = f"0x{feat:01X} {FEAT_NAMES.get(feat, '')}"
        ax.scatter(sims, abss, alpha=0.8, s=120, edgecolors="white", linewidths=1,
                   color=colors[idx], label=label)
        for word, s, a in pts:
            ax.annotate(word, (s, a), fontsize=8, ha="center", va="bottom")

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_xlabel("sim (8-bit)")
    ax.set_ylabel("abs (8-bit)")
    ax.set_title("HSH 编码空间分布（256 × 256）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    return ax


def plot_drawer_heatmap(
    drawer_stats,
    figsize: Tuple[float, float] = (10, 10),
    ax=None,
):
    """绘制 Drawer 文档密度热力图。

    Args:
        drawer_stats: ``DrawerStats`` 对象（来自 ``mem.get_drawer_stats(feat)``）。
        figsize: 当 *ax* 为 None 时创建新图形的尺寸。
        ax: 可选的 matplotlib Axes。

    Returns:
        matplotlib Axes 对象。
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    grid = np.zeros((16, 16))
    for key, doc_count, _ in drawer_stats.keys:
        sim = (key >> 8) & 0xFF
        abs = key & 0xFF
        gx = sim // 16
        gy = abs // 16
        grid[gy, gx] += doc_count

    im = ax.imshow(grid, cmap="YlOrRd", aspect="equal")
    ax.set_xlabel("sim // 16")
    ax.set_ylabel("abs // 16")
    ax.set_title(f"Drawer 0x{drawer_stats.feat:01X} — 文档密度热力图 (16×16 网格)")
    plt.colorbar(im, ax=ax, label="文档引用数")

    for i in range(16):
        for j in range(16):
            if grid[i, j] > 0:
                ax.text(
                    j, i, f"{int(grid[i, j])}",
                    ha="center", va="center", fontsize=6,
                    color="white" if grid[i, j] > grid.max() * 0.5 else "black",
                )
    return ax


def plot_drawer_btree(
    drawer_stats,
    figsize: Tuple[float, float] = (14, 4),
    ax=None,
):
    """绘制 Drawer B-tree 叶子节点分布。

    Args:
        drawer_stats: ``DrawerStats`` 对象。
        figsize: 当 *ax* 为 None 时创建新图形的尺寸。
        ax: 可选的 matplotlib Axes。

    Returns:
        matplotlib Axes 对象。
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    keys = sorted(drawer_stats.keys, key=lambda k: k[0])
    x_pos = np.arange(len(keys))
    doc_counts = [doc_count for _, doc_count, _ in keys]
    colors = plt.cm.viridis(np.array(doc_counts) / max(doc_counts, default=1))

    ax.bar(x_pos, doc_counts, color=colors, width=1.0, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("B-tree 叶子节点顺序 (key sorted)")
    ax.set_ylabel("文档引用数")
    ax.set_title(f"Drawer 0x{drawer_stats.feat:01X} — B-tree 叶子节点分布")
    ax.set_xlim(0, len(keys))
    return ax


def plot_search_results(
    results,
    figsize: Tuple[float, float] = (10, 4),
    ax=None,
):
    """绘制检索结果得分条形图。

    Args:
        results: ``QueryResult`` 列表（来自 ``mem.query()``）。
        figsize: 当 *ax* 为 None 时创建新图形的尺寸。
        ax: 可选的 matplotlib Axes。

    Returns:
        matplotlib Axes 对象。
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if not results:
        ax.text(0.5, 0.5, "无检索结果", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return ax

    doc_labels = [f"doc_{r.doc_id}" for r in results]
    scores = [r.score for r in results]
    bar_colors = [LEVEL_COLORS.get(r.match_level, "#95A5A6") for r in results]

    bars = ax.barh(doc_labels, scores, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Score")
    ax.set_title("检索结果得分")
    ax.set_xlim(0, 1.1)

    for bar, r in zip(bars, results):
        level = LEVEL_NAMES.get(r.match_level, "?")
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{level} ({r.score:.3f})",
            va="center", fontsize=9,
        )
    return ax


def plot_memory_stats(
    stats,
    figsize: Tuple[float, float] = (10, 4),
    ax=None,
):
    """绘制记忆库统计概览。

    Args:
        stats: ``MemoryStats`` 对象（来自 ``mem.get_stats()``）。
        figsize: 当 *ax* 为 None 时创建新图形的尺寸。
        ax: 可选的 matplotlib Axes。

    Returns:
        matplotlib Axes 对象。
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    labels = ["文档数", "WM 已用", "WM 容量", "Token Buffer"]
    values = [
        stats.doc_count,
        stats.working_memory_used,
        stats.working_memory_capacity,
        stats.token_store_buffer_size,
    ]
    colors = ["#3498DB", "#E74C3C", "#95A5A6", "#2ECC71"]

    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    ax.set_ylabel("数量")
    ax.set_title(f"Cabinet 统计概览（精度: {stats.precision}）")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10)
    return ax


def plot_feat_distribution(
    codes: List[Tuple[str, str, int]],
    figsize: Tuple[float, float] = (10, 4),
    ax=None,
):
    """绘制特征码分布柱状图。

    Args:
        codes: [(word, pos_tag, hsh_raw), ...] 来自 ``encode_detail()``。
        figsize: 当 *ax* 为 None 时创建新图形的尺寸。
        ax: 可选的 matplotlib Axes。

    Returns:
        matplotlib Axes 对象。
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    from collections import Counter
    feat_counts = Counter(h.feat for _, _, h in codes)
    labels = [f"0x{i:01X}\n{FEAT_NAMES.get(i, '')}" for i in range(16)]
    values = [feat_counts.get(i, 0) for i in range(16)]

    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    bars = ax.bar(range(16), values, color=colors, edgecolor="white")
    ax.set_xticks(range(16))
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("词数")
    ax.set_title("各语义类别 (feat) 词汇分布")

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, str(int(h)),
                    ha="center", va="bottom", fontsize=8)
    return ax
