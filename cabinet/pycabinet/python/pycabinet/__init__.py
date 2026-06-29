# -*- coding: utf-8 -*-
"""pycabinet - Python bindings for Cabinet HSH memory retrieval."""

try:
    from pycabinet._pycabinet import (
        Memory, QueryResult, HSHCode, Encoder, MemoryStats, DrawerStats
    )
except ImportError as _e:
    raise ImportError(
        "cabinet-hsh 的 Rust 扩展未编译。请先在项目根目录运行：\n"
        "  cd I:\\Cabinet_HSH\\库文件项目\\cabinet\n"
        "  maturin develop   # 开发安装（推荐）\n"
        "  # 或\n"
        "  maturin build --release   # 构建 wheel\n"
        f"原始错误: {_e}"
    ) from _e

__all__ = [
    "Memory",
    "QueryResult",
    "HSHCode",
    "Encoder",
    "MemoryStats",
    "DrawerStats",
    "plot",              # 延迟导入，见下
    "document_parser",   # 文档解析（延迟导入）
    "context_decoder",   # 上下文解码（延迟导入）
]


# plot 模块延迟导入，避免 matplotlib 硬依赖
class _LazyPlot:
    """延迟加载 plot 模块，仅在首次访问时导入 matplotlib。"""

    def __getattr__(self, name: str):
        import importlib
        _plot_module = importlib.import_module('pycabinet.plot')
        return getattr(_plot_module, name)


plot = _LazyPlot()


# 文档解析模块延迟导入（避免 pdfplumber/python-docx/openpyxl 硬依赖）
class _LazyDocParser:
    """延迟加载 document_parser 模块。"""

    def __getattr__(self, name: str):
        import importlib
        _dp_module = importlib.import_module('pycabinet.document_parser')
        return getattr(_dp_module, name)


document_parser = _LazyDocParser()


# 上下文解码模块直接导入（无外部依赖，轻量级）
from pycabinet.context_decoder import decode_context

__all__.append("decode_context")
