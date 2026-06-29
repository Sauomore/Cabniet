# Cabinet 🗄️

> 面向 AI Agent 的**离散语义记忆检索系统**  
> 用 20-bit 结构化整数替代 768 维浮点向量，让 AI 在纯 CPU 上记住、想起并解释"为什么想起这个"。

[![PyPI](https://img.shields.io/pypi/v/cabinet-hsh)](https://pypi.org/project/cabinet-hsh/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-green)](LICENSE)

---

## 为什么需要 Cabinet？

现有的 RAG 系统（FAISS / Milvus / Chroma）虽然强大，但在 Agent 场景中暴露了三个结构性缺陷：

| 缺陷 | 问题 | Cabinet 的解决 |
|------|------|---------------|
| **不可解释** | 向量相似度是黑箱，无法审计为什么选中某段文本 | 检索路径完全可审计：类别→簇→词，四级匹配 |
| **更新成本高** | 新增文档需要重建 IVF 聚类或整个索引 | 仅追加写入 + 后台 LSM 合并，增量更新 |
| **硬件依赖** | 需要 GPU 或大内存才能接受延迟 | 纯 CPU，索引体积缩小 ~1000 倍，笔记本即可运行 |

> **一句话**：如果你需要在普通电脑上、以极低成本、让 AI 记住大量知识并解释"为什么想起这个"——用 Cabinet。

---

## 核心概念：HSH（层次语义哈希）

Cabinet 的核心创新是 **Hierarchical Semantic Hashing (HSH)**：

每个词被编码为一个 **20-bit 结构化整数**：
```
┌───┬────────┬────────┐
│feat│  sim   │  abs   │
│4-bit│ 8-bit │ 8-bit │
└───┴────────┴────────┘
↓     ↓        ↓
类别   语义簇    簇内唯一
```
例：`0x0_15_01` = 名词(0x0) + 簇#21(0x15) + 桶内词#1(0x01)

- **feat (4-bit)**：词性类别（名词/动词/形容词...共16类）
- **sim (8-bit)**：K-means 语义簇 ID（256簇）
- **abs (8-bit)**：簇内准完美哈希（256桶）

**检索退化为 B-tree 前缀匹配**——一次 O(log n) 的整数比较，无需 GPU。

---

## 快速开始

### 从 PyPI 安装

```bash
pip install cabinet-hsh

# 含绘图支持
pip install cabinet-hsh[plot]

# 含文档解析支持（PDF/Word/Excel）
pip install cabinet-hsh[docs]

# 全量安装
pip install cabinet-hsh[plot,docs]
```

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/Sauomore/Cabinet.git
cd Cabinet/cabinet

# 编译 Python 绑定（需要 Rust + maturin）
pip install maturin
maturin build --release
pip install target/wheels/cabinet_hsh-0.1.1-*.whl
```

---

## 使用指南

### 基础编码与检索

```python
import pycabinet
from pycabinet import Encoder, Memory

# 1. 编码可视化
enc = Encoder()
codes = enc.encode_detail("用户明天下午3点开会，准备PPT。")
for w, p, c in codes:
    print(f"{w}({p}) → feat=0x{c.feat:01X}, sim=0x{c.sim:02X}, abs=0x{c.abs:02X}")

# 2. 记忆库操作
mem = Memory(
    path="./agent_memory.db",
    precision="light",
    pos_threshold=50,
    max_context=4096,
)

# 插入
mem.insert("用户明天下午3点开会，准备PPT。")
mem.insert("明天需要准备会议材料。")

# 检索（四级匹配：精确→同簇→同类→关联）
results = mem.query("会议准备", top_k=5)
for r in results:
    level = ["关联", "同类", "同簇", "精确"][r.match_level - 1]
    text = mem.decode(r)
    print(f"[{level}] score={r.score:.3f} doc_id={r.doc_id}: {text}")

# 统计
stats = mem.get_stats()
print(f"文档数: {stats.doc_count}, 精度: {stats.precision}")

# 快照
mem.snapshot("./backup.db")
mem.close()
```

### 文档解析（PDF/Word/Excel/txt/md/csv）

```python
from pycabinet import document_parser, Memory

# 解析 PDF 文档
paragraphs = document_parser.parse_document("./report.pdf")

# 批量插入到记忆库
mem = Memory(path="./agent.db")
for text in paragraphs:
    if len(text) > 10:
        mem.insert(text)

# 或一行搞定
count = document_parser.batch_insert_from_file(mem, "./report.pdf")
print(f"已插入 {count} 段")
```

支持格式：
| 格式 | 扩展名 | 说明 |
|------|--------|------|
| PDF | `.pdf` | pdfplumber 逐页提取 |
| Word | `.docx` / `.doc` | python-docx 段落提取 |
| Excel | `.xlsx` / `.xls` | openpyxl 逐单元格提取 |
| 文本 | `.txt` / `.md` / `.rst` | 按行切分 |
| CSV | `.csv` | 逐单元格提取 |

### 上下文解码（返回整句/前后词/前后句）

```python
from pycabinet import decode_context

results = mem.query("会议准备", top_k=5)
for r in results:
    # 返回整段（默认）
    print("整段:", decode_context(mem, r, mode="paragraph"))
    
    # 返回包含匹配词的整句话
    print("整句:", decode_context(mem, r, mode="sentence"))
    
    # 返回前后 3 个词（含匹配词本身）
    print("前后词:", decode_context(mem, r, mode="window", window_size=3))
    
    # 仅返回匹配词前 2 个词
    print("前词:", decode_context(mem, r, mode="before", window_size=2))
    
    # 仅返回匹配词后 2 个词
    print("后词:", decode_context(mem, r, mode="after", window_size=2))
    
    # 返回前后各 1 句话
    print("前后句:", decode_context(mem, r, mode="window_sent", window_size=1))
```

| 模式 | 说明 | 示例 |
|------|------|------|
| `paragraph` | 返回整段文档（默认） | 完整段落 |
| `sentence` | 返回包含匹配词的整句话 | 逗号/句号分隔 |
| `window` | 返回前后 `window_size` 个词 | "前后词" |
| `before` | 仅返回匹配词前 `window_size` 个词 | "前词" |
| `after` | 仅返回匹配词后 `window_size` 个词 | "后词" |
| `window_sent` | 返回前后 `window_size` 句话 | 语义窗口 |

### 可编程可视化（matplotlib）

```python
import matplotlib.pyplot as plt
from pycabinet.plot import plot_hsh_space, plot_search_results

codes = enc.encode_detail("用户明天下午3点开会，准备PPT。")
results = mem.query("会议准备", top_k=5)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_hsh_space(codes, ax=axes[0])
plot_search_results(results, ax=axes[1])
plt.savefig("cabinet_analysis.png", dpi=150)
```

### Rust 使用

```rust
use cabinet_core::{Memory, Config, QueryOpts};

fn main() -> anyhow::Result<()> {
    let mut mem = Memory::open(Config::new("./agent_memory.db"))?;
    let doc_id = mem.insert("用户明天下午3点开会，准备PPT。")?;
    let results = mem.query("会议准备", QueryOpts::default().top_k(5))?;
    for hit in results {
        println!("doc={}, level={}, score={:.3}", hit.doc_id, hit.match_level, hit.score);
    }
    mem.close()?;
    Ok(())
}
```

### Streamlit 示例 GUI（可选）

```bash
cd examples/gui
pip install streamlit pandas matplotlib
streamlit run main.py
```

6 个可视化页面：
| 页面 | 内容 |
|------|------|
| 🏠 首页 | 项目概览、HSH 结构表 |
| 🔢 编码可视化 | 分词→HSH 编码→空间散点图 |
| 🗂️ 记忆架构 | 16 抽屉网格、三层交互流程 |
| 🔍 检索路径 | 四级匹配流程、结果聚合 |
| 📁 索引浏览器 | Drawer 热力图、B-tree 分布 |
| ⚡ 操作控制台 | 插入/查询/统计（实时调用 Rust） |

---

## 三层记忆架构

Cabinet 模拟人类认知科学中的记忆分层：

```
┌──────────────────────────────────────────┐
│  Layer 3: Working Memory（工作记忆）      │
│  LRU 缓存，推理期间热点记忆，4096 tokens  │
│  命中则直接返回，O(1)                     │
└──────────────────┬───────────────────────┘
                   │ 未命中 → 查询
┌──────────────────┴───────────────────────┐
│  Layer 2: Archive Index（档案柜索引）       │
│  16 个 Feature Drawer（按词性分类）       │
│  每个 Drawer 内 B-tree 按 (sim, abs) 索引  │
│  四级匹配：精确→同簇→同类→关联            │
│  VByte + Delta 压缩                       │
└──────────────────┬───────────────────────┘
                   │ 后台合并
┌──────────────────┴───────────────────────┐
│  Layer 1: Token Store（词元层）             │
│  原始文档 HSH 序列，仅追加缓冲区            │
│  满 1000 条 → LSM 合并到 Archive           │
│  WAL 预写日志，崩溃 100% 恢复               │
└──────────────────────────────────────────┘
```

---

## 项目结构

```
cabinet/
├── crates/
│   ├── cabinet-hsh/          # 编码层：jieba 分词、POS、HSH 20-bit 编码、准完美哈希
│   ├── cabinet-index/          # 索引层：B-tree 前缀索引、LSM 合并、VByte/Delta 压缩
│   ├── cabinet-store/          # 存储层：SQLite 后端、WAL 崩溃恢复
│   ├── cabinet-router/         # 路由层：RelRouter 关联权重（硬编码 + MLP 扩展位）
│   ├── cabinet-core/           # 核心编排：Memory API、三层架构
│   ├── cabinet-cli/            # 命令行工具
│   └── cabinet-tools/          # 离线工具：聚类中心构建
├── pycabinet/                  # PyO3 Python 绑定
│   ├── src/lib.rs             # PyO3 扩展：Memory / Encoder / HSHCode / Stats
│   ├── python/pycabinet/      # Python 纯代码
│   │   ├── __init__.py        # 核心导出 + 延迟加载
│   │   ├── plot.py            # matplotlib 可编程可视化（6 个函数）
│   │   ├── document_parser.py # 文档解析（PDF/Word/Excel/txt/csv）
│   │   └── context_decoder.py # 上下文解码（6 种模式）
│   ├── Cargo.toml
│   └── pyproject.toml         # PyPI 包配置（maturin）
├── examples/                   # 示例代码
│   ├── basic.py / basic.rs
│   ├── sample_corpus.txt
│   └── gui/                   # Streamlit 示例应用（6 页面）
│       ├── main.py
│       └── pages/
├── bench/                      # Benchmark
├── paper/                      # LaTeX 学术论文（HSH 定理 + 对比实验）
│   └── cabinet_paper.tex
├── Cargo.toml                  # Workspace 根配置
└── pyproject.toml              # Python 包根配置（可选依赖 plot/docs/gui/dev）
```

---

## 性能指标（目标）

| 指标 | 目标 | 对比 |
|------|------|------|
| 索引体积 | ~2.5 bytes/词 | vs 768×4=3072 bytes/词（FAISS） |
| 压缩比 | ~1228× | 相比稠密向量 |
| 单线程编码 | > 5万词/秒 | MVP 目标 |
| 检索延迟 P99 | < 10ms | 10万文档规模 |
| 内存占用 | ~4MB + SQLite | 无需 GPU |
| 增量写入 | 毫秒级 | 无需重建索引 |

---

## 使用场景

| 场景 | 为什么用 Cabinet | 为什么不用 FAISS |
|------|----------------|------------------|
| 个人 AI 助手 | 笔记本本地运行，记住用户历史偏好 | GPU 不现实，向量库太大 |
| 边缘设备 | 工业控制器/树莓派，4MB 内存足够 | 需要 GPU 或大内存 |
| 高频增量 | 每日万级新闻流，秒级写入 | 每次重建 IVF 分钟级延迟 |
| 可解释审计 | 法律/医疗系统必须解释为什么选中某段 | 向量相似度是黑箱 |
| 课程/论文原型 | 技术亮点抓眼球，可解释性强 | 缺乏差异化创新点 |

---

## 开发

```bash
# 克隆仓库
git clone https://github.com/Sauomore/Cabinet.git
cd Cabinet/cabinet

# 运行 Rust 测试
cargo test --workspace

# 格式化 + 检查
cargo fmt
cargo clippy --workspace

# 构建 CLI
cargo build --release -p cabinet-cli

# 编译 Python 绑定（开发模式）
cd pycabinet
maturin develop
```

---

## 技术路线

- **v0.1.1（当前）**：Light 精度、SQLite 后端、硬编码 Router、文档解析、上下文解码、matplotlib 可视化
- **v0.5**：Hybrid 精度（热桶保留残差）、RelRouter MLP（ONNX）、SIMD 加速
- **v1.0**：RocksDB 后端、自定义词典、列族映射、SST 快照、真实 BERT 向量 + 离线 K-means 聚类

---

## License

MIT OR Apache-2.0

**Cabinet — 让 AI 记住并解释它记住了什么。**
