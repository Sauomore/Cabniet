# Cabinet 🗄️

> 面向 AI Agent 的**离散语义记忆检索系统**  
> 用 20-bit / 32-bit 结构化整数替代 768 维浮点向量，让 AI 在纯 CPU 上记住、想起并解释"为什么想起这个"。

[![Rust](https://img.shields.io/badge/rust-1.72%2B-orange)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

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

Cabinet 的核心创新是 **Hierarchical Semantic Hashing (HSH)**，提供两种精度模式：

### 1. HSH-20（基础版，20-bit）

每个词被编码为一个 **20-bit 结构化整数**：

```
┌───┬────────┬────────┐
│feat│  sim   │  abs   │
│4-bit│ 8-bit │ 8-bit │
└───┴────────┴────────┘
  ↓     ↓        ↓
类别   语义簇    簇内唯一

例：0x0_15_01 = 名词(0x0) + 簇#21(0x15) + 桶内词#1(0x01)
```

- **feat (4-bit)**：词性类别（名词/动词/形容词...共16类）
- **sim (8-bit)**：K-means 语义簇 ID（256簇）
- **abs (8-bit)**：簇内准完美哈希（256桶）

**检索退化为 B-tree 前缀匹配**——一次 O(log n) 的整数比较，无需 GPU。

### 2. HSH-32（扩展版，32-bit，含反义词检测）

基于论文 *HSH-32: Hierarchical Semantic Hashing with 32-bit Continuous Direction Encoding*，每个词被编码为一个 **32-bit 结构化整数**：

```
┌───┬────────────┬────────┐
│feat│     sim     │  abs   │
│4-bit│  20-bit   │ 8-bit │
└───┴────────────┴────────┘
  ↓         ↓          ↓
类别    语义方向码    簇内唯一

例：0xA_42_123_F0 = 形容词(0xA) + 方向码0x42123(20-bit) + 桶内词#0xF0
```

- **feat (4-bit)**：词性类别（同 HSH-20）
- **sim (20-bit)**：PCA 降维 + Sign 量化后的语义方向码（约 100 万方向）
- **abs (8-bit)**：簇内准完美哈希（同 HSH-20）

**HSH-32 的额外能力**：
- ✅ **反义词检测**：通过语义方向码的按位取反关系检测反义词（如"好"vs"坏"），评分自动乘以 -0.5
- ✅ **方向评分**：20-bit 方向码提供约 100 万唯一方向，语义区分度远高于 8-bit（256 簇）
- ✅ **连续方向编码**：PCA 投影 + Sign 量化将 768-dim BERT 向量压缩为 20-bit 方向码，保留约 70%~85% 方差
- ✅ **编码空间**：16 × 2^20 × 256 ≈ 43 亿唯一编码，支持大规模词典

---

## 快速开始

### 安装

```bash
# Python 用户（推荐）
pip install pycabinet

# 含可视化 GUI
pip install pycabinet[gui]

# Rust 开发者
cargo install cabinet-cli

# Docker 用户
docker-compose up -d
```

### Python 使用（HSH-20）

```python
import pycabinet

# 初始化记忆库（~4MB 内存 + SQLite 单文件）
mem = pycabinet.Memory(
    path="./agent_memory.db",
    precision="light",    # light | hybrid | precise
    pos_threshold=50,      # 常用词晋升阈值
    max_context=4096,      # 工作记忆窗口
)

# 插入记忆（自动分词、编码、入库）
mem.insert("用户明天下午3点开会，准备PPT。")
mem.insert("用户喜欢听管弦乐。")
mem.insert("5号楼邻居有梯子，平时放在车库。")

# 检索记忆（四级匹配：精确→同簇→同类→关联）
results = mem.query("会议准备", top_k=5)
for r in results:
    level = ["关联", "同类", "同簇", "精确"][r.match_level - 1]
    print(f"[{level}] score={r.score:.3f} doc_id={r.doc_id}")
    if r.match_level >= 3:  # 只解码高置信度
        text = mem.decode(r)
        print(f"  → {text}")

# 快照与迁移
mem.snapshot("./backup/2026-06-25.db")
mem.close()
```

### Python 使用（HSH-32，含反义词检测）

```python
from pycabinet import Encoder, hsh32_is_antonym, hsh32_compute_score

# 1. HSH-32 编码可视化
enc = Encoder()
codes = enc.encode_hsh32("用户喜欢听管弦乐，不喜欢噪音。")
for c in codes:
    print(f"HSH-32: feat=0x{c.feat:01X}, sim=0x{c.sim:05X}, abs=0x{c.abs:02X}")

# 2. 反义词检测
sim_a = codes[0].sim   # "喜欢" 的方向码
sim_b = codes[3].sim   # "不喜欢" 的方向码
if hsh32_is_antonym(sim_a, sim_b):
    print("检测到反义词！检索时评分将乘以 -0.5")

# 3. 综合评分计算（方向 + 位置 + 频率）
score, is_ant = hsh32_compute_score(sim_a, sim_b, 0, 0, 1.0)
print(f"Score={score:.3f}, is_antonym={is_ant}")
```

### Rust 使用（HSH-32）

```rust
use cabinet_hsh::{Encoder32, HSHCode32, is_antonym, query_hsh32, CandidateDoc};
use std::collections::HashMap;

fn main() {
    // 编码
    let encoder = Encoder32::new();
    let code = encoder.encode_word("测试", 0x01);
    println!("HSH-32: feat={}, sim=0x{:05X}, abs=0x{:02X}", 
             code.feat(), code.sim(), code.abs());

    // 反义词检测
    assert!(is_antonym(0, 0xF_FFFF, 4096));  // 精确反义
    assert!(!is_antonym(0, 100_000, 4096));  // 非反义

    // 查询（在 [sim_q - delta, sim_q + delta] 范围内扫描）
    let mut buckets = HashMap::new();
    // ... 填充 bucket 数据
    let query = HSHCode32::new(0, 50_000, 0);
    let results = query_hsh32(query, &buckets, 5, 1024, 0.7, 0.2, 0.1, 4096);
}
```

### CLI 命令行

```bash
# 编码演示（HSH-20）
cabinet encode "明天下午3点开会"
# 输出：6 HSH codes → feat=0x0 sim=0xXX abs=0xXX

# HSH-32 编码（长方向码）
cabinet encode32 "明天下午3点开会"
# 输出：6 HSH-32 codes → feat=0x0 sim=0xXXXXX abs=0xXX

# 插入记忆
cabinet insert "准备PPT材料"

# 检索记忆
cabinet query "会议准备" --top-k 5
# 输出：
#   1. [EXACT] score=0.950 doc_id=1
#   2. [CLUSTER] score=0.720 doc_id=5

# 批量导入（从文本文件，每行一条）
cabinet batch ./news_corpus.txt

# 查看统计
cabinet stats

# 创建快照
cabinet snapshot ./backup.db
```

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

## 可视化界面

```bash
cd cabinet-gui
pip install -r requirements.txt
streamlit run app.py
```

或双击 `启动.bat`（Windows）

6 个可视化页面：

| 页面 | 内容 |
|------|------|
| 🏠 **首页** | 项目概览、核心指标、HSH 结构表 |
| 🔢 **编码可视化** | 分词→HSH 编码→二进制拆解→空间散点图 |
| 🗂️ **记忆架构** | 16 抽屉网格图、三层交互流程图 |
| 🔍 **检索路径** | 查询词四级匹配流程图、文档聚合排序 |
| 📁 **索引浏览器** | Drawer 热力图、B-tree 叶子节点分布、前缀扫描演示 |
| ⚡ **操作控制台** | 插入/查询/WAL 日志/系统统计（交互式） |

---

## 项目结构

```
cabinet/
├── crates/
│   ├── cabinet-hsh/          # 编码层：分词、POS、HSH 20-bit/32-bit 编码、准完美哈希
│   │   ├── src/hsh_code.rs   # HSH-20 编码结构
│   │   ├── src/hsh32.rs      # HSH-32 编码：PCA + Sign 量化 + 反义词检测 + 查询算法
│   │   ├── src/encoder.rs    # Encoder（支持 HSH-20 + HSH-32）
│   │   ├── src/cluster.rs    # 聚类中心 + mock_embed
│   │   ├── src/perfect_hash.rs  # 种子表 + 准完美哈希
│   │   ├── src/pos_map.rs    # 词性映射表
│   │   └── src/error.rs      # 编码错误类型
│   ├── cabinet-index/          # 索引层：B-tree 前缀索引、LSM 合并、VByte/Delta 压缩
│   ├── cabinet-store/          # 存储层：SQLite 后端、WAL 崩溃恢复、原子替换
│   ├── cabinet-router/         # 路由层：RelRouter 关联权重（硬编码 + MLP 扩展）
│   ├── cabinet-core/           # 核心编排：Memory API、三层架构、并发控制
│   ├── cabinet-cli/            # 命令行工具：insert/query/batch/stats/encode/encode32
│   └── cabinet-tools/          # 离线工具：聚类中心构建、种子表搜索
├── pycabinet/                  # PyO3 Python 绑定（pip install pycabinet）
│   ├── src/lib.rs             # Python Memory 类 + HSH-32 辅助函数
│   ├── pyproject.toml         # PyPI 发布配置
│   └── tests/                 # Python 集成测试
├── cabinet-gui/                # Streamlit 可视化应用（独立目录）
│   ├── app.py                 # 主入口
│   ├── pages/                 # 6 个可视化页面
│   ├── utils.py               # 中文字体配置
│   └── 启动.bat / 启动.ps1   # 一键启动脚本
├── examples/                   # 示例代码和数据
│   ├── basic.rs               # Rust 示例（HSH-20）
│   ├── basic_hsh32.rs          # Rust 示例（HSH-32）
│   ├── basic.py               # Python 示例（HSH-20）
│   └── basic_hsh32.py        # Python 示例（HSH-32）
│   └── sample_corpus.txt      # 10 条社区互助语料
├── bench/                      # 基准测试（insert/query 吞吐量）
├── scripts/                    # 构建和测试脚本
│   ├── build-release.bat      # Windows 构建 wheel
│   ├── build-release.sh       # Linux/macOS 构建
│   └── test-install.py        # 安装后功能验证
├── .github/workflows/          # CI/CD
│   ├── ci.yml                 # PR 自动测试
│   └── publish.yml            # 推 tag 自动发布 PyPI wheel
├── Cargo.toml                 # Workspace 根配置（8 个 crate）
├── Makefile                   # make build/test/fmt/clippy
├── Dockerfile                 # 多阶段构建
├── docker-compose.yml         # 组合启动
├── config.toml                # 运行时配置（路径/精度/阈值）
├── CONTRIBUTING.md            # 贡献指南
├── CHANGELOG.md             # 版本变更
└── README.md                  # 本文档
```

---

## HSH-32 技术细节

### 编码流程

```
输入词 → jieba 分词 + POS → 获取 BERT 向量(768-dim)
                                      ↓
                              PCA 投影(20-dim)
                                      ↓
                              Sign 量化 → 20-bit sim
                                      ↓
                              完美哈希 → 8-bit abs
                                      ↓
                           HSHCode32(feat:4, sim:20, abs:8)
```

### 反义词检测原理

论文定理：当两个向量方向相反（夹角 ≈ 180°）时，它们的 20-bit sim 码在二进制上呈现按位取反关系。

```
判定条件：|sim_a + sim_b - 0xFFFFF| <= tolerance

例：sim_a = 0x00000 (全 0)  ← 反义 →  sim_b = 0xFFFFF (全 1)
   sim_a = 0x80000          ← 反义 →  sim_b = 0x7FFFF

默认 tolerance = 4096（约 0.4% 误差容忍）
```

检测到反义词后，检索评分自动乘以 **-0.5**，有效降低语义相反文档的排序。

### 查询评分公式

```
Score(q, d) = α·DirectionScore + β·PositionScore + γ·FrequencyScore

其中：
- DirectionScore = 1 - |sim_q - sim_d| / 2^20   （方向相似度，范围 [0,1]）
- PositionScore  = 1 - |abs_q - abs_d| / 255     （位置接近度，范围 [0,1]）
- FrequencyScore = (1 + ln(freq)) / 2            （频率评分，范围 [0,1]）

论文默认值：α = 0.7, β = 0.2, γ = 0.1
反义词惩罚：若 is_antonym，score *= -0.5
```

### 查询算法

在 `[sim_q - delta, sim_q + delta]` 范围内扫描所有候选 bucket，收集文档并按评分降序返回 `top_k`。

```rust
let results = query_hsh32(
    query_code,     // HSHCode32
    &buckets,       // HashMap<bucket_id, Vec<CandidateDoc>>
    top_k,          // 返回数量
    delta,          // 扫描范围（默认 1024）
    alpha, beta, gamma,  // 评分权重
    tolerance,      // 反义词检测阈值
);
```

---

## 性能指标（目标）

| 指标 | HSH-20 | HSH-32 | 对比 |
|------|--------|--------|------|
| 索引体积 | ~2.5 bytes/词 | ~4 bytes/词 | vs 768×4=3072 bytes/词（FAISS） |
| 压缩比 | ~1228× | ~768× | 相比稠密向量 |
| 单线程编码 | > 5万词/秒 | > 3万词/秒 | MVP 目标 |
| 检索延迟 P99 | < 10ms | < 15ms | 10万文档规模 |
| 内存占用 | ~4MB + SQLite | ~6MB + SQLite | 无需 GPU |
| 增量写入 | 毫秒级 | 毫秒级 | 无需重建索引 |
| 语义区分度 | 256 方向 | ~100万 方向 | HSH-32 提升 ~4096× |
| 反义词检测 | ❌ | ✅ | HSH-32 独有 |

---

## 使用场景

| 场景 | 为什么用 Cabinet | 为什么不用 FAISS |
|------|---------------|-----------------|
| **个人 AI 助手** | 笔记本本地运行，记住用户历史偏好 | GPU 不现实，向量库太大 |
| **边缘设备** | 工业控制器/树莓派，4MB 内存足够 | 需要 GPU 或大内存 |
| **高频增量** | 每日万级新闻流，秒级写入 | 每次重建 IVF 分钟级延迟 |
| **可解释审计** | 法律/医疗系统必须解释为什么选中某段 | 向量相似度是黑箱 |
| **语义反义词过滤** | 自动检测并降低"喜欢"vs"讨厌"的误匹配 | 无法区分反义方向 |
| **课程/论文原型** | 技术亮点抓眼球，可解释性强 | 缺乏差异化创新点 |

---

## 开发

```bash
# 克隆仓库
git clone https://github.com/yourname/cabinet
cd cabinet

# 运行测试
cargo test --workspace
make test

# 格式化 + 检查
cargo fmt
cargo clippy --workspace -- -D warnings
make clippy

# 构建 CLI
cargo build --release -p cabinet-cli

# 安装 Python 绑定（开发模式）
cd pycabinet
maturin develop
python ../scripts/test-install.py

# 启动可视化
cd ../cabinet-gui
streamlit run app.py
```

---

## 技术路线

- **MVP v0.1.0**（当前）：Light 精度、SQLite 后端、硬编码 Router、jieba 分词、**HSH-32 编码** ✅
- **v0.2**：HSH-32 端到端集成（Memory 层支持 HSH-32 索引与查询）
- **v0.5**：Hybrid 精度（热桶保留残差）、RelRouter MLP（ONNX）、SIMD 加速
- **v1.0**：RocksDB 后端、自定义词典、列族映射、SST 快照、真实 BERT 向量 + 离线 PCA 训练

详见 [技术路线支持文档](../技术路线支持.md)。

---

## 贡献

欢迎 Issue 和 PR！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

核心设计原则：
- 编码层零 IO（纯内存计算）
- 索引层零存储细节（抽象字节流）
- 存储层可插拔（SQLite → RocksDB）
- Python 层薄如纸（只做类型转换）

---

## License

MIT OR Apache-2.0

---

> **Cabinet** — 让 AI 记住并解释它记住了什么。  
> 构建中，欢迎试用和反馈！
