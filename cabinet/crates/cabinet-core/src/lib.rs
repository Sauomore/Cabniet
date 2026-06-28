//! Cabinet Core：核心编排层
//!
//! 整合编码层、索引层、存储层、路由层，提供高层 Memory API。
//!
//! 三层记忆架构：
//! 1. Token Store（Layer 1）：原始文档 HSH 序列，append-only
//! 2. Archive Index（Layer 2）：按 feat 分组的倒排索引
//! 3. Working Memory（Layer 3）：推理期间热点缓存

use cabinet_hsh::{Encoder, EncoderConfig, HSHCode};
use cabinet_index::{
    archive_index::{ArchiveIndex, Hit},
    token_store::{TokenRecord, TokenStore},
    working_memory::{MemorySnippet, WorkingMemory},
};
use cabinet_router::Router;
use cabinet_store::{Backend, SQLiteBackend, StoreConfig, WalRecord, WalType};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("编码错误: {0}")]
    Encode(#[from] cabinet_hsh::error::EncodeError),
    #[error("存储错误: {0}")]
    Store(String),
    #[error("索引错误: {0}")]
    Index(String),
    #[error("配置错误: {0}")]
    Config(String),
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
}

/// 精度模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// 纯 HSH，最轻量
    Light,
    /// 热桶保留残差向量（v0.5）
    Hybrid,
    /// 全残差向量（v1.0）
    Precise,
}

impl Precision {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "light" | "Light" => Some(Precision::Light),
            "hybrid" | "Hybrid" => Some(Precision::Hybrid),
            "precise" | "Precise" => Some(Precision::Precise),
            _ => None,
        }
    }
}

/// 配置
pub struct Config {
    pub path: PathBuf,
    pub precision: Precision,
    pub working_memory_size: usize,
    pub pos_threshold: u32,
    pub sim_threshold: f32,
    pub backend_type: String,
    pub merge_interval: usize,
    pub wal_sync: bool,
}

impl Config {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Config {
            path: path.as_ref().to_path_buf(),
            precision: Precision::Light,
            working_memory_size: 4096,
            pos_threshold: 50,
            sim_threshold: 0.5,
            backend_type: "sqlite".to_string(),
            merge_interval: 1000,
            wal_sync: true,
        }
    }

    pub fn precision(mut self, p: Precision) -> Self {
        self.precision = p;
        self
    }

    pub fn working_memory_size(mut self, size: usize) -> Self {
        self.working_memory_size = size;
        self
    }

    pub fn pos_threshold(mut self, t: u32) -> Self {
        self.pos_threshold = t;
        self
    }
}

/// 查询选项
#[derive(Debug, Clone, Default)]
pub struct QueryOpts {
    pub top_k: usize,
    pub min_match_level: u8,
    pub include_text: bool,
}

impl QueryOpts {
    pub fn new() -> Self {
        QueryOpts {
            top_k: 5,
            min_match_level: 1,
            include_text: false,
        }
    }

    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn min_match_level(mut self, level: u8) -> Self {
        self.min_match_level = level;
        self
    }
}

/// 查询结果
#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    pub hsh: u32,         // 20-bit HSH 编码
    pub doc_id: u64,
    pub position: u32,
    pub match_level: u8, // 4=exact, 3=cluster, 2=category, 1=related
    pub score: f32,
    pub text: Option<String>, // 仅当 include_text=true 或 match_level>=3 时填充
}

impl QueryResult {
    fn from_hit(hit: Hit, text: Option<String>) -> Self {
        QueryResult {
            hsh: 0, // 由查询层填充
            doc_id: hit.doc_id,
            position: hit.position,
            match_level: hit.match_level,
            score: hit.score,
            text,
        }
    }
}

/// 记忆库统计信息
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub doc_count: usize,
    pub next_doc_id: u64,
    pub working_memory_capacity: usize,
    pub working_memory_used: usize,
    pub token_store_buffer_size: usize,
    pub precision: String,
}

/// Drawer 统计信息
#[derive(Debug, Clone)]
pub struct DrawerStats {
    pub feat: u8,
    pub key_count: usize,
    pub total_doc_refs: usize,
    pub keys: Vec<(u16, u32, usize)>, // (key, doc_count, posting_bytes)
}

/// Memory：核心 API
pub struct Memory {
    config: Config,
    encoder: Encoder,
    token_store: TokenStore,
    archive_index: ArchiveIndex,
    working_memory: WorkingMemory,
    router: Router,
    backend: Option<SQLiteBackend>, // MVP 只用 SQLite
    next_doc_id: u64,
    // 原始文本缓存（用于 decode）
    text_cache: std::collections::HashMap<u64, String>,
}

impl Memory {
    /// 打开或创建记忆库
    pub fn open(config: Config) -> Result<Self, MemoryError> {
        let encoder = Encoder::with_config(EncoderConfig {
            pos_threshold: config.pos_threshold,
            sim_threshold: config.sim_threshold,
            ..Default::default()
        })?;

        let token_store = TokenStore::new(config.merge_interval);
        let archive_index = ArchiveIndex::new();
        let working_memory = WorkingMemory::new(config.working_memory_size);
        let router = Router::new();

        // 尝试打开后端存储
        let backend = if config.backend_type == "sqlite" {
            let store_config = StoreConfig {
                path: config.path.to_string_lossy().to_string(),
                backend_type: cabinet_store::BackendType::SQLite,
                wal_sync: config.wal_sync,
            };
            Some(
                SQLiteBackend::open(&config.path, &store_config)
                    .map_err(|e| MemoryError::Store(e.to_string()))?,
            )
        } else {
            None
        };

        let mut memory = Memory {
            config,
            encoder,
            token_store,
            archive_index,
            working_memory,
            router,
            backend,
            next_doc_id: 1,
            text_cache: std::collections::HashMap::new(),
        };

        // 尝试从 WAL 恢复
        memory.recover_wal()?;

        Ok(memory)
    }

    /// 插入记忆
    pub fn insert(&mut self, text: &str) -> Result<u64, MemoryError> {
        let hsh_seq = self.encoder.encode(text)?;
        let doc_id = self.next_doc_id;
        self.next_doc_id += 1;

        // 存储原始文本
        self.text_cache.insert(doc_id, text.to_string());

        // 写入 Token Store（不强制 doc_id 同步，token_store 内部维护缓冲区索引）
        let _token_doc_id = self.token_store.insert(hsh_seq.clone());

        // 写入 Archive Index
        self.archive_index.merge_from_hsh_seq(doc_id, &hsh_seq);

        // 写入后端（如果存在）
        if let Some(ref backend) = self.backend {
            let record = WalRecord {
                record_type: WalType::Insert,
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                doc_id,
                hsh_seq: hsh_seq.clone(),
            };
            backend
                .append_wal(&record)
                .map_err(|e| MemoryError::Store(e.to_string()))?;
            backend
                .write_token(doc_id, &hsh_seq)
                .map_err(|e| MemoryError::Store(e.to_string()))?;
        }

        // 检查是否需要 LSM 合并
        if self.token_store.should_merge() {
            self.merge()?;
        }

        Ok(doc_id)
    }

    /// 检索记忆
    pub fn query(&mut self, text: &str, opts: QueryOpts) -> Result<Vec<QueryResult>, MemoryError> {
        let query_hsh_seq = self.encoder.encode(text)?;
        let mut all_results = Vec::new();

        for q_hsh in query_hsh_seq {
            // 先查工作记忆
            if let Some(snippet) = self.working_memory.query(q_hsh) {
                all_results.push(QueryResult {
                    hsh: q_hsh.raw(),
                    doc_id: snippet.doc_id,
                    position: 0,
                    match_level: 4,
                    score: 1.0,
                    text: Some(snippet.text.clone()),
                });
            }

            // 再查 Archive Index
            let related = self.router.related_feats_for_hsh(q_hsh);
            let hits = self.archive_index.query(q_hsh, &related);
            for hit in hits {
                if hit.match_level >= opts.min_match_level {
                    let text = if opts.include_text || hit.match_level >= 3 {
                        self.text_cache.get(&hit.doc_id).cloned()
                    } else {
                        None
                    };
                    let mut qr = QueryResult::from_hit(hit, text);
                    qr.hsh = q_hsh.raw();
                    all_results.push(qr);
                }
            }
        }

        // 去重、聚合、排序
        all_results = Self::dedup_and_sort(all_results);
        all_results.truncate(opts.top_k);

        Ok(all_results)
    }

    /// 解码查询结果为文本
    pub fn decode(&self, result: &QueryResult) -> Option<String> {
        self.text_cache.get(&result.doc_id).cloned()
    }

    /// 逻辑运算：扫描单个桶（feat, sim）
    pub fn scan_bucket(&self, feat: u8, sim: u8) -> Vec<u64> {
        let drawer = self.archive_index.drawer(feat);
        let mut doc_ids = Vec::new();
        for key in 0..=255u16 {
            let full_key = ((sim as u16) << 8) | key;
            if let Some(pl) = drawer.tree.get(&full_key) {
                doc_ids.extend(pl.doc_ids());
            }
        }
        doc_ids.sort_unstable();
        doc_ids.dedup();
        doc_ids
    }

    /// 批量插入
    pub fn insert_batch(&mut self, texts: &[String]) -> Result<Vec<u64>, MemoryError> {
        let mut ids = Vec::with_capacity(texts.len());
        for text in texts {
            ids.push(self.insert(text)?);
        }
        Ok(ids)
    }

    /// 快照备份
    pub fn snapshot(&self, dst: &Path) -> Result<(), MemoryError> {
        if let Some(ref backend) = self.backend {
            backend
                .snapshot(dst)
                .map_err(|e| MemoryError::Store(e.to_string()))?;
        }
        Ok(())
    }

    /// 关闭（同步数据）
    pub fn close(&mut self) -> Result<(), MemoryError> {
        self.merge()?;
        // 后端关闭由 Drop 自动处理
        Ok(())
    }

    /// 编码文本，返回每个词的 (word, pos, HSHCode)
    pub fn encode_detail(&self, text: &str) -> Result<Vec<(String, String, HSHCode)>, MemoryError> {
        self.encoder.encode_detail(text)
            .map_err(MemoryError::Encode)
    }

    /// 获取记忆库统计信息
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            doc_count: self.text_cache.len(),
            next_doc_id: self.next_doc_id,
            working_memory_capacity: self.working_memory.max_tokens,
            working_memory_used: self.working_memory.len(),
            token_store_buffer_size: self.token_store.len(),
            precision: format!("{:?}", self.config.precision),
        }
    }

    /// 获取指定 Drawer 的统计信息
    pub fn get_drawer_stats(&self, feat: u8) -> DrawerStats {
        let drawer = self.archive_index.drawer(feat);
        let mut keys = Vec::new();
        let mut total_doc_refs = 0usize;
        for (key, pl) in &drawer.tree {
            total_doc_refs += pl.doc_count as usize;
            keys.push((*key, pl.doc_count, pl.to_bytes().len()));
        }
        DrawerStats {
            feat,
            key_count: keys.len(),
            total_doc_refs,
            keys,
        }
    }

    // 内部：LSM 合并
    fn merge(&mut self) -> Result<(), MemoryError> {
        let _records = self.token_store.drain();
        // 已在 ArchiveIndex 中实时更新，无需额外合并
        // 实际 LSM 合并：将 ArchiveIndex 的 PostingList 写入后端
        if let Some(ref backend) = self.backend {
            for feat in 0..16u8 {
                let drawer = self.archive_index.drawer(feat);
                for (key, pl) in &drawer.tree {
                    backend
                        .write_posting(feat, *key, pl)
                        .map_err(|e| MemoryError::Store(e.to_string()))?;
                }
            }
        }
        Ok(())
    }

    // 内部：WAL 恢复
    fn recover_wal(&mut self) -> Result<(), MemoryError> {
        if let Some(ref backend) = self.backend {
            let records = backend
                .read_wal()
                .map_err(|e| MemoryError::Store(e.to_string()))?;
            for record in records {
                if record.record_type == WalType::Insert {
                    self.archive_index
                        .merge_from_hsh_seq(record.doc_id, &record.hsh_seq);
                    self.next_doc_id = self.next_doc_id.max(record.doc_id + 1);
                }
            }
        }
        Ok(())
    }

    // 内部：去重 + 排序
    fn dedup_and_sort(results: Vec<QueryResult>) -> Vec<QueryResult> {
        use std::collections::HashMap;
        let mut map: HashMap<u64, QueryResult> = HashMap::new();
        for r in results {
            map.entry(r.doc_id)
                .and_modify(|existing| {
                    if r.score > existing.score {
                        *existing = r.clone();
                    }
                })
                .or_insert(r);
        }
        let mut sorted: Vec<_> = map.into_values().collect();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_insert_and_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut mem = Memory::open(Config::new(dir.path().join("test.db"))).unwrap();

        let id = mem.insert("用户明天下午3点开会，准备PPT。").unwrap();
        assert_eq!(id, 1);

        let results = mem.query("会议准备", QueryOpts::new().top_k(5)).unwrap();
        assert!(!results.is_empty());
        println!("查询结果: {:?}", results);
    }

    #[test]
    fn test_memory_batch() {
        let dir = tempfile::tempdir().unwrap();
        let mut mem = Memory::open(Config::new(dir.path().join("test.db"))).unwrap();

        let texts: Vec<String> = vec![
            "这是一个测试文档".to_string(),
            "另一个文档内容".to_string(),
            "测试数据很多".to_string(),
        ];
        let ids = mem.insert_batch(&texts).unwrap();
        assert_eq!(ids.len(), 3);

        let results = mem.query("测试", QueryOpts::new().top_k(10)).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_memory_scan_bucket() {
        let dir = tempfile::tempdir().unwrap();
        let mut mem = Memory::open(Config::new(dir.path().join("test.db"))).unwrap();
        mem.insert("这是一个测试文档").unwrap();
        mem.insert("测试数据").unwrap();

        // 扫描名词抽屉（feat=0）的某个 sim
        let ids = mem.scan_bucket(0x0, 0x00); // 0x00 可能不命中，但 API 可用
        // 至少能运行不 panic
        println!("scan_bucket 结果: {:?}", ids);
    }
}
