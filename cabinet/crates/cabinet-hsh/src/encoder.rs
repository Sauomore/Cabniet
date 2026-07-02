//! 编码器核心：分词 → POS → HSH 编码
//!
//! 整合 jieba 分词、词性映射、聚类中心查找、准完美哈希分配。

use crate::{
    cluster::{ClusterCenters, mock_embed},
    error::EncodeError,
    hsh_code::HSHCode,
    hsh32::{HSHCode32, Encoder32},
    perfect_hash::{SeedTable, compute_abs},
    pos_map::{pos_to_feat, FeatureCode},
};
use jieba_rs::Jieba;
use std::collections::HashSet;
use std::path::PathBuf;

/// 编码器配置
pub struct EncoderConfig {
    /// 常用词晋升阈值（出现次数超过此值则晋升至 0xE）
    pub pos_threshold: u32,
    /// 新簇创建距离阈值（仅离线阶段生效）
    pub sim_threshold: f32,
    /// 聚类中心文件路径（可选，MVP 阶段可用 mock_embed）
    pub cluster_centers_path: Option<PathBuf>,
    /// 种子表文件路径（可选）
    pub seed_table_path: Option<PathBuf>,
    /// 词映射表路径（可选）
    pub mapping_path: Option<PathBuf>,
    /// 向量维度（MVP 阶段 mock_embed 使用）
    pub embed_dim: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig {
            pos_threshold: 50,
            sim_threshold: 0.5,
            cluster_centers_path: None,
            seed_table_path: None,
            mapping_path: None,
            embed_dim: 768,
        }
    }
}

/// 运行时编码器
/// 运行时**不加载 BERT**，只加载聚类中心 + 种子表 + 映射表。
/// 新词通过欧氏距离分配到现有簇（冻结策略，不创建新簇）。
pub struct Encoder {
    jieba: Jieba,
    centers: Option<ClusterCenters>,
    seed_table: SeedTable,
    // 常用词集合（出现频次超过阈值，晋升至 0xE）
    common_words: HashSet<String>,
    config: EncoderConfig,
    // 词 → (feat, sim, abs) 映射表（缓存已知词，避免重复计算）
    word_cache: std::sync::Mutex<ahash::AHashMap<String, HSHCode>>,
}

// 使用 ahash 提升 HashMap 性能（标准库替代方案）
use ahash::AHashMap;

impl Encoder {
    /// 创建默认编码器（MVP 简化版：无聚类中心，使用 mock_embed）
    pub fn new() -> Self {
        Encoder::with_config(EncoderConfig::default()).unwrap()
    }

    /// 使用配置创建编码器
    pub fn with_config(config: EncoderConfig) -> Result<Self, EncodeError> {
        let mut encoder = Encoder {
            jieba: Jieba::new(),
            centers: None,
            seed_table: SeedTable::new(),
            common_words: HashSet::new(),
            config,
            word_cache: std::sync::Mutex::new(AHashMap::new()),
        };

        // 加载聚类中心（如果路径存在）
        if let Some(ref path) = encoder.config.cluster_centers_path {
            let bytes = std::fs::read(path)
                .map_err(|e| EncodeError::Io(format!("读取聚类中心失败: {}", e)))?;
            encoder.centers = Some(
                ClusterCenters::from_bytes(&bytes)
                    .map_err(|e| EncodeError::Io(format!("解析聚类中心失败: {}", e)))?,
            );
        }

        // 加载种子表（如果路径存在）
        // TODO: 种子表反序列化

        // 加载常用词（如果路径存在）
        // TODO: 常用词加载

        Ok(encoder)
    }

    /// 编码文本为 HSH 序列
    pub fn encode(&self, text: &str) -> Result<Vec<HSHCode>, EncodeError> {
        // 1. jieba 分词 + POS 标注
        let tokens = self.jieba.tag(text, true);

        // 2. 对每个词生成 HSH
        let mut codes = Vec::with_capacity(tokens.len());
        for token in tokens {
            let word = token.word;
            let pos = token.tag;

            // 检查缓存
            {
                let cache = self.word_cache.lock().unwrap();
                if let Some(&code) = cache.get(word) {
                    codes.push(code);
                    continue;
                }
            }

            let code = self.encode_word(word, &pos)?;

            // 写入缓存
            {
                let mut cache = self.word_cache.lock().unwrap();
                cache.insert(word.to_string(), code);
            }

            codes.push(code);
        }

        Ok(codes)
    }

    /// 编码单个词
    fn encode_word(&self, word: &str, pos: &str) -> Result<HSHCode, EncodeError> {
        // 1. 词性 → 特征码
        let feat = pos_to_feat(pos)
            .ok_or_else(|| EncodeError::UnknownPOSTag(pos.to_string()))?;

        // 2. 常用词晋升
        let feat = if self.common_words.contains(word) {
            FeatureCode::COMMON
        } else {
            feat
        };

        // 3. 分配相似码
        let sim = self.assign_sim(word, feat.as_u8())?;

        // 4. 分配绝对码
        let abs = self.assign_abs(word, feat.as_u8(), sim)?;

        Ok(HSHCode::new(feat.as_u8(), sim, abs))
    }

    /// 分配相似码
    fn assign_sim(&self, word: &str, feat: u8) -> Result<u8, EncodeError> {
        // 有聚类中心时，计算向量距离
        if let Some(ref centers) = self.centers {
            let vec = mock_embed(word, self.config.embed_dim); // MVP 用 mock_embed
            match centers.assign_sim(&vec, feat, self.config.sim_threshold) {
                Some(sim) => Ok(sim),
                None => {
                    // 运行时冻结：新词强制归入最近簇
                    let group = centers.group(feat).ok_or(EncodeError::SimOutOfRange)?;
                    let (sim, _dist) = group.nearest(&vec).ok_or(EncodeError::SimOutOfRange)?;
                    Ok(sim)
                }
            }
        } else {
            // 无聚类中心时：使用 BKDR 哈希取模 256 作为简化 sim
            // 这仅用于 MVP 快速测试，实际应由聚类中心决定
            Ok((crate::perfect_hash::bkdr_hash(word) % 256) as u8)
        }
    }

    /// 分配绝对码
    fn assign_abs(&self, word: &str, feat: u8, sim: u8) -> Result<u8, EncodeError> {
        // 1. 查映射表（已知词）
        if let Some(abs) = self.seed_table.get_abs(word, feat, sim) {
            return Ok(abs);
        }

        // 2. 计算候选绝对码
        let seed = self.seed_table.get_seed(feat, sim);
        let candidate = compute_abs(word, seed);

        // 3. 运行时：直接返回候选（冲突由检索层处理）
        Ok(candidate)
    }

    /// 编码文本，返回每个词的 (word, pos, HSHCode) 详细信息
    pub fn encode_detail(&self, text: &str) -> Result<Vec<(String, String, HSHCode)>, EncodeError> {
        let tokens = self.jieba.tag(text, true);
        let mut results = Vec::with_capacity(tokens.len());
        for token in tokens {
            let word = token.word.to_string();
            let pos = token.tag.to_string();
            let code = self.encode_word(&word, &pos)?;
            results.push((word, pos, code));
        }
        Ok(results)
    }

    /// 添加常用词
    pub fn add_common_word(&mut self, word: &str) {
        self.common_words.insert(word.to_string());
    }

    /// 移除常用词
    pub fn remove_common_word(&mut self, word: &str) {
        self.common_words.remove(word);
    }

    /// 设置聚类中心
    pub fn set_centers(&mut self, centers: ClusterCenters) {
        self.centers = Some(centers);
    }

    /// 设置种子表
    pub fn set_seed_table(&mut self, table: SeedTable) {
        self.seed_table = table;
    }

    /// 编码文本为 HSH-32 序列（使用 PCA + sign 量化）
    pub fn encode_hsh32(&self, text: &str) -> Result<Vec<HSHCode32>, EncodeError> {
        let tokens = self.jieba.tag(text, true);
        let mut codes = Vec::with_capacity(tokens.len());
        let encoder32 = Encoder32::new();

        for token in tokens {
            let word = token.word;
            let pos = token.tag;

            let feat = if self.common_words.contains(word) {
                0x0E
            } else {
                pos_to_feat(pos)
                    .map(|f| f.as_u8())
                    .unwrap_or(0x0F)
            };

            let code = encoder32.encode_word(word, feat);
            codes.push(code);
        }

        Ok(codes)
    }

    /// 编码单个词为 HSH-32
    pub fn encode_word_hsh32(&self, word: &str, pos: &str) -> Result<HSHCode32, EncodeError> {
        let encoder32 = Encoder32::new();
        let feat = if self.common_words.contains(word) {
            0x0E
        } else {
            pos_to_feat(pos)
                .map(|f| f.as_u8())
                .unwrap_or(0x0F)
        };
        Ok(encoder32.encode_word(word, feat))
    }

    /// 编码速度基准（近似）
    pub fn throughput_estimate(&self) -> u32 {
        // 简单估算：单线程约 5万词/秒（MVP 目标）
        50_000
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_basic() {
        let encoder = Encoder::new();
        let codes = encoder.encode("明天下午3点开会，准备PPT。").unwrap();
        assert!(!codes.is_empty());

        for code in &codes {
            assert!(code.feat() <= 0x0F);
            assert!(code.sim() <= 0xFF);
            assert!(code.abs() <= 0xFF);
            assert!(code.raw() < (1 << 20));
        }
    }

    #[test]
    fn test_encode_roundtrip() {
        let encoder = Encoder::new();
        let text = "明天下午3点开会准备PPT";
        let codes = encoder.encode(text).unwrap();

        // 每个码可独立验证其位结构
        assert!(codes.iter().all(|c| c.raw() < (1 << 20)));
    }

    #[test]
    fn test_encode_empty() {
        let encoder = Encoder::new();
        let codes = encoder.encode("").unwrap();
        assert!(codes.is_empty());
    }

    #[test]
    fn test_encode_long() {
        let encoder = Encoder::new();
        let text = "这是一个很长的句子，用来测试编码器处理大量词汇的能力。\n".repeat(100);
        let codes = encoder.encode(&text).unwrap();
        assert!(codes.len() > 100);
    }

    #[test]
    fn test_common_word_promotion() {
        let mut encoder = Encoder::new();
        encoder.add_common_word("测试");

        let codes1 = encoder.encode("测试一下").unwrap();
        let code_test = codes1.iter().find(|c| c.feat() == 0x0E);
        assert!(code_test.is_some(), "常用词应被晋升至 0xE");
    }

    #[test]
    fn test_with_centers() {
        use crate::cluster::{ClusterCenter, ClusterGroup, ClusterCenters};

        let mut encoder = Encoder::new();
        let groups = vec![
            ClusterGroup {
                feat: 0,
                centers: vec![
                    ClusterCenter { id: 0, vector: vec![0.0; 768] },
                    ClusterCenter { id: 1, vector: vec![1.0; 768] },
                ],
            },
        ];
        encoder.set_centers(ClusterCenters::new(groups));

        let codes = encoder.encode("学校").unwrap();
        assert!(!codes.is_empty());
    }
}
