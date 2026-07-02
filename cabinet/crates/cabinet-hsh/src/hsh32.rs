//! HSH-32 编码层：Hierarchical Semantic Hashing with 32-bit Continuous Direction Encoding
//!
//! 位结构：
//! - feat: 4-bit [31:28]  词性类别
//! - sim:  20-bit [27:8]  PCA 降维 + sign 量化后的语义方向码
//! - abs:  8-bit [7:0]    簇内完美哈希
//!
//! 总空间：16 × 2^20 × 256 ≈ 4.3B 个唯一编码

use crate::error::EncodeError;
use std::collections::HashMap;

/// HSH-32 编码，内部用 u32 存储
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HSHCode32(u32);

impl HSHCode32 {
    pub const BITS: u32 = 32;
    pub const FEAT_BITS: u32 = 4;
    pub const SIM_BITS: u32 = 20;
    pub const ABS_BITS: u32 = 8;
    pub const SIM_MAX: u32 = 1 << 20; // 1,048,576
    pub const ABS_MAX: u32 = 1 << 8;  // 256

    /// 创建新的 HSH-32 编码
    /// # Panics
    /// 当 feat > 0xF, sim > 0xFFFFF, 或 abs > 0xFF 时 panic
    pub fn new(feat: u8, sim: u32, abs: u8) -> Self {
        assert!(feat <= 0x0F, "feat 超出 4-bit 范围");
        assert!(sim <= 0xFFFFF, "sim 超出 20-bit 范围");
        let value = ((feat as u32) << 28) | ((sim & 0xFFFFF) << 8) | (abs as u32);
        HSHCode32(value)
    }

    /// 从原始 u32 值创建（完整 32-bit）
    pub fn from_raw(value: u32) -> Self {
        HSHCode32(value)
    }

    /// 获取原始 u32 值
    pub fn raw(&self) -> u32 {
        self.0
    }

    /// 获取特征码（高 4-bit）
    pub fn feat(&self) -> u8 {
        ((self.0 >> 28) & 0x0F) as u8
    }

    /// 获取相似码（中 20-bit）
    pub fn sim(&self) -> u32 {
        (self.0 >> 8) & 0xFFFFF
    }

    /// 获取绝对码（低 8-bit）
    pub fn abs(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    /// 打包为 4 bytes（大端序）
    pub fn to_bytes(&self) -> [u8; 4] {
        self.0.to_be_bytes()
    }

    /// 从 4 bytes 解码（大端序）
    pub fn from_bytes(b: [u8; 4]) -> Self {
        HSHCode32(u32::from_be_bytes(b))
    }

    /// 编码为 u32 整数（便于 Python 层传递）
    pub fn to_u32(&self) -> u32 {
        self.0
    }

    /// 从 u32 整数解码
    pub fn from_u32(value: u32) -> Self {
        Self::from_raw(value)
    }

    /// 获取 24-bit bucket ID（feat + sim，用于索引）
    pub fn bucket_id(&self) -> u32 {
        (self.0 >> 8) & 0xFFFFFF
    }

    /// 编码 HSH-32 序列为字节流（含长度前缀）
    pub fn encode_seq(codes: &[HSHCode32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + codes.len() * 4);
        buf.extend_from_slice(&(codes.len() as u32).to_be_bytes());
        for code in codes {
            buf.extend_from_slice(&code.to_bytes());
        }
        buf
    }

    /// 从字节流解码 HSH-32 序列（含长度前缀）
    pub fn decode_seq(buf: &[u8]) -> Option<Vec<HSHCode32>> {
        if buf.len() < 4 {
            return None;
        }
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        if buf.len() < 4 + len * 4 {
            return None;
        }
        let mut codes = Vec::with_capacity(len);
        for i in 0..len {
            let offset = 4 + i * 4;
            codes.push(HSHCode32::from_bytes([
                buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3],
            ]));
        }
        Some(codes)
    }
}

/// PCA 投影矩阵（简化版：MVP 阶段使用预计算或 mock 矩阵）
///
/// 论文方法：将 768-dim BERT 向量通过 PCA 降维到 20-dim，
/// 然后对每维用 sign 函数量化为 1-bit，组成 20-bit sim。
pub struct PcaProjection {
    /// 投影矩阵：20 × dim 的 f32 矩阵（行优先）
    /// 每行是一个主成分向量，已归一化
    pub components: Vec<Vec<f32>>,
    pub dim: usize,
}

impl PcaProjection {
    /// 创建空的 PCA 投影（用于测试）
    pub fn new(dim: usize) -> Self {
        PcaProjection {
            components: Vec::new(),
            dim,
        }
    }

    /// 从 mock 数据创建（MVP 阶段：用随机正交基模拟 PCA）
    pub fn mock(dim: usize) -> Self {
        let mut components = Vec::with_capacity(20);
        for i in 0..20 {
            let mut vec = vec![0.0f32; dim];
            // 简化的 mock：每个主成分只有一个维度有值
            // 实际应由离线训练得到
            if i < dim {
                vec[i] = 1.0;
            }
            components.push(vec);
        }
        PcaProjection { components, dim }
    }

    /// 投影向量到 20 维子空间
    pub fn project(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(vector.len(), self.dim, "维度不匹配");
        let mut result = Vec::with_capacity(20);
        for comp in &self.components {
            let dot: f32 = comp.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
            result.push(dot);
        }
        result
    }

    /// 将向量投影并归一化到单位球面
    pub fn project_normalized(&self, vector: &[f32]) -> Vec<f32> {
        let mut proj = self.project(vector);
        let norm: f32 = proj.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut proj {
                *x /= norm;
            }
        }
        proj
    }
}

/// Sign 量化：将 20 维浮点向量量化为 20-bit 整数
///
/// 公式：bj = sign(uj) = 1 if uj >= 0 else 0
/// sim = Σ bj · 2^(20-j)  (j=1..20)
pub fn sign_quantize(projection: &[f32]) -> u32 {
    assert_eq!(projection.len(), 20, "投影向量必须是 20 维");
    let mut sim = 0u32;
    for (j, &uj) in projection.iter().enumerate() {
        let bj = if uj >= 0.0 { 1 } else { 0 };
        sim |= bj << (19 - j);
    }
    sim
}

/// 计算两个 20-bit sim 的 Hamming 距离
pub fn hamming_distance(sim_a: u32, sim_b: u32) -> u32 {
    (sim_a ^ sim_b).count_ones()
}

/// 反义词检测
///
/// 论文定理：当两个向量方向相反时（夹角 ≈ 180°），
/// 它们的 20-bit sim 码在二进制上呈现按位取反关系，
/// 即 sim_b ≈ 0xFFFFF - sim_a。
///
/// 判定条件：|diff - 2^19| <= tolerance
/// 默认 tolerance = 4096（约 0.4% 的误差容忍）
pub fn is_antonym(sim_a: u32, sim_b: u32, tolerance: u32) -> bool {
    // 反义词判定：sim_b 应该接近 0xFFFFF - sim_a
    // 即 sim_a + sim_b 应该接近 0xFFFFF
    let sum = sim_a.wrapping_add(sim_b);
    let expected = 0xFFFFF; // 1,048,575
    let diff = if sum >= expected {
        sum - expected
    } else {
        expected - sum
    };
    diff <= tolerance
}

/// 使用默认 tolerance（4096）的反义词检测
pub fn is_antonym_default(sim_a: u32, sim_b: u32) -> bool {
    is_antonym(sim_a, sim_b, 4096)
}

/// 查询候选文档结构
#[derive(Debug, Clone)]
pub struct CandidateDoc {
    pub doc_id: u64,
    pub sim: u32,
    pub abs: u8,
    pub feat: u8,
    pub position: u32,
    pub frequency: f32,
}

/// 查询结果
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub doc: CandidateDoc,
    pub score: f32,
    pub direction_score: f32,
    pub position_score: f32,
    pub frequency_score: f32,
    pub is_antonym: bool,
}

/// 方向评分：1 - |sim_q - sim_d| / 2^20
///
/// 当 sim 越接近时，方向评分越高（趋近于 1）
/// 当 sim 相差越大时，方向评分越低（趋近于 0）
pub fn direction_score(sim_q: u32, sim_d: u32) -> f32 {
    let max_sim = (1u32 << 20) as f32; // 1,048,576
    let diff = if sim_q >= sim_d {
        sim_q - sim_d
    } else {
        sim_d - sim_q
    };
    1.0 - (diff as f32) / max_sim
}

/// 位置评分：基于 abs 码的接近程度
///
/// 在相同 bucket 内，abs 越接近表示位置越接近
pub fn position_score(abs_q: u8, abs_d: u8) -> f32 {
    let diff = if abs_q >= abs_d {
        abs_q - abs_d
    } else {
        abs_d - abs_q
    };
    1.0 - (diff as f32) / 255.0
}

/// 频率评分：基于文档频率（TF-IDF 简化）
pub fn frequency_score(freq: f32) -> f32 {
    // 简单对数频率评分
    (1.0 + freq.ln_1p()).min(2.0) / 2.0
}

/// 综合评分
///
/// Score(q, d) = α·DirectionScore + β·PositionScore + γ·FrequencyScore
/// 论文默认值：α = 0.7, β = 0.2, γ = 0.1
/// 如果检测到反义词，评分乘以 -0.5
pub fn compute_score(
    sim_q: u32,
    sim_d: u32,
    abs_q: u8,
    abs_d: u8,
    freq_d: f32,
    alpha: f32,
    beta: f32,
    gamma: f32,
    tolerance: u32,
) -> (f32, bool) {
    let antonym = is_antonym(sim_q, sim_d, tolerance);
    let mut score = alpha * direction_score(sim_q, sim_d)
        + beta * position_score(abs_q, abs_d)
        + gamma * frequency_score(freq_d);
    if antonym {
        score *= -0.5;
    }
    (score, antonym)
}

/// 使用默认参数的评分
pub fn compute_score_default(
    sim_q: u32,
    sim_d: u32,
    abs_q: u8,
    abs_d: u8,
    freq_d: f32,
) -> (f32, bool) {
    compute_score(sim_q, sim_d, abs_q, abs_d, freq_d, 0.7, 0.2, 0.1, 4096)
}

/// 查询算法（论文算法）
///
/// 在 [sim_q - delta, sim_q + delta] 范围内扫描所有候选 bucket，
/// 收集文档并排序返回 top_k。
pub fn query_hsh32(
    query_code: HSHCode32,
    buckets: &HashMap<u32, Vec<CandidateDoc>>,
    top_k: usize,
    delta: u32,
    alpha: f32,
    beta: f32,
    gamma: f32,
    tolerance: u32,
) -> Vec<QueryResult> {
    let mut candidates = Vec::new();
    let sim_q = query_code.sim();
    let feat_q = query_code.feat() as u32;
    let abs_q = query_code.abs();

    let sim_min = sim_q.saturating_sub(delta);
    let sim_max = (sim_q + delta).min(0xFFFFF);

    for sim in sim_min..=sim_max {
        let bucket_id = (feat_q << 20) | sim;
        if let Some(docs) = buckets.get(&bucket_id) {
            for doc in docs {
                let (score, is_ant) = compute_score(
                    sim_q, doc.sim, abs_q, doc.abs, doc.frequency,
                    alpha, beta, gamma, tolerance,
                );
                candidates.push(QueryResult {
                    doc: doc.clone(),
                    score,
                    direction_score: direction_score(sim_q, doc.sim),
                    position_score: position_score(abs_q, doc.abs),
                    frequency_score: frequency_score(doc.frequency),
                    is_antonym: is_ant,
                });
            }
        }
    }

    // 按评分降序排序，返回 top_k
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k);
    candidates
}

/// 使用默认参数的查询
pub fn query_hsh32_default(
    query_code: HSHCode32,
    buckets: &HashMap<u32, Vec<CandidateDoc>>,
    top_k: usize,
) -> Vec<QueryResult> {
    query_hsh32(query_code, buckets, top_k, 1024, 0.7, 0.2, 0.1, 4096)
}

/// HSH-32 编码器（运行时轻量版）
pub struct Encoder32 {
    pca: PcaProjection,
    seed_table: crate::perfect_hash::SeedTable,
    common_words: std::collections::HashSet<String>,
}

impl Encoder32 {
    /// 创建默认编码器（使用 mock PCA）
    pub fn new() -> Self {
        Encoder32 {
            pca: PcaProjection::mock(768),
            seed_table: crate::perfect_hash::SeedTable::new(),
            common_words: std::collections::HashSet::new(),
        }
    }

    /// 使用自定义 PCA 创建编码器
    pub fn with_pca(pca: PcaProjection) -> Self {
        Encoder32 {
            pca,
            seed_table: crate::perfect_hash::SeedTable::new(),
            common_words: std::collections::HashSet::new(),
        }
    }

    /// 编码单个词为 HSH-32
    ///
    /// 流程：
    /// 1. 获取词向量（mock_embed）
    /// 2. PCA 降维到 20 维
    /// 3. sign 量化为 20-bit sim
    /// 4. 计算 8-bit abs（完美哈希）
    pub fn encode_word(&self, word: &str, feat: u8) -> HSHCode32 {
        // 1. 获取 mock 向量
        let vec = crate::cluster::mock_embed(word, self.pca.dim);

        // 2. PCA 降维并归一化
        let proj = self.pca.project_normalized(&vec);

        // 3. sign 量化 → 20-bit sim
        let sim = sign_quantize(&proj);

        // 4. 计算 abs（完美哈希）
        let seed = self.seed_table.get_seed(feat, (sim % 256) as u8);
        let abs = crate::perfect_hash::compute_abs(word, seed);

        HSHCode32::new(feat, sim, abs)
    }

    /// 编码文本（分词后逐词编码）
    pub fn encode_text(&self, text: &str) -> Result<Vec<HSHCode32>, EncodeError> {
        use jieba_rs::Jieba;
        use crate::pos_map::pos_to_feat;

        let jieba = Jieba::new();
        let tokens = jieba.tag(text, true);
        let mut codes = Vec::with_capacity(tokens.len());

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

            let code = self.encode_word(word, feat);
            codes.push(code);
        }

        Ok(codes)
    }

    /// 添加常用词
    pub fn add_common_word(&mut self, word: &str) {
        self.common_words.insert(word.to_string());
    }
}

impl Default for Encoder32 {
    fn default() -> Self {
        Self::new()
    }
}

/// 估算 PCA 保留方差比例
///
/// 论文：BERT-wwm-ext 768-dim → 20-dim 保留约 70%~85% 方差
pub fn explained_variance_ratio(eigenvalues: &[f32], top_k: usize) -> f32 {
    let total: f32 = eigenvalues.iter().sum();
    let top_sum: f32 = eigenvalues.iter().take(top_k).sum();
    if total > 0.0 {
        top_sum / total
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsh32_roundtrip() {
        let code = HSHCode32::new(0x0A, 0x42_123, 0xF0);
        assert_eq!(code.feat(), 0x0A);
        assert_eq!(code.sim(), 0x42_123);
        assert_eq!(code.abs(), 0xF0);
        assert_eq!(code.raw(), (0x0Au32 << 28) | (0x42_123u32 << 8) | 0xF0u32);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let code = HSHCode32::new(0x0F, 0xF_FFFF, 0xFF);
        let bytes = code.to_bytes();
        let decoded = HSHCode32::from_bytes(bytes);
        assert_eq!(code, decoded);
    }

    #[test]
    fn test_sign_quantize() {
        // 全正向量 → 全 1
        let proj_all_pos = vec![1.0; 20];
        assert_eq!(sign_quantize(&proj_all_pos), 0xF_FFFF);

        // 全负向量 → 全 0
        let proj_all_neg = vec![-1.0; 20];
        assert_eq!(sign_quantize(&proj_all_neg), 0);

        // 交替正负
        let mut proj_alt = Vec::with_capacity(20);
        for i in 0..20 {
            proj_alt.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        let sim = sign_quantize(&proj_alt);
        // 偶数位置（0,2,4...）为 1，奇数位置为 0
        // b0=1, b1=0, b2=1, b3=0...
        // sim = 1<<19 | 1<<17 | 1<<15 | ... = 0xAA_AAA
        let expected = 0xAA_AAA;
        assert_eq!(sim, expected);
    }

    #[test]
    fn test_antonym_detection() {
        // 精确反义：sim_a = 0, sim_b = 0xFFFFF
        assert!(is_antonym(0, 0xF_FFFF, 4096));

        // 接近反义：sim_a = 100_000, sim_b ≈ 0xFFFFF - 100_000 = 948_575
        assert!(is_antonym(100_000, 948_575, 4096));

        // 中点反义：sim_a = 0x80000, sim_b = 0x7FFFF
        assert!(is_antonym(0x80_000, 0x7_FFFF, 4096));

        // 不是反义
        assert!(!is_antonym(0, 100_000, 4096));
        assert!(!is_antonym(500_000, 600_000, 4096));
    }

    #[test]
    fn test_direction_score() {
        // 相同 sim → 1.0
        assert!((direction_score(100_000, 100_000) - 1.0).abs() < 0.001);

        // 相差一半 → 0.5
        assert!((direction_score(0, 524_288) - 0.5).abs() < 0.001);

        // 最大差异 → 0.0
        assert!((direction_score(0, 0xF_FFFF) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_antonym_score() {
        let (score, is_ant) = compute_score_default(0, 0xF_FFFF, 0, 0, 1.0);
        assert!(is_ant);
        assert!(score < 0.0, "反义词评分应为负值: {}", score);
    }

    #[test]
    fn test_query_algorithm() {
        let mut buckets = HashMap::new();

        // 创建一些测试文档
        for i in 0..10u32 {
            let sim = i * 10_000;
            let bucket_id = (0u32 << 20) | sim;
            let docs = vec![
                CandidateDoc {
                    doc_id: i as u64,
                    sim,
                    abs: (i * 10) as u8,
                    feat: 0,
                    position: i,
                    frequency: 1.0,
                },
            ];
            buckets.insert(bucket_id, docs);
        }

        let query = HSHCode32::new(0, 50_000, 0);
        let results = query_hsh32(query, &buckets, 5, 20_000, 0.7, 0.2, 0.1, 4096);

        assert!(!results.is_empty());
        // 第一个结果应该是 sim=50_000 的文档（精确匹配）
        assert_eq!(results[0].doc.sim, 50_000);
    }

    #[test]
    fn test_encoder32() {
        let encoder = Encoder32::new();
        let code = encoder.encode_word("测试", 0x01);
        assert!(code.feat() <= 0x0F);
        assert!(code.sim() <= 0xF_FFFF);
    }

    #[test]
    fn test_seq_codec() {
        let codes = vec![
            HSHCode32::new(0x0, 0x1, 0x2),
            HSHCode32::new(0xF, 0xF_FFFF, 0xFF),
            HSHCode32::new(0x5, 0x42_123, 0x88),
        ];
        let encoded = HSHCode32::encode_seq(&codes);
        let decoded = HSHCode32::decode_seq(&encoded).unwrap();
        assert_eq!(codes, decoded);
    }

    #[test]
    fn test_hamming_distance() {
        // 0x00000 vs 0xFFFFF → 20 bits 全不同
        assert_eq!(hamming_distance(0, 0xF_FFFF), 20);

        // 相同 → 0
        assert_eq!(hamming_distance(0x42_123, 0x42_123), 0);
    }
}
