//! Cabinet HSH 编码层
//!
//! 20-bit 层次语义哈希（Hierarchical Semantic Hashing）的纯计算实现。
//! 零 IO、零线程、零网络，仅操作内存中的字符串和整数。

pub mod cluster;
pub mod encoder;
pub mod error;
pub mod hsh_code;
pub mod perfect_hash;
pub mod pos_map;

pub mod hsh32;

pub use cluster::{ClusterCenter, ClusterGroup, ClusterCenters, mock_embed};
pub use encoder::{Encoder, EncoderConfig};
pub use error::EncodeError;
pub use hsh_code::HSHCode;
pub use hsh32::{HSHCode32, Encoder32, PcaProjection, sign_quantize, is_antonym, is_antonym_default, direction_score, position_score, frequency_score, compute_score, compute_score_default, query_hsh32, query_hsh32_default, CandidateDoc, QueryResult, hamming_distance, explained_variance_ratio};
pub use perfect_hash::{SeedTable, bkdr_hash, compute_abs, search_seed};
pub use pos_map::{feat_name, pos_to_feat, FeatureCode};
