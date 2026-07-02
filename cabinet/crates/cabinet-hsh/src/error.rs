use std::fmt;

/// HSH 编码层错误类型
#[derive(Debug, Clone, PartialEq)]
pub enum EncodeError {
    /// 未知词性标签
    UnknownPOSTag(String),
    /// 聚类中心未加载
    CentersNotLoaded,
    /// 种子表未加载
    SeedTableNotLoaded,
    /// 词映射表未加载
    MappingNotLoaded,
    /// 无法分配相似码（超出范围）
    SimOutOfRange,
    /// 无法分配相似码（超出 20-bit 范围）
    Sim32OutOfRange,
    /// 无法分配绝对码（溢出严重）
    AbsOverflow,
    /// IO 错误
    Io(String),
    /// 配置错误
    Config(String),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::UnknownPOSTag(tag) => write!(f, "未知词性标签: {}", tag),
            EncodeError::CentersNotLoaded => write!(f, "聚类中心未加载"),
            EncodeError::SeedTableNotLoaded => write!(f, "种子表未加载"),
            EncodeError::MappingNotLoaded => write!(f, "词映射表未加载"),
            EncodeError::SimOutOfRange => write!(f, "相似码超出 0-255 范围"),
            EncodeError::Sim32OutOfRange => write!(f, "相似码超出 20-bit 范围（0-1,048,575）"),
            EncodeError::AbsOverflow => write!(f, "绝对码分配溢出（该簇已满）"),
            EncodeError::Io(msg) => write!(f, "IO 错误: {}", msg),
            EncodeError::Config(msg) => write!(f, "配置错误: {}", msg),
        }
    }
}

impl std::error::Error for EncodeError {}

impl From<std::io::Error> for EncodeError {
    fn from(e: std::io::Error) -> Self {
        EncodeError::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = EncodeError::UnknownPOSTag("xyz".to_string());
        assert_eq!(e.to_string(), "未知词性标签: xyz");
    }
}
