//! Working Memory：Layer 3，瞬态上下文缓存
//!
//! 推理期间暂存的热点记忆，LRU 淘汰策略。

use cabinet_hsh::HSHCode;
use std::collections::HashMap;

/// 记忆片段（工作记忆单元）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySnippet {
    pub doc_id: u64,
    pub text: String, // 可选的原始文本片段
}

/// 工作记忆：LRU 缓存（简化版：HashMap + 访问计数）
pub struct WorkingMemory {
    pub max_tokens: usize, // 最大 HSH 数量（不是 snippet 数量）
    cache: HashMap<HSHCode, MemorySnippet>,
    access_count: HashMap<HSHCode, u64>, // 访问计数，用于淘汰
}

impl WorkingMemory {
    pub fn new(max_tokens: usize) -> Self {
        WorkingMemory {
            max_tokens,
            cache: HashMap::new(),
            access_count: HashMap::new(),
        }
    }

    /// 加载 snippet 到工作记忆
    pub fn load(&mut self, hsh: HSHCode, snippet: MemorySnippet) {
        if self.cache.len() >= self.max_tokens && !self.cache.contains_key(&hsh) {
            // 淘汰访问计数最低的项
            if let Some((&evict_key, _)) = self
                .access_count
                .iter()
                .min_by_key(|(_, &count)| count)
            {
                self.cache.remove(&evict_key);
                self.access_count.remove(&evict_key);
            }
        }
        self.cache.insert(hsh, snippet);
        self.access_count.insert(hsh, 1);
    }

    /// 查询工作记忆
    pub fn query(&mut self, hsh: HSHCode) -> Option<&MemorySnippet> {
        let result = self.cache.get(&hsh);
        if result.is_some() {
            *self.access_count.get_mut(&hsh).unwrap() += 1;
        }
        result
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_count.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cabinet_hsh::HSHCode;

    #[test]
    fn test_working_memory() {
        let mut wm = WorkingMemory::new(2);
        let h1 = HSHCode::new(0, 1, 2);
        let h2 = HSHCode::new(0, 1, 3);
        let h3 = HSHCode::new(0, 1, 4);

        wm.load(h1, MemorySnippet { doc_id: 1, text: "a".to_string() });
        wm.load(h2, MemorySnippet { doc_id: 2, text: "b".to_string() });
        assert_eq!(wm.len(), 2);

        // 访问 h1 提升热度
        wm.query(h1);
        wm.query(h1);

        // 加载 h3，应淘汰 h2（访问计数最低）
        wm.load(h3, MemorySnippet { doc_id: 3, text: "c".to_string() });
        assert_eq!(wm.len(), 2);
        assert!(wm.query(h2).is_none());
        assert!(wm.query(h1).is_some());
    }
}
