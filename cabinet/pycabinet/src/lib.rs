use pyo3::prelude::*;
use pyo3::types::PyList;
use std::path::PathBuf;

use cabinet_core::{
    Config, Memory, Precision, QueryOpts, QueryResult,
    MemoryStats, DrawerStats,
};
use cabinet_hsh::{Encoder, EncoderConfig, HSHCode, HSHCode32, Encoder32, is_antonym_default, direction_score, position_score, frequency_score, compute_score_default};

// ===== HSHCode =====
#[pyclass(name = "HSHCode")]
#[derive(Clone, Copy)]
struct PyHSHCode {
    inner: HSHCode,
}

#[pymethods]
impl PyHSHCode {
    #[new]
    fn new(feat: u8, sim: u8, abs: u8) -> Self {
        PyHSHCode { inner: HSHCode::new(feat, sim, abs) }
    }

    #[getter]
    fn feat(&self) -> u8 { self.inner.feat() }
    #[getter]
    fn sim(&self) -> u8 { self.inner.sim() }
    #[getter]
    fn abs(&self) -> u8 { self.inner.abs() }
    #[getter]
    fn raw(&self) -> u32 { self.inner.raw() }

    fn __repr__(&self) -> String {
        format!(
            "HSHCode(feat=0x{:01X}, sim=0x{:02X}, abs=0x{:02X}, raw=0x{:05X})",
            self.feat(), self.sim(), self.abs(), self.raw()
        )
    }
}

// ===== HSHCode32 =====
#[pyclass(name = "HSHCode32")]
#[derive(Clone, Copy)]
struct PyHSHCode32 {
    inner: HSHCode32,
}

#[pymethods]
impl PyHSHCode32 {
    #[new]
    fn new(feat: u8, sim: u32, abs: u8) -> Self {
        PyHSHCode32 { inner: HSHCode32::new(feat, sim, abs) }
    }

    #[getter]
    fn feat(&self) -> u8 { self.inner.feat() }
    #[getter]
    fn sim(&self) -> u32 { self.inner.sim() }
    #[getter]
    fn abs(&self) -> u8 { self.inner.abs() }
    #[getter]
    fn raw(&self) -> u32 { self.inner.raw() }
    #[getter]
    fn bucket_id(&self) -> u32 { self.inner.bucket_id() }

    fn __repr__(&self) -> String {
        format!(
            "HSHCode32(feat=0x{:01X}, sim=0x{:05X}, abs=0x{:02X}, raw=0x{:08X})",
            self.feat(), self.sim(), self.abs(), self.raw()
        )
    }
}

// ===== Encoder =====
#[pyclass(name = "Encoder")]
struct PyEncoder {
    inner: Encoder,
}

#[pymethods]
impl PyEncoder {
    #[new]
    fn new() -> PyResult<Self> {
        let encoder = Encoder::with_config(EncoderConfig::default())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyEncoder { inner: encoder })
    }

    fn encode(&self, text: String) -> PyResult<Vec<PyHSHCode>> {
        let codes = self.inner.encode(&text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(codes.into_iter().map(|c| PyHSHCode { inner: c }).collect())
    }

    fn encode_detail(&self, text: String) -> PyResult<Vec<(String, String, PyHSHCode)>> {
        let results = self.inner.encode_detail(&text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(results.into_iter().map(|(w, p, c)| (w, p, PyHSHCode { inner: c })).collect())
    }

    fn encode_hsh32(&self, text: String) -> PyResult<Vec<PyHSHCode32>> {
        let codes = self.inner.encode_hsh32(&text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(codes.into_iter().map(|c| PyHSHCode32 { inner: c }).collect())
    }

    fn encode_word_hsh32(&self, word: String, pos: String) -> PyResult<PyHSHCode32> {
        let code = self.inner.encode_word_hsh32(&word, &pos)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyHSHCode32 { inner: code })
    }
}

// ===== MemoryStats =====
#[pyclass(name = "MemoryStats")]
#[derive(Clone)]
struct PyMemoryStats {
    #[pyo3(get)]
    pub doc_count: usize,
    #[pyo3(get)]
    pub next_doc_id: u64,
    #[pyo3(get)]
    pub working_memory_capacity: usize,
    #[pyo3(get)]
    pub working_memory_used: usize,
    #[pyo3(get)]
    pub token_store_buffer_size: usize,
    #[pyo3(get)]
    pub precision: String,
}

// ===== DrawerStats =====
#[pyclass(name = "DrawerStats")]
#[derive(Clone)]
struct PyDrawerStats {
    #[pyo3(get)]
    pub feat: u8,
    #[pyo3(get)]
    pub key_count: usize,
    #[pyo3(get)]
    pub total_doc_refs: usize,
    #[pyo3(get)]
    pub keys: Vec<(u16, u32, usize)>,
}

// ===== PyMemory =====
#[pyclass(name = "Memory")]
struct PyMemory {
    inner: Memory,
}

#[pymethods]
impl PyMemory {
    #[new]
    #[pyo3(signature = (path, precision="light", pos_threshold=50, max_context=4096))]
    fn new(path: String, precision: &str, pos_threshold: u32, max_context: usize) -> PyResult<Self> {
        let p = Precision::from_str(precision)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("precision 必须是 'light'/'hybrid'/'precise'"))?;
        let config = Config::new(PathBuf::from(path))
            .precision(p)
            .pos_threshold(pos_threshold)
            .working_memory_size(max_context);
        let mem = Memory::open(config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyMemory { inner: mem })
    }

    fn insert(&mut self, text: String) -> PyResult<u64> {
        self.inner
            .insert(&text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[pyo3(signature = (texts, show_progress=false))]
    fn insert_batch(&mut self, texts: &Bound<'_, PyList>, show_progress: bool) -> PyResult<Vec<u64>> {
        let mut vec = Vec::with_capacity(texts.len());
        for (i, item) in texts.iter().enumerate() {
            let text: String = item.extract()?;
            vec.push(text);
            if show_progress && (i + 1) % 100 == 0 {
                println!("已插入 {}/{} 条", i + 1, texts.len());
            }
        }
        self.inner
            .insert_batch(&vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[pyo3(signature = (query, top_k=5))]
    fn query(&mut self, query: String, top_k: usize) -> PyResult<Vec<PyQueryResult>> {
        let opts = QueryOpts::new().top_k(top_k);
        let results = self.inner
            .query(&query, opts)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(results.into_iter().map(|r| r.into()).collect())
    }

    fn decode(&self, result: &PyQueryResult) -> PyResult<Option<String>> {
        let qr = QueryResult {
            hsh: result.hsh,
            doc_id: result.doc_id,
            position: result.position,
            match_level: result.match_level,
            score: result.score,
            text: None,
        };
        Ok(self.inner.decode(&qr))
    }

    fn snapshot(&self, dst: String) -> PyResult<()> {
        self.inner
            .snapshot(&PathBuf::from(dst))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner
            .close()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    // ===== 新增：编码可视化 =====
    fn encode_detail(&self, text: String) -> PyResult<Vec<(String, String, PyHSHCode)>> {
        let results = self.inner.encode_detail(&text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(results.into_iter().map(|(w, p, c)| (w, p, PyHSHCode { inner: c })).collect())
    }

    // ===== 新增：统计信息 =====
    fn get_stats(&self) -> PyResult<PyMemoryStats> {
        let s = self.inner.get_stats();
        Ok(PyMemoryStats {
            doc_count: s.doc_count,
            next_doc_id: s.next_doc_id,
            working_memory_capacity: s.working_memory_capacity,
            working_memory_used: s.working_memory_used,
            token_store_buffer_size: s.token_store_buffer_size,
            precision: s.precision,
        })
    }

    fn get_drawer_stats(&self, feat: u8) -> PyResult<PyDrawerStats> {
        let s = self.inner.get_drawer_stats(feat);
        Ok(PyDrawerStats {
            feat: s.feat,
            key_count: s.key_count,
            total_doc_refs: s.total_doc_refs,
            keys: s.keys,
        })
    }

    fn scan_bucket(&self, feat: u8, sim: u8) -> PyResult<Vec<u64>> {
        Ok(self.inner.scan_bucket(feat, sim))
    }
}

// ===== HSH-32 辅助函数 =====
#[pyfunction]
fn hsh32_is_antonym(sim_a: u32, sim_b: u32, tolerance: Option<u32>) -> bool {
    match tolerance {
        Some(t) => cabinet_hsh::is_antonym(sim_a, sim_b, t),
        None => is_antonym_default(sim_a, sim_b),
    }
}

#[pyfunction]
fn hsh32_direction_score(sim_q: u32, sim_d: u32) -> f32 {
    direction_score(sim_q, sim_d)
}

#[pyfunction]
fn hsh32_position_score(abs_q: u8, abs_d: u8) -> f32 {
    position_score(abs_q, abs_d)
}

#[pyfunction]
fn hsh32_frequency_score(freq: f32) -> f32 {
    frequency_score(freq)
}

#[pyfunction]
fn hsh32_compute_score(sim_q: u32, sim_d: u32, abs_q: u8, abs_d: u8, freq_d: f32) -> (f32, bool) {
    compute_score_default(sim_q, sim_d, abs_q, abs_d, freq_d)
}

// ===== QueryResult =====
#[pyclass(name = "QueryResult")]
#[derive(Clone)]
struct PyQueryResult {
    #[pyo3(get)]
    hsh: u32,
    #[pyo3(get)]
    doc_id: u64,
    #[pyo3(get)]
    position: u32,
    #[pyo3(get)]
    match_level: u8,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    text: Option<String>,
}

impl From<QueryResult> for PyQueryResult {
    fn from(r: QueryResult) -> Self {
        PyQueryResult {
            hsh: r.hsh,
            doc_id: r.doc_id,
            position: r.position,
            match_level: r.match_level,
            score: r.score,
            text: r.text,
        }
    }
}

#[pymethods]
impl PyQueryResult {
    fn __repr__(&self) -> String {
        format!(
            "QueryResult(hsh=0x{:05X}, doc_id={}, match_level={}, score={:.3})",
            self.hsh, self.doc_id, self.match_level, self.score
        )
    }
}

// ===== 模块注册 =====
#[pymodule]
fn _pycabinet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemory>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyHSHCode>()?;
    m.add_class::<PyHSHCode32>()?;
    m.add_class::<PyEncoder>()?;
    m.add_class::<PyMemoryStats>()?;
    m.add_class::<PyDrawerStats>()?;
    m.add_wrapped(wrap_pyfunction!(hsh32_is_antonym))?;
    m.add_wrapped(wrap_pyfunction!(hsh32_direction_score))?;
    m.add_wrapped(wrap_pyfunction!(hsh32_position_score))?;
    m.add_wrapped(wrap_pyfunction!(hsh32_frequency_score))?;
    m.add_wrapped(wrap_pyfunction!(hsh32_compute_score))?;
    Ok(())
}
