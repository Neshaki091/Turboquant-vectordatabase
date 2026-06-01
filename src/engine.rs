use serde_json::Value;
use std::collections::HashMap;

// Cấu trúc lõi Vector Database
pub struct TQEngine {
    // Trong thực tế, dữ liệu này sẽ được map thẳng xuống ổ cứng bằng mmap
    vectors: HashMap<u64, Vec<f32>>,
    payloads: HashMap<u64, Value>,
}

impl TQEngine {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            payloads: HashMap::new(),
        }
    }

    pub fn add(&mut self, id: u64, vector: Vec<f32>, payload: Option<Value>) {
        self.vectors.insert(id, vector);
        if let Some(p) = payload {
            self.payloads.insert(id, p);
        }
    }

    pub fn delete(&mut self, id: u64) {
        self.vectors.remove(&id);
        self.payloads.remove(&id);
    }

    pub fn clear(&mut self) {
        self.vectors.clear();
        self.payloads.clear();
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<crate::SearchResult> {
        // TẠM THỜI: Thuật toán Cosine cơ bản để chứng minh API hoạt động.
        // TƯƠNG LAI: Chèn mã nguồn SIMD 4-bit (SQ+QJL) của TurboQuant vào đây!
        let mut results = vec![];
        for (id, vec) in &self.vectors {
            let score: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            results.push(crate::SearchResult {
                id: *id,
                score,
                vector: Some(vec.clone()),
                payload: self.payloads.get(id).cloned(),
            });
        }
        
        // Sắp xếp giảm dần theo điểm
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }
    
    pub fn get_all(&self) -> Vec<crate::PointDetail> {
        let mut results = vec![];
        for (id, vec) in &self.vectors {
            results.push(crate::PointDetail {
                id: *id,
                vector: vec.clone(),
                payload: self.payloads.get(id).cloned(),
            });
        }
        results
    }
}
