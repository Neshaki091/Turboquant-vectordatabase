use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Serialize, Deserialize};
use std::collections::HashSet;
use ndarray::Array1;
use ndarray_npy::{read_npy, write_npy};
use std::path::Path;

#[derive(Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    pub n_list: Option<usize>, // Cấu hình số cụm IVF (nếu None sẽ tự động tính sqrt(N))
    pub quantize_bits: usize,  // 2 cho 2-bit, 4 cho 4-bit
}

#[derive(Serialize, Deserialize, Clone)]
pub struct IndexedSegment {
    pub n_vectors: usize,
    pub rot_op: Vec<f32>,
    pub sq_centroids: Vec<f32>,
    pub sq_flat: Vec<u8>,
    pub signs_flat: Vec<u8>,
    pub norms_sl: Vec<f32>,
    pub res_norms: Vec<f32>,
    pub vector_ids: Vec<i64>,
    pub coarse_view: Vec<f32>,
    pub offsets_sl: Vec<i32>,
}

// Cấu trúc lõi Vector Database
#[derive(Serialize, Deserialize)]
pub struct TQEngine {
    // Dữ liệu thô gốc (cho mục đích fallback và trả về payload)
    vectors: HashMap<u64, Vec<f32>>,
    payloads: HashMap<u64, Value>,

    // Phân mảnh đã nén
    pub indexed_segment: Option<IndexedSegment>,
    
    // Buffer tạm thời cho các vector mới (chưa nén)
    pub unindexed_ids: HashSet<u64>,

    pub dim: usize,
    pub index_config: IndexConfig, // Lưu trữ cấu hình Index
}

#[derive(Serialize, Deserialize)]
struct EngineState {
    vectors: HashMap<u64, Vec<f32>>,
    payload_strings: HashMap<u64, String>,
    unindexed_ids: HashSet<u64>,
    dim: usize,
    index_config: IndexConfig,
}

impl TQEngine {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            payloads: HashMap::new(),
            indexed_segment: None,
            unindexed_ids: HashSet::new(),
            dim: 0,
            index_config: IndexConfig { n_list: None, quantize_bits: 4 }, // Default TurboQuant 4-bit
        }
    }

    pub fn add(&mut self, id: u64, vector: Vec<f32>, payload: Option<Value>) {
        if self.dim == 0 {
            self.dim = vector.len();
        }
        self.vectors.insert(id, vector);
        if let Some(p) = payload {
            self.payloads.insert(id, p);
        }
        self.unindexed_ids.insert(id);
    }

    pub fn delete(&mut self, id: u64) {
        self.vectors.remove(&id);
        self.payloads.remove(&id);
        self.unindexed_ids.remove(&id);
    }

    pub fn clear(&mut self) {
        self.vectors.clear();
        self.payloads.clear();
        self.indexed_segment = None;
        self.unindexed_ids.clear();
    }

    pub fn reset_index(&mut self) {
        self.indexed_segment = None;
        self.unindexed_ids = self.vectors.keys().cloned().collect();
    }

    pub fn save_to_disk(&self, dir_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir_path)?;

        if let Some(seg) = &self.indexed_segment {
            write_npy(format!("{}/rot_op.npy", dir_path), &Array1::from_vec(seg.rot_op.clone()))?;
            write_npy(format!("{}/sq_centroids.npy", dir_path), &Array1::from_vec(seg.sq_centroids.clone()))?;
            write_npy(format!("{}/sq_codes.npy", dir_path), &Array1::from_vec(seg.sq_flat.clone()))?;
            write_npy(format!("{}/qjl_signs.npy", dir_path), &Array1::from_vec(seg.signs_flat.clone()))?;
            write_npy(format!("{}/norms.npy", dir_path), &Array1::from_vec(seg.norms_sl.clone()))?;
            write_npy(format!("{}/res_norms.npy", dir_path), &Array1::from_vec(seg.res_norms.clone()))?;
            write_npy(format!("{}/vector_ids.npy", dir_path), &Array1::from_vec(seg.vector_ids.clone()))?;
            write_npy(format!("{}/coarse_centroids.npy", dir_path), &Array1::from_vec(seg.coarse_view.clone()))?;
            write_npy(format!("{}/list_offsets.npy", dir_path), &Array1::from_vec(seg.offsets_sl.clone()))?;
            
            let meta = serde_json::json!({
                "n_vectors": seg.n_vectors,
                "dim": self.dim,
                "quantize_bits": self.index_config.quantize_bits,
                "n_list": seg.offsets_sl.len() - 1,
            });
            std::fs::write(format!("{}/metadata.json", dir_path), serde_json::to_string_pretty(&meta)?)?;
        }

        let file = File::create(format!("{}/engine_state.bin", dir_path))?;
        let writer = BufWriter::new(file);
        
        // Chuyển serde_json::Value sang String để tránh lỗi bincode DeserializeAnyNotSupported
        let payload_strings: HashMap<u64, String> = self.payloads.iter()
            .map(|(k, v)| (*k, serde_json::to_string(v).unwrap_or_default()))
            .collect();

        let state = EngineState {
            vectors: self.vectors.clone(),
            payload_strings,
            unindexed_ids: self.unindexed_ids.clone(),
            dim: self.dim,
            index_config: self.index_config.clone(),
        };
        bincode::serialize_into(writer, &state)?;
        Ok(())
    }

    pub fn load_from_disk(&mut self, dir_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let state_path = format!("{}/engine_state.bin", dir_path);
        if Path::new(&state_path).exists() {
            let file = File::open(state_path)?;
            let reader = BufReader::new(file);
            let state: EngineState = bincode::deserialize_from(reader)?;
            self.vectors = state.vectors;
            self.unindexed_ids = state.unindexed_ids;
            self.dim = state.dim;
            self.index_config = state.index_config;
            
            // Khôi phục lại serde_json::Value
            self.payloads = state.payload_strings.into_iter()
                .filter_map(|(k, v)| serde_json::from_str(&v).ok().map(|val| (k, val)))
                .collect();
        }

        if Path::new(&format!("{}/sq_codes.npy", dir_path)).exists() {
            let rot_op: Array1<f32> = read_npy(format!("{}/rot_op.npy", dir_path))?;
            let sq_centroids: Array1<f32> = read_npy(format!("{}/sq_centroids.npy", dir_path))?;
            let sq_flat: Array1<u8> = read_npy(format!("{}/sq_codes.npy", dir_path))?;
            let signs_flat: Array1<u8> = read_npy(format!("{}/qjl_signs.npy", dir_path))?;
            let norms_sl: Array1<f32> = read_npy(format!("{}/norms.npy", dir_path))?;
            let vector_ids: Array1<i64> = read_npy(format!("{}/vector_ids.npy", dir_path))?;
            let coarse_view: Array1<f32> = read_npy(format!("{}/coarse_centroids.npy", dir_path))?;
            let offsets_sl: Array1<i32> = read_npy(format!("{}/list_offsets.npy", dir_path))?;
            
            let meta_str = std::fs::read_to_string(format!("{}/metadata.json", dir_path))?;
            let meta: Value = serde_json::from_str(&meta_str)?;
            let n_vectors = meta["n_vectors"].as_u64().unwrap_or(0) as usize;

            let res_norms: Vec<f32> = if Path::new(&format!("{}/res_norms.npy", dir_path)).exists() {
                let r: Array1<f32> = read_npy(format!("{}/res_norms.npy", dir_path))?;
                r.into_raw_vec()
            } else {
                vec![1.0; n_vectors]
            };

            self.indexed_segment = Some(IndexedSegment {
                n_vectors,
                rot_op: rot_op.into_raw_vec(),
                sq_centroids: sq_centroids.into_raw_vec(),
                sq_flat: sq_flat.into_raw_vec(),
                signs_flat: signs_flat.into_raw_vec(),
                norms_sl: norms_sl.into_raw_vec(),
                res_norms,
                vector_ids: vector_ids.into_raw_vec(),
                coarse_view: coarse_view.into_raw_vec(),
                offsets_sl: offsets_sl.into_raw_vec(),
            });
        }
        Ok(())
    }

fn train_lloyd_max(x_rot: &[f32], sq_k: usize, iters: usize) -> Vec<f32> {
    let total_elements = x_rot.len();
    if total_elements == 0 {
        return vec![0.0; sq_k];
    }
    
    let sample_len = 1_000_000.min(total_elements);
    let mut sample = Vec::with_capacity(sample_len);
    let mut rng = rand::rng();
    
    if total_elements <= 1_000_000 {
        sample.extend_from_slice(x_rot);
    } else {
        use rand::RngExt;
        for _ in 0..sample_len {
            let idx = rng.random_range(0..total_elements);
            sample.push(x_rot[idx]);
        }
    }
    
    sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q01_idx = (sample.len() as f64 * 0.01) as usize;
    let q99_idx = (sample.len() as f64 * 0.99) as usize;
    let q_low = sample[q01_idx];
    let q_high = sample[q99_idx];
    for v in &mut sample {
        *v = v.clamp(q_low, q_high);
    }
    
    let mut centroids = vec![0.0f32; sq_k];
    for i in 0..sq_k {
        let p_mid = (i as f64 + 0.5) / (sq_k as f64);
        let idx = (sample.len() as f64 * p_mid) as usize;
        centroids[i] = sample[idx.min(sample.len() - 1)];
    }
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    for _ in 0..iters {
        let mut boundaries = vec![0.0f32; sq_k + 1];
        boundaries[0] = f32::NEG_INFINITY;
        boundaries[sq_k] = f32::INFINITY;
        for i in 0..(sq_k - 1) {
            boundaries[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
        }
        
        let mut sums = vec![0.0f64; sq_k];
        let mut counts = vec![0u64; sq_k];
        
        for &v in &sample {
            let mut lo = 0;
            let mut hi = sq_k;
            while lo + 1 < hi {
                let mid = (lo + hi) >> 1;
                if v >= boundaries[mid] {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            sums[lo] += v as f64;
            counts[lo] += 1;
        }
        
        let mut new_centroids = centroids.clone();
        for i in 0..sq_k {
            if counts[i] > 0 {
                new_centroids[i] = (sums[i] / counts[i] as f64) as f32;
            }
        }
        new_centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut diff = 0.0f32;
        for i in 0..sq_k {
            diff += (centroids[i] - new_centroids[i]).abs();
        }
        centroids = new_centroids;
        if diff < 1e-5 {
            break;
        }
    }
    centroids
}

    pub fn build_index(&mut self) {
        if self.vectors.is_empty() { return; }
        
        let n = self.vectors.len();
        let d = self.dim;

        // 1. Ma trận xoay Orthogonal (Random) dùng Gram-Schmidt
        let mut rot_op = vec![0.0f32; d * d];
        use rand::RngExt;
        let mut rng = rand::rng();
        for i in 0..d {
            for j in 0..d {
                rot_op[i * d + j] = rng.random_range(-1.0..1.0);
            }
        }
        for i in 0..d {
            for j in 0..i {
                let mut dot = 0.0f32;
                for k in 0..d { dot += rot_op[i * d + k] * rot_op[j * d + k]; }
                for k in 0..d { rot_op[i * d + k] -= dot * rot_op[j * d + k]; }
            }
            let mut nrm = 0.0f32;
            for k in 0..d { nrm += rot_op[i * d + k] * rot_op[i * d + k]; }
            nrm = nrm.sqrt();
            if nrm > 0.0 {
                for k in 0..d { rot_op[i * d + k] /= nrm; }
            }
        }

        // 2. Thu thập x_arr và vector_ids
        let mut x_arr = vec![0.0f32; n * d];
        let mut raw_ids = vec![0i64; n];
        for (i, (&id, vec)) in self.vectors.iter().enumerate() {
            raw_ids[i] = id as i64;
            for j in 0..d { x_arr[i * d + j] = vec[j]; }
        }

        // 3. Phân cụm IVF (K-Means)
        let default_n_list = if n < 2 {
            1
        } else {
            let target = 2.0 * (n as f64).sqrt();
            let log2_target = target.log2();
            let p_lower = 2usize.pow(log2_target.floor() as u32);
            let p_upper = 2usize.pow(log2_target.ceil() as u32);
            if (target - p_lower as f64).abs() < (target - p_upper as f64).abs() {
                p_lower
            } else {
                p_upper
            }
        };
        let n_list = self.index_config.n_list.unwrap_or(default_n_list).min(n);
        let coarse_centroids = crate::turboquant::tq_kmeans_train(&x_arr, n, d, n_list, 15);
        let assignments = crate::turboquant::tq_assign_clusters(&x_arr, n, d, &coarse_centroids, n_list);

        let mut cluster_to_indices = vec![Vec::new(); n_list];
        for i in 0..n {
            cluster_to_indices[assignments[i] as usize].push(i);
        }

        // 4. Tạo offsets và x_rot (Residuals)
        let mut offsets_sl = vec![0i32; n_list + 1];
        let mut x_rot = vec![0.0f32; n * d];
        let mut out_vector_ids = Vec::with_capacity(n);
        let mut original_norms = Vec::with_capacity(n);
        
        let mut cur = 0;
        for c in 0..n_list {
            offsets_sl[c] = cur as i32;
            for &i in &cluster_to_indices[c] {
                out_vector_ids.push(raw_ids[i]);
                
                // Compute original vector norm
                let mut original_nrm = 0.0f32;
                for j in 0..d {
                    let v = x_arr[i * d + j];
                    original_nrm += v * v;
                }
                original_norms.push(original_nrm.sqrt());

                // Tính Residuals = X - C
                let mut raw_res = vec![0.0f32; d];
                for j in 0..d {
                    raw_res[j] = x_arr[i * d + j] - coarse_centroids[c * d + j];
                }
                // Áp dụng ma trận xoay: x_rot = raw_res * rot_op^T
                for j in 0..d {
                    let mut s = 0.0f32;
                    for k in 0..d { s += raw_res[k] * rot_op[k * d + j]; }
                    x_rot[cur * d + j] = s;
                }
                cur += 1;
            }
        }
        offsets_sl[n_list] = cur as i32;

        // 5. Train SQ centroids using Lloyd-Max (Gaussian codebook optimization)
        let actual_sq_bits = self.index_config.quantize_bits - 1;
        let sq_k = 1usize << actual_sq_bits;
        let sq_centroids = Self::train_lloyd_max(&x_rot, sq_k, 30);

        // 6. Lượng tử hóa theo cấu hình quantize_bits
        if let Ok((sq_codes, qjl_signs, res_norms)) = crate::turboquant::tq_quantize_rotated(&x_rot, n, d, &sq_centroids, actual_sq_bits as i32) {
            self.indexed_segment = Some(IndexedSegment {
                n_vectors: n,
                rot_op,
                sq_centroids,
                sq_flat: sq_codes,
                signs_flat: qjl_signs,
                norms_sl: original_norms,
                res_norms,
                vector_ids: out_vector_ids,
                coarse_view: coarse_centroids,
                offsets_sl,
            });
            self.unindexed_ids.clear(); // Xóa buffer sau khi đã nén xong tất cả
        } else {
            eprintln!("❌ TurboQuant Index Failed!");
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize, params: Option<crate::SearchParams>) -> Vec<crate::SearchResult> {
        let exact = params.as_ref().and_then(|p| p.exact).unwrap_or(false);
        let n_probe_param = params.as_ref().and_then(|p| p.n_probe);
        let rerank = params.as_ref().and_then(|p| p.rerank).unwrap_or(false);
        let rerank_factor = params.as_ref().and_then(|p| p.rerank_factor).unwrap_or(10);
        let with_vector = params.as_ref().and_then(|p| p.with_vector).unwrap_or(false);
        let mut results = vec![];

        // 1. Search trên Indexed Segment
        if let Some(segment) = &self.indexed_segment {
            if self.dim == query.len() && !exact {
                let n_list = segment.offsets_sl.len() - 1;
                let n_probe = n_probe_param.unwrap_or(n_list);
                
                // If reranking is enabled, retrieve more candidates for exact distance scoring
                let retrieve_k = if rerank { top_k * rerank_factor } else { top_k };
                
                let mut padded_cent = vec![0.0f32; 8];
                for i in 0..segment.sq_centroids.len().min(8) {
                    padded_cent[i] = segment.sq_centroids[i];
                }

                // Compute the correct mathematical QJL scaling factor
                let qjl_scale = (2.0f32 / std::f32::consts::PI).sqrt() / (self.dim as f32).sqrt();

                let (scores, ids) = crate::turboquant::tq_unified_search(
                    query,
                    &segment.rot_op,
                    &segment.coarse_view,
                    &segment.offsets_sl,
                    &segment.vector_ids,
                    &segment.sq_flat,
                    &padded_cent,
                    &segment.norms_sl,
                    &segment.signs_flat,
                    &segment.res_norms,
                    qjl_scale,
                    self.dim,
                    self.index_config.quantize_bits - 1,
                    n_probe,
                    retrieve_k,
                    None,
                );

                for (i, id) in ids.into_iter().enumerate() {
                    if id >= 0 {
                        let uid = id as u64;
                        // Tombstone filter: chỉ lấy nếu vector vẫn còn trong DB
                        if let Some(vec) = self.vectors.get(&uid) {
                            let score = if rerank {
                                // Compute exact dot product for re-ranking
                                query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum()
                            } else {
                                scores[i]
                            };

                            results.push(crate::SearchResult {
                                id: uid,
                                score,
                                vector: if with_vector { Some(vec.clone()) } else { None },
                                payload: self.payloads.get(&uid).cloned(),
                            });
                        }
                    }
                }
            }
        }

        // 2. Search trên Unindexed Buffer (Flat Search)
        // Nếu exact=true hoặc chưa có Index, quét TOÀN BỘ vectors. Nếu không, chỉ quét unindexed_ids
        let ids_to_flat_search: Box<dyn Iterator<Item = &u64>> = if exact || self.indexed_segment.is_none() {
            Box::new(self.vectors.keys())
        } else {
            Box::new(self.unindexed_ids.iter())
        };

        for uid in ids_to_flat_search {
            if let Some(vec) = self.vectors.get(uid) {
                let score: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                // Loại bỏ trùng lặp nếu có (mặc dù unindexed_ids và indexed_segment không nên giao nhau)
                if !results.iter().any(|r| r.id == *uid) {
                    results.push(crate::SearchResult {
                        id: *uid,
                        score,
                        vector: if with_vector { Some(vec.clone()) } else { None },
                        payload: self.payloads.get(uid).cloned(),
                    });
                }
            }
        }
        
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);
        results
    }

    pub fn search_batch(&self, queries: &[Vec<f32>], top_k: usize, params: Option<crate::SearchParams>) -> Vec<Vec<crate::SearchResult>> {
        use rayon::prelude::*;
        queries.par_iter()
            .map(|q| self.search(q, top_k, params.clone()))
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use ndarray::Array2;

    #[test]
    fn test_benchmark_pure_qps() {
        let mut engine = TQEngine::new();
        engine.load_from_disk("data/tq_index").unwrap();
        
        let npy_path = "e:\\ARQ-RAG\\ARQ-RAG-turboquant-main\\tq_java_test\\Qasper_E5\\corpus_embedded_norm.npy";
        let matrix: Array2<f32> = read_npy(npy_path).expect("Failed to load npy file");
        
        let mut queries = vec![];
        for i in 0..100 {
            let row: Vec<f32> = matrix.row(i).to_vec();
            queries.push(row);
        }

        // Warm up
        for q in &queries {
            let _ = engine.search(q, 16, Some(crate::SearchParams {
                exact: Some(false),
                n_probe: Some(16),
                rerank: Some(true),
                rerank_factor: Some(10),
                with_vector: Some(false),
            }));
        }

        let start = Instant::now();
        let loops = 10;
        for _ in 0..loops {
            for q in &queries {
                let _ = engine.search(q, 16, Some(crate::SearchParams {
                    exact: Some(false),
                    n_probe: Some(16),
                    rerank: Some(true),
                    rerank_factor: Some(10),
                    with_vector: Some(false),
                }));
            }
        }
        let duration = start.elapsed();
        let total_queries = loops * queries.len();
        let qps = (total_queries as f64) / duration.as_secs_f64();
        println!("\n=============================================");
        println!("🔥 PURE ENGINE BENCHMARK (In-Memory, No HTTP/JSON)");
        println!("=============================================");
        println!("Pure Engine QPS: {:.2} QPS", qps);
        println!("Duration for {} queries: {:?}", total_queries, duration);
        println!("=============================================\n");
    }
}
