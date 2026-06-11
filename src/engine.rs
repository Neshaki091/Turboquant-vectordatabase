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
    pub max_training_samples: Option<usize>, // User config for max samples
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
    // Dữ liệu thô lưu trên đĩa để chạy Edge (1GB RAM)
    vector_offsets: HashMap<u64, usize>,
    raw_file_path: String,
    #[serde(skip)]
    pub raw_file: Option<std::fs::File>,
    
    payload_offsets: HashMap<u64, (u64, u32)>,
    payload_file_path: String,
    #[serde(skip)]
    pub payload_file: Option<std::fs::File>,

    // Phân mảnh đã nén
    pub indexed_segment: Option<IndexedSegment>,
    
    // Buffer tạm thời cho các vector mới (chưa nén)
    pub unindexed_ids: HashSet<u64>,

    pub dim: usize,
    pub index_config: IndexConfig, // Lưu trữ cấu hình Index
}

#[derive(Serialize, Deserialize)]
struct EngineState {
    vector_offsets: HashMap<u64, usize>,
    raw_file_path: String,
    payload_offsets: HashMap<u64, (u64, u32)>,
    payload_file_path: String,
    unindexed_ids: HashSet<u64>,
    dim: usize,
    index_config: IndexConfig,
}

impl TQEngine {
    pub fn new() -> Self {
        std::fs::create_dir_all("data/tq_index").unwrap();
        Self {
            vector_offsets: HashMap::new(),
            raw_file_path: "data/tq_index/raw_vectors.bin".to_string(),
            raw_file: None,
            payload_offsets: HashMap::new(),
            payload_file_path: "data/tq_index/payloads.bin".to_string(),
            payload_file: None,
            indexed_segment: None,
            unindexed_ids: HashSet::new(),
            dim: 0,
            index_config: IndexConfig { n_list: None, quantize_bits: 4, max_training_samples: None },
        }
    }

    pub fn add(&mut self, id: u64, vector: Vec<f32>, payload: Option<Value>) {
        if self.dim == 0 {
            self.dim = vector.len();
        }
        if self.raw_file.is_none() {
            self.raw_file = Some(std::fs::OpenOptions::new().read(true).create(true).append(true).open(&self.raw_file_path).unwrap());
        }
        let row_idx = self.vector_offsets.len();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vector.as_ptr() as *const u8, vector.len() * 4)
        };
        use std::io::Write;
        self.raw_file.as_mut().unwrap().write_all(bytes).unwrap();
        
        self.vector_offsets.insert(id, row_idx);
        
        if self.payload_file.is_none() {
            self.payload_file = Some(std::fs::OpenOptions::new().create(true).append(true).read(true).open(&self.payload_file_path).unwrap());
        }
        
        let mut p_offset = 0;
        let mut p_len = 0;
        if let Some(p) = payload {
            let json_str = serde_json::to_string(&p).unwrap();
            let bytes = json_str.as_bytes();
            p_len = bytes.len() as u32;
            let mut file = self.payload_file.as_mut().unwrap();
            use std::io::{Seek, SeekFrom, Write};
            p_offset = file.seek(SeekFrom::End(0)).unwrap();
            file.write_all(bytes).unwrap();
        }
        if p_len > 0 {
            self.payload_offsets.insert(id, (p_offset, p_len));
        }

        self.unindexed_ids.insert(id);
    }

    pub fn delete(&mut self, id: u64) {
        self.vector_offsets.remove(&id);
        self.payload_offsets.remove(&id);
        self.unindexed_ids.remove(&id);
    }

    pub fn clear(&mut self) {
        self.vector_offsets.clear();
        self.payload_offsets.clear();
        self.indexed_segment = None;
        self.unindexed_ids.clear();
        if let Some(file) = &mut self.payload_file {
            let _ = file.set_len(0);
        }
        if let Some(file) = &mut self.raw_file {
            let _ = file.set_len(0);
        }
    }

    pub fn reset_index(&mut self) {
        self.indexed_segment = None;
        self.unindexed_ids = self.vector_offsets.keys().cloned().collect();
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
        
        let state = EngineState {
            vector_offsets: self.vector_offsets.clone(), 
            raw_file_path: self.raw_file_path.clone(),
            payload_offsets: self.payload_offsets.clone(),
            payload_file_path: self.payload_file_path.clone(),
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
            self.vector_offsets = state.vector_offsets;
            self.raw_file_path = state.raw_file_path;
            self.raw_file = Some(std::fs::OpenOptions::new().read(true).create(true).append(true).open(&self.raw_file_path).unwrap());
            self.unindexed_ids = state.unindexed_ids;
            self.dim = state.dim;
            self.index_config = state.index_config;
            self.payload_offsets = state.payload_offsets;
            self.payload_file_path = state.payload_file_path;
            self.payload_file = Some(std::fs::OpenOptions::new().create(true).append(true).read(true).open(&self.payload_file_path).unwrap());
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
        if self.vector_offsets.is_empty() { return; }
        
        let n = self.vector_offsets.len();
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

        // We must flush raw_file before mmapping
        if let Some(ref mut f) = self.raw_file {
            use std::io::Write;
            let _ = f.flush();
        }

        let mmap_opt = self.read_mmap();
        if mmap_opt.is_none() { return; }
        let mmap = mmap_opt.unwrap();
        let raw_f32: &[f32] = unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) };

        let mut id_by_row = vec![0i64; n];
        for (&id, &row) in &self.vector_offsets {
            id_by_row[row] = id as i64;
        }

        // 2. Lấy mẫu ngẫu nhiên cho K-Means để tiết kiệm RAM
        let sample_limit = self.index_config.max_training_samples.unwrap_or(50_000);
        let sample_size = n.min(sample_limit);
        let mut sample = Vec::with_capacity(sample_size * d);
        use rand::seq::IteratorRandom;
        let chosen_indices: Vec<usize> = (0..n).choose_multiple(&mut rng, sample_size);
        for &idx in &chosen_indices {
            let start = idx * d;
            sample.extend_from_slice(&raw_f32[start..start+d]);
        }

        // 3. Phân cụm IVF (K-Means) trên mẫu nhỏ
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
        let coarse_centroids = crate::turboquant::tq_kmeans_train(&sample, sample_size, d, n_list, 15);
        let assignments = crate::turboquant::tq_assign_clusters(raw_f32, n, d, &coarse_centroids, n_list);

        let mut cluster_to_indices = vec![Vec::new(); n_list];
        for i in 0..n {
            cluster_to_indices[assignments[i] as usize].push(i);
        }

        // 4. Tạo offsets và x_rot (Residuals) - Dùng MMAP để tránh OOM 740MB RAM
        let mut offsets_sl = vec![0i32; n_list + 1];
        let mut cur = 0;
        for c in 0..n_list {
            offsets_sl[c] = cur as i32;
            cur += cluster_to_indices[c].len();
        }
        offsets_sl[n_list] = cur as i32;

        let x_rot_path = "data/tq_index/temp_x_rot.bin";
        let x_rot_file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(x_rot_path).unwrap();
        x_rot_file.set_len((n * d * 4) as u64).unwrap();
        let mut x_rot_mmap = unsafe { memmap2::MmapMut::map_mut(&x_rot_file).unwrap() };
        let mut out_vector_ids = vec![0i64; n];
        let mut original_norms = vec![0.0f32; n];

        struct SyncPtr<T>(*mut T);
        unsafe impl<T> Send for SyncPtr<T> {}
        unsafe impl<T> Sync for SyncPtr<T> {}

        let x_rot_ptr = SyncPtr(x_rot_mmap.as_mut_ptr() as *mut f32);
        let ids_ptr = SyncPtr(out_vector_ids.as_mut_ptr());
        let norms_ptr = SyncPtr(original_norms.as_mut_ptr());

        use rayon::prelude::*;
        (0..n_list).into_par_iter().for_each(|c| {
            let cur_start = offsets_sl[c] as usize;
            for (idx, &i) in cluster_to_indices[c].iter().enumerate() {
                let cur = cur_start + idx;
                
                unsafe { std::ptr::write(ids_ptr.0.add(cur), id_by_row[i]); }
                
                let mut original_nrm = 0.0f32;
                for j in 0..d {
                    let v = raw_f32[i * d + j];
                    original_nrm += v * v;
                }
                unsafe { std::ptr::write(norms_ptr.0.add(cur), original_nrm.sqrt()); }

                let mut raw_res = vec![0.0f32; d];
                for j in 0..d {
                    raw_res[j] = raw_f32[i * d + j] - coarse_centroids[c * d + j];
                }

                for j in 0..d {
                    let mut s = 0.0f32;
                    for k in 0..d { s += raw_res[k] * rot_op[k * d + j]; }
                    unsafe { std::ptr::write(x_rot_ptr.0.add(cur * d + j), s); }
                }
            }
        });
        
        let x_rot: &[f32] = unsafe { std::slice::from_raw_parts(x_rot_mmap.as_ptr() as *const f32, n * d) };

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
        
        // Dọn dẹp file MMAP tạm thời
        drop(x_rot_mmap);
        std::fs::remove_file(x_rot_path).ok();
    }

    fn read_payload_mmap(&self) -> Option<memmap2::Mmap> {
        if let Some(ref file) = self.payload_file {
            unsafe { memmap2::MmapOptions::new().map(file).ok() }
        } else if let Ok(file) = std::fs::File::open(&self.payload_file_path) {
            unsafe { memmap2::MmapOptions::new().map(&file).ok() }
        } else {
            None
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize, params: Option<crate::SearchParams>) -> Vec<crate::SearchResult> {
        let exact = params.as_ref().and_then(|p| p.exact).unwrap_or(false);
        let n_probe_param = params.as_ref().and_then(|p| p.n_probe);
        let rerank = params.as_ref().and_then(|p| p.rerank).unwrap_or(false);
        let rerank_factor = params.as_ref().and_then(|p| p.rerank_factor).unwrap_or(10);
        let with_vector = params.as_ref().and_then(|p| p.with_vector).unwrap_or(false);
        let mut results = vec![];

        let mmap_opt = self.read_mmap();
        let raw_f32: &[f32] = if let Some(ref mmap) = mmap_opt { unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) } } else { &[] };
        
        let get_vec = |id: u64| -> Option<&[f32]> {
            if let Some(&row_idx) = self.vector_offsets.get(&id) {
                let start = row_idx * self.dim;
                let end = start + self.dim;
                if end <= raw_f32.len() {
                    return Some(&raw_f32[start..end]);
                }
            }
            None
        };
        
        let payload_mmap = self.read_payload_mmap();
        let get_payload = |id: u64| -> Option<Value> {
            if let Some(&(offset, len)) = self.payload_offsets.get(&id) {
                if let Some(ref mmap) = payload_mmap {
                    let start = offset as usize;
                    let end = start + len as usize;
                    if end <= mmap.len() {
                        return serde_json::from_slice(&mmap[start..end]).ok();
                    }
                }
            }
            None
        };

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
                        if let Some(vec) = get_vec(uid) {
                            let score = if rerank {
                                // Compute exact dot product for re-ranking
                                query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum()
                            } else {
                                scores[i]
                            };

                            results.push(crate::SearchResult {
                                id: uid,
                                score,
                                vector: if with_vector { get_vec(uid).map(|s| s.to_vec()) } else { None },
                                payload: get_payload(uid),
                            });
                        }
                    }
                }
            }
        }

        // 2. Search trên Unindexed Buffer (Flat Search)
        // Nếu exact=true hoặc chưa có Index, quét TOÀN BỘ vectors. Nếu không, chỉ quét unindexed_ids
        let ids_to_flat_search: Box<dyn Iterator<Item = &u64>> = if exact || self.indexed_segment.is_none() {
            Box::new(self.vector_offsets.keys())
        } else {
            Box::new(self.unindexed_ids.iter())
        };

        let mut flat_results = std::collections::BinaryHeap::with_capacity(top_k);
        
        #[inline(always)]
        fn f32_to_u32(f: f32) -> u32 {
            let bits = f.to_bits();
            if bits & 0x80000000 != 0 { !bits } else { bits | 0x80000000 }
        }
        #[inline(always)]
        fn u32_to_f32(u: u32) -> f32 {
            let bits = if u & 0x80000000 == 0 { !u } else { u & 0x7FFFFFFF };
            f32::from_bits(bits)
        }

        for uid in ids_to_flat_search {
            if let Some(vec_slice) = get_vec(*uid) {
                let mut score: f32 = 0.0;
                for j in 0..self.dim {
                    score += query[j] * vec_slice[j];
                }
                
                let cmp_score = std::cmp::Reverse(f32_to_u32(score));
                if flat_results.len() < top_k {
                    flat_results.push((cmp_score, *uid));
                } else if cmp_score.0 > flat_results.peek().unwrap().0.0 {
                    flat_results.pop();
                    flat_results.push((cmp_score, *uid));
                }
            }
        }
        
        for (std::cmp::Reverse(u32_score), uid) in flat_results.into_iter() {
            let score = u32_to_f32(u32_score);
            if !results.iter().any(|r| r.id == uid) {
                results.push(crate::SearchResult {
                    id: uid,
                    score,
                    vector: if with_vector { get_vec(uid).map(|s| s.to_vec()) } else { None },
                    payload: get_payload(uid),
                });
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
    
    fn read_mmap(&self) -> Option<memmap2::Mmap> {
        if let Some(ref file) = self.raw_file {
            unsafe { memmap2::MmapOptions::new().map(file).ok() }
        } else if let Ok(file) = std::fs::File::open(&self.raw_file_path) {
            unsafe { memmap2::MmapOptions::new().map(&file).ok() }
        } else {
            None
        }
    }
    
    pub fn get_all(&self) -> Vec<crate::PointDetail> {
        let mut results = vec![];
        let mmap_opt = self.read_mmap();
        let raw_f32: &[f32] = if let Some(ref mmap) = mmap_opt { unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) } } else { &[] };
        let payload_mmap = self.read_payload_mmap();
        
        for (&id, &row_idx) in &self.vector_offsets {
            let start = row_idx * self.dim;
            let end = start + self.dim;
            let vec = if end <= raw_f32.len() { raw_f32[start..end].to_vec() } else { vec![0.0; self.dim] };
            
            let mut payload = None;
            if let Some(&(offset, len)) = self.payload_offsets.get(&id) {
                if let Some(ref mmap) = payload_mmap {
                    let p_start = offset as usize;
                    let p_end = p_start + len as usize;
                    if p_end <= mmap.len() {
                        payload = serde_json::from_slice(&mmap[p_start..p_end]).ok();
                    }
                }
            }
            
            results.push(crate::PointDetail {
                id,
                vector: vec,
                payload,
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
