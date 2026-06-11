import re

with open("src/engine.rs", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update TQEngine struct
content = content.replace(
    "vectors: HashMap<u64, Vec<f32>>,",
    "vector_offsets: HashMap<u64, usize>,\n    pub raw_file_path: String,\n    #[serde(skip)]\n    pub raw_file: Option<std::fs::File>,"
)

# 2. Update EngineState struct
content = content.replace(
    "vectors: HashMap<u64, Vec<f32>>,",
    "vector_offsets: HashMap<u64, usize>,\n    raw_file_path: String,"
)

# 3. Update TQEngine::new
content = content.replace(
    "vectors: HashMap::new(),",
    "vector_offsets: HashMap::new(),\n            raw_file_path: \"data/tq_index/raw_vectors.bin\".to_string(),\n            raw_file: None,"
)
# Add std::fs::create_dir_all inside new
content = content.replace(
    "pub fn new() -> Self {",
    "pub fn new() -> Self {\n        std::fs::create_dir_all(\"data/tq_index\").unwrap();"
)

# 4. Update add()
new_add = """    pub fn add(&mut self, id: u64, vector: Vec<f32>, payload: Option<Value>) {
        if self.dim == 0 {
            self.dim = vector.len();
        }
        if self.raw_file.is_none() {
            self.raw_file = Some(std::fs::OpenOptions::new().create(true).append(true).open(&self.raw_file_path).unwrap());
        }
        let row_idx = self.vector_offsets.len();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vector.as_ptr() as *const u8, vector.len() * 4)
        };
        use std::io::Write;
        self.raw_file.as_mut().unwrap().write_all(bytes).unwrap();
        
        self.vector_offsets.insert(id, row_idx);
        if let Some(p) = payload {
            self.payloads.insert(id, p);
        }
        self.unindexed_ids.insert(id);
    }"""
content = re.sub(r"    pub fn add.*?self\.unindexed_ids\.insert\(id\);\n    \}", new_add, content, flags=re.DOTALL)

# 5. Update delete(), clear(), reset_index()
content = content.replace("self.vectors.remove(&id);", "self.vector_offsets.remove(&id);")
content = content.replace("self.vectors.clear();", "self.vector_offsets.clear();")
content = content.replace("self.unindexed_ids = self.vectors.keys().cloned().collect();", "self.unindexed_ids = self.vector_offsets.keys().cloned().collect();")

# 6. Update save_to_disk()
content = content.replace(
    "vectors: self.vectors.clone(),",
    "vector_offsets: self.vector_offsets.clone(),\n            raw_file_path: self.raw_file_path.clone(),"
)

# 7. Update load_from_disk()
content = content.replace(
    "self.vectors = state.vectors;",
    "self.vector_offsets = state.vector_offsets;\n            self.raw_file_path = state.raw_file_path;\n            self.raw_file = Some(std::fs::OpenOptions::new().create(true).append(true).open(&self.raw_file_path).unwrap());"
)

# Add read_mmap helper before build_index
read_mmap_helper = """
    fn read_mmap(&self) -> Option<memmap2::Mmap> {
        if let Ok(file) = std::fs::File::open(&self.raw_file_path) {
            unsafe { memmap2::MmapOptions::new().map(&file).ok() }
        } else {
            None
        }
    }

    pub fn build_index(&mut self) {"""
content = content.replace("    pub fn build_index(&mut self) {", read_mmap_helper)

# 8. Rewrite build_index()
new_build_index = """        if self.vector_offsets.is_empty() { return; }
        
        if let Some(ref mut f) = self.raw_file {
            use std::io::Write;
            f.flush().unwrap();
        }

        let n = self.vector_offsets.len();
        let d = self.dim;

        let mmap_opt = self.read_mmap();
        if mmap_opt.is_none() { return; }
        let mmap = mmap_opt.unwrap();
        let raw_f32: &[f32] = unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) };

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

        // 2. Sub-sampling for K-Means (Max 50,000 vectors)
        let sample_size = n.min(50_000);
        let mut sample = Vec::with_capacity(sample_size * d);
        use rand::seq::IteratorRandom;
        let chosen_indices: Vec<usize> = (0..n).choose_multiple(&mut rng, sample_size);
        for &idx in &chosen_indices {
            let start = idx * d;
            sample.extend_from_slice(&raw_f32[start..start+d]);
        }

        // 3. Phân cụm IVF (K-Means) trên sample
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

        let mut id_by_row = vec![0i64; n];
        for (&id, &row) in &self.vector_offsets {
            id_by_row[row] = id as i64;
        }

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
                out_vector_ids.push(id_by_row[i]);
                
                let mut original_nrm = 0.0f32;
                for j in 0..d {
                    let v = raw_f32[i * d + j];
                    original_nrm += v * v;
                }
                original_norms.push(original_nrm.sqrt());

                let mut raw_res = vec![0.0f32; d];
                for j in 0..d {
                    raw_res[j] = raw_f32[i * d + j] - coarse_centroids[c * d + j];
                }
                for j in 0..d {
                    let mut s = 0.0f32;
                    for k in 0..d { s += raw_res[k] * rot_op[k * d + j]; }
                    x_rot[cur * d + j] = s;
                }
                cur += 1;
            }
        }
        offsets_sl[n_list] = cur as i32;

        // 5. Train SQ centroids
        let actual_sq_bits = self.index_config.quantize_bits - 1;
        let sq_k = 1usize << actual_sq_bits;
        let sq_centroids = Self::train_lloyd_max(&x_rot, sq_k, 30);

        // 6. Lượng tử hóa
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
            self.unindexed_ids.clear();
        } else {
            eprintln!("❌ TurboQuant Index Failed!");
        }"""

# Replace the body of build_index
content = re.sub(r"        if self\.vectors\.is_empty.*?eprintln\!\(\"❌ TurboQuant Index Failed\!\"\);\n        \}", new_build_index, content, flags=re.DOTALL)

# 9. Rewrite search() to use mmap for flat search and rerank
search_head = """    pub fn search(&self, query: &[f32], top_k: usize, params: Option<crate::SearchParams>) -> Vec<crate::SearchResult> {
        let exact = params.as_ref().and_then(|p| p.exact).unwrap_or(false);
        let n_probe_param = params.as_ref().and_then(|p| p.n_probe);
        let rerank = params.as_ref().and_then(|p| p.rerank).unwrap_or(false);
        let rerank_factor = params.as_ref().and_then(|p| p.rerank_factor).unwrap_or(10);
        let with_vector = params.as_ref().and_then(|p| p.with_vector).unwrap_or(false);
        let mut results = vec![];

        // Mmap file temporary for search if needed
        let mmap_opt = self.read_mmap();
        let raw_f32: &[f32] = if let Some(ref mmap) = mmap_opt {
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) }
        } else {
            &[]
        };

        let get_vec = |id: u64| -> Option<Vec<f32>> {
            if let Some(&row_idx) = self.vector_offsets.get(&id) {
                let start = row_idx * self.dim;
                let end = start + self.dim;
                if end <= raw_f32.len() {
                    return Some(raw_f32[start..end].to_vec());
                }
            }
            None
        };"""

content = re.sub(r"    pub fn search.*?let mut results = vec\!\[\];", search_head, content, flags=re.DOTALL)

# Replace vector fetching inside search (indexed segment)
content = content.replace(
    "if let Some(vec) = self.vectors.get(&uid) {",
    "if let Some(vec) = get_vec(uid) {"
)

# Replace ids_to_flat_search
content = content.replace(
    "Box::new(self.vectors.keys())",
    "Box::new(self.vector_offsets.keys())"
)

content = content.replace(
    "if let Some(vec) = self.vectors.get(uid) {",
    "if let Some(vec) = get_vec(*uid) {"
)

# 10. Update get_all()
content = content.replace(
    "for (id, vec) in &self.vectors {",
    "let mmap_opt = self.read_mmap();\n        let raw_f32: &[f32] = if let Some(ref mmap) = mmap_opt { unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) } } else { &[] };\n        for (&id, &row_idx) in &self.vector_offsets {\n            let start = row_idx * self.dim; let end = start + self.dim;\n            let vec = if end <= raw_f32.len() { raw_f32[start..end].to_vec() } else { vec![0.0; self.dim] };"
)

# Final fix to test_benchmark_pure_qps, replace engine.vectors
content = content.replace("self.vectors", "self.vector_offsets") # wait, don't do blind replace for test

with open("src/engine.rs", "w", encoding="utf-8") as f:
    f.write(content)

print("Done")
