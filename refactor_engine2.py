import re

with open("src/engine.rs", "r", encoding="utf-8") as f:
    text = f.read()

# 1. TQEngine Struct
text = text.replace(
"""pub struct TQEngine {
    // Dữ liệu thô gốc (cho mục đích fallback và trả về payload)
    vectors: HashMap<u64, Vec<f32>>,""",
"""pub struct TQEngine {
    // Dữ liệu thô lưu trên đĩa để chạy Edge (1GB RAM)
    vector_offsets: HashMap<u64, usize>,
    raw_file_path: String,
    #[serde(skip)]
    pub raw_file: Option<std::fs::File>,"""
)

# 2. EngineState Struct
text = text.replace(
"""struct EngineState {
    vectors: HashMap<u64, Vec<f32>>,""",
"""struct EngineState {
    vector_offsets: HashMap<u64, usize>,
    raw_file_path: String,"""
)

# 3. TQEngine::new
text = text.replace(
"""    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),""",
"""    pub fn new() -> Self {
        std::fs::create_dir_all("data/tq_index").unwrap();
        Self {
            vector_offsets: HashMap::new(),
            raw_file_path: "data/tq_index/raw_vectors.bin".to_string(),
            raw_file: None,"""
)

# 4. add()
old_add = """    pub fn add(&mut self, id: u64, vector: Vec<f32>, payload: Option<Value>) {
        if self.dim == 0 {
            self.dim = vector.len();
        }
        self.vectors.insert(id, vector);
        if let Some(p) = payload {
            self.payloads.insert(id, p);
        }
        self.unindexed_ids.insert(id);
    }"""
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
text = text.replace(old_add, new_add)

# 5. delete, clear, reset_index
text = text.replace("self.vectors.remove(&id);", "self.vector_offsets.remove(&id);")
text = text.replace("self.vectors.clear();", "self.vector_offsets.clear();")
text = text.replace("self.vectors.keys()", "self.vector_offsets.keys()")

# 6. save_to_disk
text = text.replace("vectors: self.vectors.clone(),", "vector_offsets: self.vector_offsets.clone(), raw_file_path: self.raw_file_path.clone(),")

# 7. load_from_disk
text = text.replace(
    "self.vectors = state.vectors;",
    "self.vector_offsets = state.vector_offsets;\n            self.raw_file_path = state.raw_file_path;\n            self.raw_file = Some(std::fs::OpenOptions::new().create(true).append(true).open(&self.raw_file_path).unwrap());"
)

# 8. get_all
old_get_all = """    pub fn get_all(&self) -> Vec<crate::PointDetail> {
        let mut results = vec![];
        for (id, vec) in &self.vectors {
            results.push(crate::PointDetail {
                id: *id,
                vector: vec.clone(),
                payload: self.payloads.get(id).cloned(),
            });
        }
        results
    }"""
new_get_all = """    fn read_mmap(&self) -> Option<memmap2::Mmap> {
        if let Ok(file) = std::fs::File::open(&self.raw_file_path) {
            unsafe { memmap2::MmapOptions::new().map(&file).ok() }
        } else {
            None
        }
    }
    
    pub fn get_all(&self) -> Vec<crate::PointDetail> {
        let mut results = vec![];
        let mmap_opt = self.read_mmap();
        let raw_f32: &[f32] = if let Some(ref mmap) = mmap_opt { unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) } } else { &[] };
        
        for (&id, &row_idx) in &self.vector_offsets {
            let start = row_idx * self.dim;
            let end = start + self.dim;
            let vec = if end <= raw_f32.len() { raw_f32[start..end].to_vec() } else { vec![0.0; self.dim] };
            results.push(crate::PointDetail {
                id,
                vector: vec,
                payload: self.payloads.get(&id).cloned(),
            });
        }
        results
    }"""
text = text.replace(old_get_all, new_get_all)

# 9. build_index
build_index_extract_old = """        // 2. Thu thập x_arr và vector_ids
        let mut x_arr = vec![0.0f32; n * d];
        let mut raw_ids = vec![0i64; n];
        for (i, (&id, vec)) in self.vectors.iter().enumerate() {
            raw_ids[i] = id as i64;
            for j in 0..d { x_arr[i * d + j] = vec[j]; }
        }

        // 3. Phân cụm IVF (K-Means)
        let default_n_list = if n < 2 {"""

build_index_extract_new = """        // We must flush raw_file before mmapping
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

        // 2. Lấy mẫu 50,000 vectors cho K-Means để tiết kiệm RAM
        let sample_size = n.min(50_000);
        let mut sample = Vec::with_capacity(sample_size * d);
        use rand::seq::IteratorRandom;
        let chosen_indices: Vec<usize> = (0..n).choose_multiple(&mut rng, sample_size);
        for &idx in &chosen_indices {
            let start = idx * d;
            sample.extend_from_slice(&raw_f32[start..start+d]);
        }

        // 3. Phân cụm IVF (K-Means) trên mẫu nhỏ
        let default_n_list = if n < 2 {"""
text = text.replace(build_index_extract_old, build_index_extract_new)

text = text.replace("let coarse_centroids = crate::turboquant::tq_kmeans_train(&x_arr, n, d, n_list, 15);", "let coarse_centroids = crate::turboquant::tq_kmeans_train(&sample, sample_size, d, n_list, 15);")
text = text.replace("let assignments = crate::turboquant::tq_assign_clusters(&x_arr, n, d, &coarse_centroids, n_list);", "let assignments = crate::turboquant::tq_assign_clusters(raw_f32, n, d, &coarse_centroids, n_list);")

text = text.replace(
"""        for c in 0..n_list {
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
                }""",
"""        for c in 0..n_list {
            offsets_sl[c] = cur as i32;
            for &i in &cluster_to_indices[c] {
                out_vector_ids.push(id_by_row[i]);
                
                // Compute original vector norm
                let mut original_nrm = 0.0f32;
                for j in 0..d {
                    let v = raw_f32[i * d + j];
                    original_nrm += v * v;
                }
                original_norms.push(original_nrm.sqrt());

                // Tính Residuals = X - C
                let mut raw_res = vec![0.0f32; d];
                for j in 0..d {
                    raw_res[j] = raw_f32[i * d + j] - coarse_centroids[c * d + j];
                }"""
)

text = text.replace("if self.vectors.is_empty() { return; }", "if self.vector_offsets.is_empty() { return; }")
text = text.replace("let n = self.vectors.len();", "let n = self.vector_offsets.len();")

# 10. search()
search_start_old = """    pub fn search(&self, query: &[f32], top_k: usize, params: Option<crate::SearchParams>) -> Vec<crate::SearchResult> {
        let exact = params.as_ref().and_then(|p| p.exact).unwrap_or(false);
        let n_probe_param = params.as_ref().and_then(|p| p.n_probe);
        let rerank = params.as_ref().and_then(|p| p.rerank).unwrap_or(false);
        let rerank_factor = params.as_ref().and_then(|p| p.rerank_factor).unwrap_or(10);
        let with_vector = params.as_ref().and_then(|p| p.with_vector).unwrap_or(false);
        let mut results = vec![];"""

search_start_new = """    pub fn search(&self, query: &[f32], top_k: usize, params: Option<crate::SearchParams>) -> Vec<crate::SearchResult> {
        let exact = params.as_ref().and_then(|p| p.exact).unwrap_or(false);
        let n_probe_param = params.as_ref().and_then(|p| p.n_probe);
        let rerank = params.as_ref().and_then(|p| p.rerank).unwrap_or(false);
        let rerank_factor = params.as_ref().and_then(|p| p.rerank_factor).unwrap_or(10);
        let with_vector = params.as_ref().and_then(|p| p.with_vector).unwrap_or(false);
        let mut results = vec![];

        let mmap_opt = self.read_mmap();
        let raw_f32: &[f32] = if let Some(ref mmap) = mmap_opt { unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, mmap.len() / 4) } } else { &[] };
        
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
text = text.replace(search_start_old, search_start_new)

text = text.replace("if let Some(vec) = self.vectors.get(&uid) {", "if let Some(vec) = get_vec(uid) {")
text = text.replace("if let Some(vec) = self.vectors.get(uid) {", "if let Some(vec) = get_vec(*uid) {")

text = text.replace("Box::new(self.vectors.keys())", "Box::new(self.vector_offsets.keys())")

with open("src/engine.rs", "w", encoding="utf-8") as f:
    f.write(text)

print("Done")
