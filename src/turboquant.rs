use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// Quantization helpers (index-time)
// - SQ bits supported: 1 or 3 (TurboQuant 2b/4b)
// =============================================================================

pub fn tq_quantize_rotated(
    
    x_rot: &[f32],        // (N, D) rotated residual-space vectors
    sq_centroids: &[f32], // (K,) scalar codebook (K=2 or 8)
    sq_bits: i32,                           // 1 or 3
) -> PyResult<(Py<PyArray2<u8>>, Py<PyArray2<u8>>, Py<PyArray1<f32>>)> {
    let x_ro = x_rot.readonly();
    let x = x_ro.as_array();
    let n = x.shape()[0];
    let d = x.shape()[1];

    let cent_ro = sq_centroids.readonly();
    let cent = cent_ro.as_slice()?;

    let bits = sq_bits as usize;
    if bits != 1 && bits != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("tq_quantize_rotated only supports sq_bits=1 or 3 (got {})", bits),
        ));
    }

    let k = 1usize << bits;
    if cent.len() != k {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("sq_centroids length must be {} for sq_bits={}, got {}", k, bits, cent.len()),
        ));
    }

    let mut boundaries = vec![0.0f32; k + 1];
    boundaries[0] = f32::NEG_INFINITY;
    boundaries[k] = f32::INFINITY;
    for i in 0..(k - 1) {
        boundaries[i + 1] = 0.5 * (cent[i] + cent[i + 1]);
    }

    let vals_per_byte = if bits == 1 { 8usize } else { 2usize };
    let packed_sq_d = (d + vals_per_byte - 1) / vals_per_byte;
    let packed_qjl_d = (d + 8 - 1) / 8;

    let mut sq_codes = vec![0u8; n * packed_sq_d];
    let mut qjl_signs = vec![0u8; n * packed_qjl_d];
    let mut res_norms = vec![0.0f32; n];

    #[inline(always)]
    fn quant_bin(v: f32, boundaries: &[f32]) -> usize {
        let mut lo = 0usize;
        let mut hi = boundaries.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if v >= boundaries[mid] { lo = mid; } else { hi = mid; }
        }
        lo
    }

    {
        sq_codes
            .par_chunks_mut(packed_sq_d)
            .zip(qjl_signs.par_chunks_mut(packed_qjl_d))
            .zip(res_norms.par_iter_mut())
            .enumerate()
            .for_each(|(row, ((sq_row, qjl_row), rn_out))| {
                for b in sq_row.iter_mut() { *b = 0; }
                for b in qjl_row.iter_mut() { *b = 0; }

                let mut sum_sq = 0.0f32;
                for j in 0..d {
                    let v = x[[row, j]];
                    let qi = quant_bin(v, &boundaries);
                    let xhat = cent[qi];
                    let r = v - xhat;
                    sum_sq += r * r;

                    if bits == 1 {
                        let byte = j >> 3;
                        let bit = j & 7;
                        sq_row[byte] |= ((qi as u8) & 1) << bit;
                    } else {
                        let byte = j >> 1;
                        let shift = (j & 1) * 3;
                        sq_row[byte] |= (qi as u8) << shift;
                    }

                    let s = (r > 0.0) as u8;
                    let qbyte = j >> 3;
                    let qbit = j & 7;
                    qjl_row[qbyte] |= (s & 1) << qbit;
                }
                *rn_out = sum_sq.sqrt();
            });
    });

    let sq_arr = PyArray1::from_vec_bound(py, sq_codes).reshape([n, packed_sq_d])?;
    let qjl_arr = PyArray1::from_vec_bound(py, qjl_signs).reshape([n, packed_qjl_d])?;
    let rn_arr = PyArray1::from_vec_bound(py, res_norms);
    Ok((sq_arr.unbind(), qjl_arr.unbind(), rn_arr.unbind()))
}

pub fn tq_scan(
    
    query: &[f32],
    sq_codes: &[u8],
    centroids: &[f32],
    norms: &[f32],
    qjl_signs: &[u8],
    res_norms: &[f32],
    qjl_query: &[f32],
    qjl_scale: f32,
    dim: i32,
    mse_bits: i32,
) -> Vec<f32> {
    let d = dim as usize;
    let q_sl = query;
    let cent_sl = centroids;
    let norms_sl = norms;
    let res_norms_sl = res_norms;
    let qjl_q_sl = qjl_query;
    let sq_view = sq_codes;
    let n = sq_view.shape()[0];
    let qjl_view = qjl_signs;
    let qjl_dim = qjl_q_sl.len();
    
    let mut output = vec![0.0f32; n];
    let sq_flat = sq_view;
    let signs_flat = qjl_view;
    
    {
        output.par_chunks_mut(8192).enumerate().for_each(|(chunk_idx, chunk)| {
            let start_idx = chunk_idx * 8192;
            use std::arch::x86_64::*;
            unsafe {
                let v_bit_indices = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
                let v_one = _mm256_set1_ps(1.0);
                let v_neg_one = _mm256_set1_ps(-1.0);
                
                for (sub_idx, score) in chunk.iter_mut().enumerate() {
                    let i = start_idx + sub_idx;
                    if i >= n { break; }

                    let sq_lut_size = if mse_bits == 1 { 256 } else { 64 };
                    let sq_stride = if mse_bits == 1 { 8 } else { 2 };
                    let packed_sq_d = d / sq_stride;
                    let packed_qjl_d = qjl_dim / 8;
                    let rsq = i * packed_sq_d;
                    let rqj = i * (qjl_dim / 8);
                    
                    let mut current_dot: f32 = 0.0;
                    if mse_bits == 1 {
                        let mut v_acc = _mm256_setzero_ps();
                        let v_pos = _mm256_set1_ps(cent_sl[1]);
                        let v_neg = _mm256_set1_ps(cent_sl[0]);
                        for k in (0..d).step_by(8) {
                            let b = sq_flat[rsq + k / 8];
                            let v_b = _mm256_set1_epi32(b as i32);
                            let v_mask = _mm256_cmpeq_epi32(_mm256_and_si256(v_b, v_bit_indices), v_bit_indices);
                            let v_q = _mm256_loadu_ps(q_sl.as_ptr().add(k));
                            let v_k = _mm256_blendv_ps(v_neg, v_pos, _mm256_castsi256_ps(v_mask));
                            v_acc = _mm256_fmadd_ps(v_q, v_k, v_acc);
                        }
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                        current_dot = tmp.iter().sum::<f32>();
                    } else {
                        let mut v_acc = _mm256_setzero_ps();
                        for k in (0..d).step_by(8) {
                            let b0 = sq_flat[rsq + k / 2];
                            let b1 = sq_flat[rsq + k / 2 + 1];
                            let b2 = sq_flat[rsq + k / 2 + 2];
                            let b3 = sq_flat[rsq + k / 2 + 3];
                            let idxs = [(b0 & 7) as i32, ((b0 >> 3) & 7) as i32, (b1 & 7) as i32, ((b1 >> 3) & 7) as i32, (b2 & 7) as i32, ((b2 >> 3) & 7) as i32, (b3 & 7) as i32, ((b3 >> 3) & 7) as i32];
                            let v_idxs = _mm256_loadu_si256(idxs.as_ptr() as *const __m256i);
                            let v_k = _mm256_i32gather_ps(cent_sl.as_ptr(), v_idxs, 4);
                            let v_q = _mm256_loadu_ps(q_sl.as_ptr().add(k));
                            v_acc = _mm256_fmadd_ps(v_q, v_k, v_acc);
                        }
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                        current_dot = tmp.iter().sum::<f32>();
                    }

                    let mut v_sum = _mm256_setzero_ps();
                    for k in (0..qjl_dim).step_by(8) {
                        let b = signs_flat[rqj + k / 8];
                        let v_b = _mm256_set1_epi32(b as i32);
                        let v_mask = _mm256_cmpeq_epi32(_mm256_and_si256(v_b, v_bit_indices), v_bit_indices);
                        let v_q = _mm256_loadu_ps(qjl_q_sl.as_ptr().add(k));
                        let v_sign = _mm256_blendv_ps(v_neg_one, v_one, _mm256_castsi256_ps(v_mask));
                        v_sum = _mm256_fmadd_ps(v_q, v_sign, v_sum);
                    }
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), v_sum);
                    let qjl_corr = tmp.iter().sum::<f32>() * qjl_scale * res_norms_sl[i];
                    *score = (current_dot + qjl_corr) * norms_sl[i];
                }
            }
        });
    });
    output
}

pub fn tq_batch_scan(
    
    queries: &[f32],
    sq_codes: &[u8],
    centroids: &[f32],
    norms: &[f32],
    qjl_signs: &[u8],
    res_norms: &[f32],
    qjl_queries: &[f32],
    qjl_scale: f32,
    dim: i32,
    mse_bits: i32,
) -> Vec<f32> {
    let d = dim as usize;
    let queries_view = queries;
    let num_queries = queries_view.shape()[0];
    let queries_flat = queries_view;
    let qjl_q_view = qjl_queries;
    let cent_sl = centroids;
    let norms_sl = norms;
    let res_norms_sl = res_norms;
    let sq_view = sq_codes;
    let n = sq_view.shape()[0];
    let qjl_view = qjl_signs;
    let qjl_dim = qjl_q_view.shape()[1];
    
    let mut output = vec![0.0f32; num_queries * n];
    let sq_flat = sq_view;
    let signs_flat = qjl_view;
    let queries_flat = queries_view;
    let qjl_queries_flat = qjl_q_view;

    {
        output.par_chunks_mut(n).enumerate().for_each(|(q_idx, scores)| {
            let q_sl = &qjl_queries_flat[q_idx * d .. (q_idx + 1) * d];
            let qjl_q_sl = &qjl_queries_flat[q_idx * qjl_dim .. (q_idx + 1) * qjl_dim];

            use std::arch::x86_64::*;
            unsafe {
                let v_bit_indices = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
                let v_one = _mm256_set1_ps(1.0);
                let v_neg_one = _mm256_set1_ps(-1.0);
                for i in 0..n {

    let sq_lut_size = if mse_bits == 1 { 256 } else { 64 };
    let sq_stride = if mse_bits == 1 { 8 } else { 2 };
    let packed_sq_d = d / sq_stride;
    let packed_qjl_d = qjl_dim / 8;
                    let rsq = i * packed_sq_d;
                    let rqj = i * (qjl_dim / 8);
                    let current_dot: f32;
                    if mse_bits == 1 {
                        let mut v_acc = _mm256_setzero_ps();
                        let v_pos = _mm256_set1_ps(cent_sl[1]);
                        let v_neg = _mm256_set1_ps(cent_sl[0]);
                        for k in (0..d).step_by(8) {
                            let b = sq_flat[rsq + k / 8];
                            let v_b = _mm256_set1_epi32(b as i32);
                            let v_mask = _mm256_cmpeq_epi32(_mm256_and_si256(v_b, v_bit_indices), v_bit_indices);
                            v_acc = _mm256_fmadd_ps(_mm256_loadu_ps(q_sl.as_ptr().add(k)), _mm256_blendv_ps(v_neg, v_pos, _mm256_castsi256_ps(v_mask)), v_acc);
                        }
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                        current_dot = tmp.iter().sum::<f32>();
                    } else {
                        let mut v_acc = _mm256_setzero_ps();
                        for k in (0..d).step_by(8) {
                            let b0 = sq_flat[rsq + k / 2];
                            let b1 = sq_flat[rsq + k / 2 + 1];
                            let b2 = sq_flat[rsq + k / 2 + 2];
                            let b3 = sq_flat[rsq + k / 2 + 3];
                            let idxs = [(b0 & 7) as i32, ((b0 >> 3) & 7) as i32, (b1 & 7) as i32, ((b1 >> 3) & 7) as i32, (b2 & 7) as i32, ((b2 >> 3) & 7) as i32, (b3 & 7) as i32, ((b3 >> 3) & 7) as i32];
                            let v_idxs = _mm256_loadu_si256(idxs.as_ptr() as *const __m256i);
                            v_acc = _mm256_fmadd_ps(_mm256_loadu_ps(q_sl.as_ptr().add(k)), _mm256_i32gather_ps(cent_sl.as_ptr(), v_idxs, 4), v_acc);
                        }
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                        current_dot = tmp.iter().sum::<f32>();
                    }
                    let mut v_sum = _mm256_setzero_ps();
                    for k in (0..qjl_dim).step_by(8) {
                        let b = signs_flat[rqj + k / 8];
                        let v_b = _mm256_set1_epi32(b as i32);
                        let v_mask = _mm256_cmpeq_epi32(_mm256_and_si256(v_b, v_bit_indices), v_bit_indices);
                        v_sum = _mm256_fmadd_ps(_mm256_loadu_ps(qjl_q_sl.as_ptr().add(k)), _mm256_blendv_ps(v_neg_one, v_one, _mm256_castsi256_ps(v_mask)), v_sum);
                    }
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), v_sum);
                    let qjl_corr = tmp.iter().sum::<f32>() * qjl_scale * res_norms_sl[i];
                    scores[i] = (current_dot + qjl_corr) * norms_sl[i];
                }
            }
        });
    });
    output
}

pub fn tq_ivf_online_scan(
    
    queries: &[f32],
    full_sq: &[u8],
    centroids: &[f32],
    full_norms: &[f32],
    full_signs: &[u8],
    full_res: &[f32],
    qjl_queries: &[f32],
    list_offsets: &[i32],
    coarse_centroids: &[f32],
    n_probe: usize,
    qjl_scale: f32,
    dim: i32,
    mse_bits: i32,
    top_k: usize,
) -> (Vec<f32>, Vec<i32>) {
    use std::sync::Mutex;
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    let d = dim as usize;
    let num_levels = if mse_bits == 1 { 2usize } else { 1usize << mse_bits };
    let queries_view = queries;
    let num_queries = queries_view.shape()[0];
    let queries_flat = queries_view;
    let qjl_q_view = qjl_queries;
    let qjl_dim = qjl_q_view.shape()[1];
    let coarse_view = coarse_centroids;
    let num_centroids = coarse_view.shape()[0];
    let offsets_sl = list_offsets;
    let cent_sl = centroids;
    let norms_sl = full_norms;
    let res_sl = full_res;
    let sq_view = full_sq;
    let sq_flat = sq_view;
    let signs_view = full_signs;
    let signs_flat = signs_view;
    let queries_flat = queries_view;
    let qjl_queries_flat = qjl_q_view;
    let coarse_flat = coarse_view;


    let sq_lut_size = if mse_bits == 1 { 256 } else { 64 };
    let sq_stride = if mse_bits == 1 { 8 } else { 2 };
    let packed_sq_d = d / sq_stride;
    let packed_qjl_d = qjl_dim / 8;

    // --- STEP 1: PRE-COMPUTE INDIVIDUAL LUTS FOR THE ENTIRE BATCH ---
    // This is the major optimization: compute cent[l] * query[k] only once per query.
    let mut all_sq_luts = vec![0.0f32; num_queries * d * 8];
    let mut all_qjl_luts = vec![0.0f32; num_queries * (qjl_dim / 4) * 16];

    all_sq_luts.par_chunks_mut(d * 8).enumerate().for_each(|(qi, lut)| {
        let q_sl = &qjl_queries_flat[qi * d .. (qi + 1) * d];
        for k in 0..d {
            let base = k * 8;
            let qv = q_sl[k];
            for b in 0..8 {
                lut[base + b] = qv * cent_sl[b];
            }
        }
    });

    all_qjl_luts.par_chunks_mut((qjl_dim / 4) * 16).enumerate().for_each(|(qi, lut)| {
        let q_rot_sl = &qjl_queries_flat[qi * qjl_dim .. (qi + 1) * qjl_dim];
        for k in 0..(qjl_dim / 4) {
            let base = k * 16;
            for b in 0..16 {
                let mut s = 0.0f32;
                for v in 0..4 {
                    let qv = q_rot_sl[k * 4 + v];
                    let sign = if ((b >> v) & 1) == 1 { 1.0f32 } else { -1.0f32 };
                    s += qv * sign;
                }
                lut[base + b] = s;
            }
        }
    });


    #[inline(always)]
    fn float_to_ordered_u32(f: f32) -> u32 {
        let bits = f.to_bits();
        if bits & 0x80000000 != 0 { !bits } else { bits | 0x80000000 }
    }
    #[inline(always)]
    fn ordered_u32_to_float(u: u32) -> f32 {
        let bits = if u & 0x80000000 == 0 { !u } else { u & 0x7FFFFFFF };
        f32::from_bits(bits)
    }

    let global_heaps: Vec<Mutex<BinaryHeap<Reverse<(u32, u32)>>>> = (0..num_queries)
        .map(|_| Mutex::new(BinaryHeap::with_capacity(top_k)))
        .collect();

    let mut query_to_clusters = vec![vec![0usize; n_probe]; num_queries];
    let mut query_to_cluster_ip = vec![vec![0.0f32; n_probe]; num_queries];

    {
        // Coarse Search: Top-n_probe clusters per query using Matrix Multiplication
        let q_mat = queries_view.to_owned();
        let c_mat = coarse_view.to_owned();
        let scores_mat = q_mat.dot(&c_mat.t());

        query_to_clusters.par_iter_mut().zip(query_to_cluster_ip.par_iter_mut()).enumerate().for_each(|(q_idx, (out, out_ip))| {
            let q_scores = scores_mat.row(q_idx);
            let mut dists: Vec<(f32, usize)> = q_scores.iter().enumerate().map(|(ci, &ip)| (-ip, ci)).collect();
            let actual_probe = n_probe.min(num_centroids);
            dists.select_nth_unstable_by(actual_probe - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            for i in 0..actual_probe {
                out[i] = dists[i].1;
                out_ip[i] = -dists[i].0;
            }
        });

        // Inverted mapping: Clusters -> List of Queries
        let mut cluster_to_queries: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_centroids];
        for q_idx in 0..num_queries {
            for j in 0..query_to_clusters[q_idx].len() {
                cluster_to_queries[query_to_clusters[q_idx][j]].push((q_idx, query_to_cluster_ip[q_idx][j]));
            }
        }

        // --- STEP 2: CLUSTER SCAN WITH INTERLEAVED PRE-COMPUTED LUTS ---
        cluster_to_queries.par_iter().enumerate().for_each(|(c_idx, cluster_queries)| {
            if cluster_queries.is_empty() { return; }
            let start = offsets_sl[c_idx] as usize;
            let end = offsets_sl[c_idx+1] as usize;
            if start >= end { return; }

            let select_threshold = (top_k * 2).max(256);

            for qchunk in cluster_queries.chunks(8) {
                let mut local_buffers: [Vec<(u32, u32)>; 8] = [(); 8].map(|_| Vec::with_capacity(select_threshold));
                let mut thresholds = [f32::MIN; 8];
                let mut lut_sq = vec![0.0f32; d * 8 * 8];
                let mut lut_qjl = vec![0.0f32; (qjl_dim / 4) * 16 * 8];
                let mut centroid_bias = [0.0f32; 8];

                // Fast interleave from pre-computed LUTs
                for (lq, &(qi, ip)) in qchunk.iter().enumerate() {
                    centroid_bias[lq] = ip;
                    let q_sq_lut = &all_sq_luts[qi * d * 8 .. (qi + 1) * d * 8];
                    let q_qjl_lut = &all_qjl_luts[qi * (qjl_dim / 4) * 16 .. (qi + 1) * (qjl_dim / 4) * 16];
                    
                    for k in 0..(d * 8) {
                        lut_sq[k * 8 + lq] = q_sq_lut[k];
                    }
                    for k in 0..((qjl_dim / 4) * 16) {
                        lut_qjl[k * 8 + lq] = q_qjl_lut[k];
                    }
                }

                unsafe {
                    use std::arch::x86_64::*;
                    let v_bias = _mm256_loadu_ps(centroid_bias.as_ptr());
                    
                    for i in start..end {
                        let rsq = i * (d / 2); // Wait, sq_flat size is d/2 if mse_bits=3
                        let rqj = i * (qjl_dim / 8);

                        if i + 4 < end {
                            let pre_rsq = (i + 4) * packed_sq_d;
                            let pre_rqj = (i + 4) * (qjl_dim / 8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq + 64) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq + 128) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq + 192) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(signs_flat.as_ptr().add(pre_rqj) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(signs_flat.as_ptr().add(pre_rqj + 64) as *const i8);
                        }
                        let v_norms = _mm256_set1_ps(norms_sl[i]);
                        let v_res = _mm256_set1_ps(res_sl[i] * qjl_scale);

                        let mut v_sq = _mm256_setzero_ps();
                        let mut v_qjl = _mm256_setzero_ps();
                        for k in 0..(d / 8) {
                            let s_idx = rsq + k * 4;
                            
                            let s0 = sq_flat[s_idx] as usize;
                            v_sq = _mm256_add_ps(v_sq, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8) * 8 + (s0 & 7)) * 8)));
                            v_sq = _mm256_add_ps(v_sq, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 1) * 8 + ((s0 >> 3) & 7)) * 8)));

                            let s1 = sq_flat[s_idx + 1] as usize;
                            v_sq = _mm256_add_ps(v_sq, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 2) * 8 + (s1 & 7)) * 8)));
                            v_sq = _mm256_add_ps(v_sq, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 3) * 8 + ((s1 >> 3) & 7)) * 8)));

                            let s2 = sq_flat[s_idx + 2] as usize;
                            v_sq = _mm256_add_ps(v_sq, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 4) * 8 + (s2 & 7)) * 8)));
                            v_sq = _mm256_add_ps(v_sq, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 5) * 8 + ((s2 >> 3) & 7)) * 8)));

                            let s3 = sq_flat[s_idx + 3] as usize;
                            v_sq = _mm256_add_ps(v_sq, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 6) * 8 + (s3 & 7)) * 8)));
                            v_sq = _mm256_add_ps(v_sq, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 7) * 8 + ((s3 >> 3) & 7)) * 8)));

                            let b = signs_flat[rqj + k] as usize;
                            let b0 = b & 15;
                            let b1 = b >> 4;
                            v_qjl = _mm256_add_ps(v_qjl, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2) * 16 + b0) * 8)));
                            v_qjl = _mm256_add_ps(v_qjl, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2 + 1) * 16 + b1) * 8)));
                        }

                        let vf = _mm256_add_ps(_mm256_mul_ps(_mm256_fmadd_ps(v_qjl, v_res, v_sq), v_norms), v_bias);
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), vf);

                        for (lq, _q) in qchunk.iter().enumerate() {
                            let s = tmp[lq];
                            if s <= thresholds[lq] { continue; }
                            let score_bits = float_to_ordered_u32(s);
                            local_buffers[lq].push((score_bits, i as u32));
                            if local_buffers[lq].len() >= select_threshold {
                                local_buffers[lq].select_nth_unstable_by(top_k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
                                local_buffers[lq].truncate(top_k);
                                thresholds[lq] = ordered_u32_to_float(local_buffers[lq][top_k - 1].0);
                            }
                        }
                    }
                }

                // Batch merge into global heaps
                for (lq, &(qi, _ip)) in qchunk.iter().enumerate() {
                    let mut g_h = global_heaps[qi].lock().unwrap();
                    for &(s, id) in local_buffers[lq].iter() {
                        if g_h.len() < top_k { g_h.push(Reverse((s, id))); }
                        else if s > g_h.peek().unwrap().0.0 { g_h.pop(); g_h.push(Reverse((s, id))); }
                    }
                }
            }
        });
    });

    let mut fs = vec![0.0f32; num_queries * top_k];
    let mut fi = vec![0i32; num_queries * top_k];
    for qi in 0..num_queries {
        let mut h = global_heaps[qi].lock().unwrap();
        let mut r = Vec::new();
        while let Some(Reverse(e)) = h.pop() { r.push(e); }
        r.reverse();
        for (k, (s, id)) in r.into_iter().enumerate() {
            if k < top_k { fs[qi * top_k + k] = ordered_u32_to_float(s); fi[qi * top_k + k] = id as i32; }
        }
    }
    (fs, fi)
}


pub fn tq_kmeans_train(
    
    x: &[f32],
    n_list: usize,
    iters: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let x_ro = x.readonly();
    let x_arr = x_ro.as_array();
    let n = x_arr.shape()[0];
    let d = x_arr.shape()[1];
    if n < n_list { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Points < n_list")); }
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.shuffle(&mut rng);
    let mut centroids = vec![0.0f32; n_list * d];
    for i in 0..n_list {
        let x_idx = idxs[i];
        for j in 0..d { centroids[i * d + j] = x_arr[[x_idx, j]]; }
    }
    for cid in 0..n_list {
        let mut nsq = 0.0f32;
        for j in 0..d { nsq += centroids[cid*d+j]*centroids[cid*d+j]; }
        let norm = nsq.sqrt() + 1e-12;
        for j in 0..d { centroids[cid*d+j] /= norm; }
    }
    for _it in 0..iters {
        let mut assignments = vec![0usize; n];
        let cent_view = ndarray::ArrayView2::from_shape((n_list, d), &centroids).unwrap();
        
        {
            let chunk_size = 1024; // Tối ưu hơn cho SIMD mà vẫn an toàn RAM
            assignments.par_chunks_mut(chunk_size).enumerate().for_each(|(c_idx, a_chunk)| {
                let start = c_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let x_chunk = x_arr.slice(ndarray::s![start..end, ..]);
                
                // Dùng dot() của ndarray (có sẵn cache-blocking)
                let scores = x_chunk.dot(&cent_view.t());
                
                a_chunk.iter_mut().enumerate().for_each(|(i, out)| {
                    let row = scores.row(i);
                    let mut ms = f32::MIN;
                    let mut bc = 0usize;
                    for (cid, &s) in row.iter().enumerate() {
                        if s > ms { ms = s; bc = cid; }
                    }
                    *out = bc;
                });
            });
        });
        let mut next = vec![0.0f32; n_list * d];
        let mut counts = vec![0usize; n_list];
        for i in 0..n {
            let cid = assignments[i];
            for j in 0..d { next[cid*d+j] += x_arr[[i, j]]; }
            counts[cid] += 1;
        }
        for cid in 0..n_list {
            let c = counts[cid] as f32;
            if c > 0.0 {
                let mut nsq = 0.0f32;
                for j in 0..d { let v = next[cid*d+j]/c; next[cid*d+j] = v; nsq += v*v; }
                let norm = nsq.sqrt() + 1e-12;
                for j in 0..d { centroids[cid*d+j] = next[cid*d+j]/norm; }
            }
        }
    }
    Ok(PyArray1::from_vec_bound(py, centroids).reshape([n_list, d])?.unbind())
}

pub fn tq_ivf_scan_with_clusters(
    
    queries: &[f32],
    full_sq: &[u8],
    centroids: &[f32],
    full_norms: &[f32],
    full_signs: &[u8],
    full_res: &[f32],
    qjl_queries: &[f32],
    list_offsets: &[i32],
    cluster_ids: &[i32],
    cluster_scores: &[f32],
    qjl_scale: f32,
    dim: i32,
    mse_bits: i32,
    top_k: usize,
) -> (Vec<f32>, Vec<i32>) {
    use std::sync::Mutex;
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    let d = dim as usize;
    let num_levels = if mse_bits == 1 { 2usize } else { 1usize << mse_bits };
    
    let queries_view = queries;
    let num_queries = queries_view.shape()[0];
    let queries_flat = queries_view;
    
    let qjl_q_view = qjl_queries;
    let qjl_dim = qjl_q_view.shape()[1];
    
    let offsets_sl = list_offsets;
    
    let cent_sl = centroids;
    
    let norms_sl = full_norms;
    
    let res_sl = full_res;
    
    let sq_view = full_sq;
    let sq_flat = sq_view;
    
    let signs_view = full_signs;
    let signs_flat = signs_view;
    
    let qjl_queries_flat = qjl_q_view;
    
    let cluster_ids_view = cluster_ids;
    let cluster_scores_view = cluster_scores;
    let n_probe = cluster_ids_view.shape()[1];


    let sq_lut_size = if mse_bits == 1 { 256 } else { 64 };
    let sq_stride = if mse_bits == 1 { 8 } else { 2 };
    let packed_sq_d = d / sq_stride;
    let packed_qjl_d = qjl_dim / 8;

    // LUTs pre-computation
    let mut all_sq_luts = vec![0.0f32; num_queries * d * 8];
    let mut all_qjl_luts = vec![0.0f32; num_queries * (qjl_dim / 4) * 16];

    all_sq_luts.par_chunks_mut(d * 8).enumerate().for_each(|(qi, lut)| {
        let q_sl = &qjl_queries_flat[qi * d .. (qi + 1) * d];
        for k in 0..d {
            let base = k * 8;
            let qv = q_sl[k];
            for b in 0..num_levels.min(8) {
                lut[base + b] = qv * cent_sl[b];
            }
        }
    });

    all_qjl_luts.par_chunks_mut((qjl_dim / 4) * 16).enumerate().for_each(|(qi, lut)| {
        let q_rot_sl = &qjl_queries_flat[qi * qjl_dim .. (qi + 1) * qjl_dim];
        for k in 0..(qjl_dim / 4) {
            let base = k * 16;
            for b in 0..16 {
                let mut s = 0.0f32;
                for v in 0..4 {
                    let qv = q_rot_sl[k * 4 + v];
                    let sign = if ((b >> v) & 1) == 1 { 1.0f32 } else { -1.0f32 };
                    s += qv * sign;
                }
                lut[base + b] = s;
            }
        }
    });


    #[inline(always)]
    fn float_to_ordered_u32(f: f32) -> u32 {
        let bits = f.to_bits();
        if bits & 0x80000000 != 0 { !bits } else { bits | 0x80000000 }
    }
    #[inline(always)]
    fn ordered_u32_to_float(u: u32) -> f32 {
        let bits = if u & 0x80000000 == 0 { !u } else { u & 0x7FFFFFFF };
        f32::from_bits(bits)
    }

    let global_heaps: Vec<Mutex<BinaryHeap<Reverse<(u32, u32)>>>> = (0..num_queries)
        .map(|_| Mutex::new(BinaryHeap::with_capacity(top_k)))
        .collect();

    // Map Cluster -> Queries
    let num_centroids = offsets_sl.len() - 1;
    let mut cluster_to_queries: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_centroids];
    for q_idx in 0..num_queries {
        for j in 0..n_probe {
            let cid = cluster_ids_view[[q_idx, j]];
            let score = cluster_scores_view[[q_idx, j]];
            if cid >= 0 && (cid as usize) < num_centroids {
                cluster_to_queries[cid as usize].push((q_idx, score));
            }
        }
    }

    {
        cluster_to_queries.par_iter().enumerate().for_each(|(c_idx, cluster_queries)| {
            if cluster_queries.is_empty() { return; }
            let start = offsets_sl[c_idx] as usize;
            let end = offsets_sl[c_idx+1] as usize;
            if start >= end { return; }

            let select_threshold = (top_k * 2).max(256);
            let mut local_buffers: [Vec<(u32, u32)>; 8] = [(); 8].map(|_| Vec::with_capacity(select_threshold));
                let mut thresholds = [f32::MIN; 8];
            let mut lut_sq = vec![0.0f32; d * 8 * 8];
            let mut lut_qjl = vec![0.0f32; (qjl_dim / 4) * 16 * 8];

            for qchunk in cluster_queries.chunks(8) {
                for buf in local_buffers.iter_mut() { buf.clear(); }
                thresholds = [f32::MIN; 8];
                let mut centroid_bias = [0.0f32; 8];

                for (lq, &(qi, score)) in qchunk.iter().enumerate() {
                    centroid_bias[lq] = score;
                    let q_sq_lut = &all_sq_luts[qi * d * 8 .. (qi + 1) * d * 8];
                    let q_qjl_lut = &all_qjl_luts[qi * (qjl_dim / 4) * 16 .. (qi + 1) * (qjl_dim / 4) * 16];
                    for k in 0..(d * 8) { lut_sq[k * 8 + lq] = q_sq_lut[k]; }
                    for k in 0..((qjl_dim / 4) * 16) { lut_qjl[k * 8 + lq] = q_qjl_lut[k]; }
                }

                unsafe {
                    use std::arch::x86_64::*;
                    let v_bias = _mm256_loadu_ps(centroid_bias.as_ptr());
                    for i in start..end {
                        let rsq = i * packed_sq_d;
                        let rqj = i * (qjl_dim / 8);
                        if i + 4 < end {
                            let pre_rsq = (i + 4) * packed_sq_d;
                            let pre_rqj = (i + 4) * (qjl_dim / 8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq + 64) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq + 128) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq + 192) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(signs_flat.as_ptr().add(pre_rqj) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(signs_flat.as_ptr().add(pre_rqj + 64) as *const i8);
                        }
                        let v_norms = _mm256_set1_ps(norms_sl[i]);
                        let v_res = _mm256_set1_ps(res_sl[i] * qjl_scale);

                        let mut v_sq0 = _mm256_setzero_ps();
                        let mut v_sq1 = _mm256_setzero_ps();
                        let mut v_sq2 = _mm256_setzero_ps();
                        let mut v_sq3 = _mm256_setzero_ps();
                        let mut v_qjl0 = _mm256_setzero_ps();
                        let mut v_qjl1 = _mm256_setzero_ps();
                        if mse_bits == 1 {
                            for k in 0..(d / 8) {
                                let s0 = sq_flat[rsq + k] as usize;
                                v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8) * 8 + (s0 & 1)) * 8)));
                                v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 1) * 8 + ((s0 >> 1) & 1)) * 8)));
                                v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 2) * 8 + ((s0 >> 2) & 1)) * 8)));
                                v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 3) * 8 + ((s0 >> 3) & 1)) * 8)));
                                v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 4) * 8 + ((s0 >> 4) & 1)) * 8)));
                                v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 5) * 8 + ((s0 >> 5) & 1)) * 8)));
                                v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 6) * 8 + ((s0 >> 6) & 1)) * 8)));
                                v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 7) * 8 + ((s0 >> 7) & 1)) * 8)));

                                let b = signs_flat[rqj + k] as usize;
                                let b0 = b & 15;
                                let b1 = b >> 4;
                                v_qjl0 = _mm256_add_ps(v_qjl0, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2) * 16 + b0) * 8)));
                                v_qjl1 = _mm256_add_ps(v_qjl1, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2 + 1) * 16 + b1) * 8)));
                            }
                        } else {
                            for k in 0..(d / 8) {
                                let s_idx = rsq + k * 4;
                                
                                let s0 = sq_flat[s_idx] as usize;
                                v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8) * 8 + (s0 & 7)) * 8)));
                                v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 1) * 8 + ((s0 >> 3) & 7)) * 8)));

                                let s1 = sq_flat[s_idx + 1] as usize;
                                v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 2) * 8 + (s1 & 7)) * 8)));
                                v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 3) * 8 + ((s1 >> 3) & 7)) * 8)));

                                let s2 = sq_flat[s_idx + 2] as usize;
                                v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 4) * 8 + (s2 & 7)) * 8)));
                                v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 5) * 8 + ((s2 >> 3) & 7)) * 8)));

                                let s3 = sq_flat[s_idx + 3] as usize;
                                v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 6) * 8 + (s3 & 7)) * 8)));
                                v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 7) * 8 + ((s3 >> 3) & 7)) * 8)));

                                let b = signs_flat[rqj + k] as usize;
                                let b0 = b & 15;
                                let b1 = b >> 4;
                                v_qjl0 = _mm256_add_ps(v_qjl0, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2) * 16 + b0) * 8)));
                                v_qjl1 = _mm256_add_ps(v_qjl1, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2 + 1) * 16 + b1) * 8)));
                            }
                        }

                        let v_sq_01 = _mm256_add_ps(v_sq0, v_sq1);
                        let v_sq_23 = _mm256_add_ps(v_sq2, v_sq3);
                        let v_sq = _mm256_add_ps(v_sq_01, v_sq_23);
                        let v_qjl = _mm256_add_ps(v_qjl0, v_qjl1);

                        let vf = _mm256_add_ps(_mm256_mul_ps(_mm256_fmadd_ps(v_qjl, v_res, v_sq), v_norms), v_bias);
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), vf);
                        for (lq, _) in qchunk.iter().enumerate() {
                            let s = tmp[lq];
                            if s <= thresholds[lq] { continue; }
                            let score_bits = float_to_ordered_u32(s);
                            local_buffers[lq].push((score_bits, i as u32));
                            if local_buffers[lq].len() >= select_threshold {
                                local_buffers[lq].select_nth_unstable_by(top_k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
                                local_buffers[lq].truncate(top_k);
                                thresholds[lq] = ordered_u32_to_float(local_buffers[lq][top_k - 1].0);
                            }
                        }
                    }
                }
                for (lq, &(qi, _)) in qchunk.iter().enumerate() {
                    let mut g_h = global_heaps[qi].lock().unwrap();
                    for &(s, id) in local_buffers[lq].iter() {
                        if g_h.len() < top_k { g_h.push(Reverse((s, id))); }
                        else if s > g_h.peek().unwrap().0.0 { g_h.pop(); g_h.push(Reverse((s, id))); }
                    }
                }
            }
        });
    });

    let mut fs = vec![0.0f32; num_queries * top_k];
    let mut fi = vec![0i32; num_queries * top_k];
    for qi in 0..num_queries {
        let mut h = global_heaps[qi].lock().unwrap();
        let mut r = Vec::new();
        while let Some(Reverse(e)) = h.pop() { r.push(e); }
        r.reverse();
        for (k, (s, id)) in r.into_iter().enumerate() {
            if k < top_k { fs[qi * top_k + k] = ordered_u32_to_float(s); fi[qi * top_k + k] = id as i32; }
        }
    }
    (fs, fi)
}

pub fn tq_assign_clusters(
    
    x: &[f32],
    centroids: &[f32],
) -> PyResult<Py<PyArray1<i32>>> {
    let x_ro = x.readonly();
    let x_arr = x_ro.as_array();
    let c_ro = centroids.readonly();
    let c_arr = c_ro.as_array();
    let n = x_arr.shape()[0];
    let d = x_arr.shape()[1];
    let mut ass = vec![0i32; n];
    
    {
        let chunk_size = 1024;
        ass.par_chunks_mut(chunk_size).enumerate().for_each(|(c_idx, a_chunk)| {
            let start = c_idx * chunk_size;
            let end = (start + chunk_size).min(n);
            let x_chunk = x_arr.slice(ndarray::s![start..end, ..]);
            
            let scores = x_chunk.dot(&c_arr.t());
            
            a_chunk.iter_mut().enumerate().for_each(|(i, out)| {
                let row = scores.row(i);
                let mut ms = f32::MIN;
                let mut bc = 0usize;
                for (cid, &s) in row.iter().enumerate() {
                    if s > ms { ms = s; bc = cid; }
                }
                *out = bc as i32;
            });
        });
    });
    Ok(PyArray1::from_vec_bound(py, ass).unbind())
}

pub fn tq_unified_search(
    
    queries: &[f32],
    rot_op: &[f32],
    coarse_centroids: &[f32],
    list_offsets: &[i32],
    vector_ids: &[i64],
    full_sq: &[u8],
    centroids: &[f32],
    full_norms: &[f32],
    full_signs: &[u8],
    full_res: &[f32],
    qjl_scale: f32,
    dim: i32,
    mse_bits: i32,
    n_probe: usize,
    top_k: usize,
    allowed_ids: Option<&[i64]>,
    raw_corpus_32: Option<&[f32]>,
    raw_corpus_16: Option<&[u16]>,
    rerank_factor: Option<usize>,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<i64>>)> {
    use std::sync::Mutex;
    use std::collections::{BinaryHeap, HashSet};
    use std::cmp::Reverse;

    let d = dim as usize;
    let num_levels = if mse_bits == 1 { 2usize } else { 1usize << mse_bits };
    
    let queries_view = queries;
    let num_queries = queries_view.shape()[0];
    let queries_flat = queries_view;
    
    let rot_view = rot_op;
    
    // 1. Query rotation in Rust: q_rot = queries.dot(&rot_op)
    let q_rot = queries_view.dot(&rot_view);
    let qjl_queries_flat = q_rot;
    let qjl_dim = q_rot.shape()[1];
    
    let coarse_view = coarse_centroids;
    let num_centroids = coarse_view.shape()[0];
    
    let offsets_sl = list_offsets;
    
    let vector_ids_sl = vector_ids;
    
    let cent_sl = centroids;
    
    let norms_sl = full_norms;
    
    let res_sl = full_res;
    
    let sq_view = full_sq;
    let sq_flat = sq_view;
    
    let signs_view = full_signs;
    let signs_flat = signs_view;
    

    let sq_lut_size = if mse_bits == 1 { 256 } else { 64 };
    let sq_stride = if mse_bits == 1 { 8 } else { 2 };
    let packed_sq_d = d / sq_stride;
    let packed_qjl_d = qjl_dim / 8;

    // Set scan target K (increase if reranking is active)
    let scan_top_k = if (raw_corpus_32.is_some() || raw_corpus_16.is_some()) && rerank_factor.is_some() {
        top_k * rerank_factor.unwrap()
    } else {
        top_k
    };

    // LUTs pre-computation
    let mut all_sq_luts = vec![0.0f32; num_queries * d * 8];
    let mut all_qjl_luts = vec![0.0f32; num_queries * (qjl_dim / 4) * 16];

    all_sq_luts.par_chunks_mut(d * 8).enumerate().for_each(|(qi, lut)| {
        let q_sl = &qjl_queries_flat[qi * d .. (qi + 1) * d];
        for k in 0..d {
            let base = k * 8;
            let qv = q_sl[k];
            for b in 0..num_levels.min(8) {
                lut[base + b] = qv * cent_sl[b];
            }
        }
    });

    all_qjl_luts.par_chunks_mut((qjl_dim / 4) * 16).enumerate().for_each(|(qi, lut)| {
        let q_rot_sl = &qjl_queries_flat[qi * qjl_dim .. (qi + 1) * qjl_dim];
        for k in 0..(qjl_dim / 4) {
            let base = k * 16;
            for b in 0..16 {
                let mut s = 0.0f32;
                for v in 0..4 {
                    let qv = q_rot_sl[k * 4 + v];
                    let sign = if ((b >> v) & 1) == 1 { 1.0f32 } else { -1.0f32 };
                    s += qv * sign;
                }
                lut[base + b] = s;
            }
        }
    });


    #[inline(always)]
    fn float_to_ordered_u32(f: f32) -> u32 {
        let bits = f.to_bits();
        if bits & 0x80000000 != 0 { !bits } else { bits | 0x80000000 }
    }
    #[inline(always)]
    fn ordered_u32_to_float(u: u32) -> f32 {
        let bits = if u & 0x80000000 == 0 { !u } else { u & 0x7FFFFFFF };
        f32::from_bits(bits)
    }

    let global_heaps: Vec<Mutex<BinaryHeap<Reverse<(u32, u32)>>>> = (0..num_queries)
        .map(|_| Mutex::new(BinaryHeap::with_capacity(scan_top_k)))
        .collect();

    // Allowed IDs HashSet
    let allowed_set: Option<HashSet<i64>> = allowed_ids.map(|arr| {
        let ro = arr.readonly();
        let sl = ro;
        sl.iter().cloned().collect()
    });

    {
        // 2. Coarse Search: queries.dot(&coarse_centroids.t())
        let scores_c = queries_view.dot(&coarse_view.t());
        
        let mut query_to_clusters = vec![vec![0usize; n_probe]; num_queries];
        let mut query_to_cluster_ip = vec![vec![0.0f32; n_probe]; num_queries];

        query_to_clusters.par_iter_mut().zip(query_to_cluster_ip.par_iter_mut()).enumerate().for_each(|(q_idx, (out, out_ip))| {
            let q_scores = scores_c.row(q_idx);
            let mut dists: Vec<(f32, usize)> = q_scores.iter().enumerate().map(|(ci, &ip)| (-ip, ci)).collect();
            let actual_probe = n_probe.min(num_centroids);
            dists.select_nth_unstable_by(actual_probe - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            for i in 0..actual_probe {
                out[i] = dists[i].1;
                out_ip[i] = -dists[i].0;
            }
        });

        // Map Cluster -> Queries
        let mut cluster_to_queries: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_centroids];
        for q_idx in 0..num_queries {
            for j in 0..n_probe {
                let cid = query_to_clusters[q_idx][j];
                let score = query_to_cluster_ip[q_idx][j];
                cluster_to_queries[cid].push((q_idx, score));
            }
        }

        // 3. Cluster Scan
        let active_clusters: Vec<(usize, &Vec<(usize, f32)>)> = cluster_to_queries
            .iter()
            .enumerate()
            .filter(|&(c_idx, cluster_queries)| {
                if cluster_queries.is_empty() { return false; }
                let start = offsets_sl[c_idx] as usize;
                let end = offsets_sl[c_idx+1] as usize;
                start < end
            })
            .collect();

        active_clusters.par_iter().for_each(|&(c_idx, cluster_queries)| {
            let start = offsets_sl[c_idx] as usize;
            let end = offsets_sl[c_idx+1] as usize;

            let select_threshold = (scan_top_k * 2).max(256);
            let mut local_buffers: [Vec<(u32, u32)>; 8] = [(); 8].map(|_| Vec::with_capacity(select_threshold));
                let mut thresholds = [f32::MIN; 8];
            let mut lut_sq = vec![0.0f32; d * 8 * 8];
            let mut lut_qjl = vec![0.0f32; (qjl_dim / 4) * 16 * 8];

            for qchunk in cluster_queries.chunks(8) {
                for buf in local_buffers.iter_mut() { buf.clear(); }
                thresholds = [f32::MIN; 8];
                let mut centroid_bias = [0.0f32; 8];

                for (lq, &(qi, score)) in qchunk.iter().enumerate() {
                    centroid_bias[lq] = score;
                    let q_sq_lut = &all_sq_luts[qi * d * 8 .. (qi + 1) * d * 8];
                    let q_qjl_lut = &all_qjl_luts[qi * (qjl_dim / 4) * 16 .. (qi + 1) * (qjl_dim / 4) * 16];
                    for k in 0..(d * 8) { lut_sq[k * 8 + lq] = q_sq_lut[k]; }
                    for k in 0..((qjl_dim / 4) * 16) { lut_qjl[k * 8 + lq] = q_qjl_lut[k]; }
                }

                unsafe {
                    use std::arch::x86_64::*;
                    let v_bias = _mm256_loadu_ps(centroid_bias.as_ptr());
                    
                    for i in start..end {
                        // FILTER: allowed_ids check
                        let global_id = vector_ids_sl[i];
                        if let Some(ref set) = allowed_set {
                            if !set.contains(&global_id) {
                                continue;
                            }
                        }

                        let rsq = i * packed_sq_d;
                        let rqj = i * (qjl_dim / 8);

                        if i + 4 < end {
                            let pre_rsq = (i + 4) * packed_sq_d;
                            let pre_rqj = (i + 4) * (qjl_dim / 8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq + 64) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq + 128) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(sq_flat.as_ptr().add(pre_rsq + 192) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(signs_flat.as_ptr().add(pre_rqj) as *const i8);
                            _mm_prefetch::<_MM_HINT_T0>(signs_flat.as_ptr().add(pre_rqj + 64) as *const i8);
                        }
                        let v_norms = _mm256_set1_ps(norms_sl[i]);
                        let v_res = _mm256_set1_ps(res_sl[i] * qjl_scale);

                        let mut v_sq0 = _mm256_setzero_ps();
                        let mut v_sq1 = _mm256_setzero_ps();
                        let mut v_sq2 = _mm256_setzero_ps();
                        let mut v_sq3 = _mm256_setzero_ps();
                        let mut v_qjl0 = _mm256_setzero_ps();
                        let mut v_qjl1 = _mm256_setzero_ps();
                        if mse_bits == 1 {
                            for k in 0..(d / 8) {
                                let s0 = sq_flat[rsq + k] as usize;
                                v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8) * 8 + (s0 & 1)) * 8)));
                                v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 1) * 8 + ((s0 >> 1) & 1)) * 8)));
                                v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 2) * 8 + ((s0 >> 2) & 1)) * 8)));
                                v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 3) * 8 + ((s0 >> 3) & 1)) * 8)));
                                v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 4) * 8 + ((s0 >> 4) & 1)) * 8)));
                                v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 5) * 8 + ((s0 >> 5) & 1)) * 8)));
                                v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 6) * 8 + ((s0 >> 6) & 1)) * 8)));
                                v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 7) * 8 + ((s0 >> 7) & 1)) * 8)));

                                let b = signs_flat[rqj + k] as usize;
                                let b0 = b & 15;
                                let b1 = b >> 4;
                                v_qjl0 = _mm256_add_ps(v_qjl0, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2) * 16 + b0) * 8)));
                                v_qjl1 = _mm256_add_ps(v_qjl1, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2 + 1) * 16 + b1) * 8)));
                            }
                        } else {
                            for k in 0..(d / 8) {
                                let s_idx = rsq + k * 4;
                                
                                let s0 = sq_flat[s_idx] as usize;
                                v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8) * 8 + (s0 & 7)) * 8)));
                                v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 1) * 8 + ((s0 >> 3) & 7)) * 8)));

                                let s1 = sq_flat[s_idx + 1] as usize;
                                v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 2) * 8 + (s1 & 7)) * 8)));
                                v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 3) * 8 + ((s1 >> 3) & 7)) * 8)));

                                let s2 = sq_flat[s_idx + 2] as usize;
                                v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 4) * 8 + (s2 & 7)) * 8)));
                                v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 5) * 8 + ((s2 >> 3) & 7)) * 8)));

                                let s3 = sq_flat[s_idx + 3] as usize;
                                v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 6) * 8 + (s3 & 7)) * 8)));
                                v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(lut_sq.as_ptr().add(((k * 8 + 7) * 8 + ((s3 >> 3) & 7)) * 8)));

                                let b = signs_flat[rqj + k] as usize;
                                let b0 = b & 15;
                                let b1 = b >> 4;
                                v_qjl0 = _mm256_add_ps(v_qjl0, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2) * 16 + b0) * 8)));
                                v_qjl1 = _mm256_add_ps(v_qjl1, _mm256_loadu_ps(lut_qjl.as_ptr().add(((k * 2 + 1) * 16 + b1) * 8)));
                            }
                        }

                        let v_sq_01 = _mm256_add_ps(v_sq0, v_sq1);
                        let v_sq_23 = _mm256_add_ps(v_sq2, v_sq3);
                        let v_sq = _mm256_add_ps(v_sq_01, v_sq_23);
                        let v_qjl = _mm256_add_ps(v_qjl0, v_qjl1);

                        let vf = _mm256_add_ps(_mm256_mul_ps(_mm256_fmadd_ps(v_qjl, v_res, v_sq), v_norms), v_bias);
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), vf);
                        for (lq, _) in qchunk.iter().enumerate() {
                            let s = tmp[lq];
                            if s <= thresholds[lq] { continue; }
                            let score_bits = float_to_ordered_u32(s);
                            local_buffers[lq].push((score_bits, i as u32));
                            if local_buffers[lq].len() >= select_threshold {
                                local_buffers[lq].select_nth_unstable_by(scan_top_k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
                                local_buffers[lq].truncate(scan_top_k);
                                thresholds[lq] = ordered_u32_to_float(local_buffers[lq][scan_top_k - 1].0);
                            }
                        }
                    }
                }
                for (lq, &(qi, _)) in qchunk.iter().enumerate() {
                    let mut g_h = global_heaps[qi].lock().unwrap();
                    for &(s, id) in local_buffers[lq].iter() {
                        if g_h.len() < scan_top_k { g_h.push(Reverse((s, id))); }
                        else if s > g_h.peek().unwrap().0.0 { g_h.pop(); g_h.push(Reverse((s, id))); }
                    }
                }
            }
        });
    });

   // 4. Extract Top-K and optionally perform f32/f16 Reranking
    let mut fs = vec![0.0f32; num_queries * top_k];
    let mut fi = vec![-1i64; num_queries * top_k];

    // Đọc map memory của 2 loại
    let raw_f32_opt = raw_corpus_32.as_ref().map(|x| x.readonly());
    let raw_f32_flat = raw_f32_opt.as_ref().map(|x| x);
    
    let raw_f16_opt = raw_corpus_16.as_ref().map(|x| x.readonly());
    let raw_f16_flat = raw_f16_opt.as_ref().map(|x| x);

    for qi in 0..num_queries {
        let mut h = global_heaps[qi].lock().unwrap();
        let mut r = Vec::new();
        while let Some(Reverse(e)) = h.pop() { r.push(e); }
        r.reverse();

        let mut final_candidates = Vec::new();
        let q_raw_sl = &queries_flat[qi * d .. (qi + 1) * d];

        // LUỒNG 1: XỬ LÝ F32 BẰNG AVX2 GỐC
        if let Some(raw_flat) = raw_f32_flat {
            for (_approx_s, id) in r {
                let global_id = vector_ids_sl[id as usize];
                let raw_vec = &raw_flat[global_id as usize * d .. (global_id as usize + 1) * d];
                
                let mut exact_dot = 0.0f32;
                unsafe {
                    use std::arch::x86_64::*;
                    let mut v_acc = _mm256_setzero_ps();
                    let mut k = 0;
                    while k + 8 <= d {
                        let v_q = _mm256_loadu_ps(q_raw_sl.as_ptr().add(k));
                        let v_r = _mm256_loadu_ps(raw_vec.as_ptr().add(k));
                        v_acc = _mm256_fmadd_ps(v_q, v_r, v_acc);
                        k += 8;
                    }
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                    exact_dot = tmp.iter().sum::<f32>();
                    for rk in k..d { exact_dot += q_raw_sl[rk] * raw_vec[rk]; }
                }
                final_candidates.push((exact_dot, global_id));
            }
            final_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        } 
        // LUỒNG 2: XỬ LÝ F16 SIÊU TỐC VỚI TẬP LỆNH F16C
        else if let Some(raw_flat_f16) = raw_f16_flat {
            for (_approx_s, id) in r {
                let global_id = vector_ids_sl[id as usize];
                let raw_vec_f16 = &raw_flat_f16[global_id as usize * d .. (global_id as usize + 1) * d];
                
                let mut exact_dot = 0.0f32;
                unsafe {
                    use std::arch::x86_64::*;
                    let mut v_acc = _mm256_setzero_ps();
                    let mut k = 0;
                    while k + 8 <= d {
                        let v_q = _mm256_loadu_ps(q_raw_sl.as_ptr().add(k));
                        let v_r_128 = _mm_loadu_si128(raw_vec_f16.as_ptr().add(k) as *const __m128i);
                        let v_r = _mm256_cvtph_ps(v_r_128);
                        v_acc = _mm256_fmadd_ps(v_q, v_r, v_acc);
                        k += 8;
                    }
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                    exact_dot = tmp.iter().sum::<f32>();
                    
                    // Sửa lỗi: Dùng mảng u16 thay vì _mm_insert_epi16 để tương thích hoàn toàn với Rust
                    for rk in k..d {
                        let mut tmp_arr = [0u16; 8];
                        tmp_arr[0] = raw_vec_f16[rk];
                        let v_128 = _mm_loadu_si128(tmp_arr.as_ptr() as *const __m128i);
                        let v_f32 = _mm256_cvtph_ps(v_128);
                        let mut out = [0.0f32; 8];
                        _mm256_storeu_ps(out.as_mut_ptr(), v_f32);
                        exact_dot += q_raw_sl[rk] * out[0];
                    }
                }
                final_candidates.push((exact_dot, global_id));
            }
            final_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        }
        // LUỒNG 3: KHÔNG RERANK
        else {
            for (approx_s, id) in r {
                let global_id = vector_ids_sl[id as usize];
                final_candidates.push((ordered_u32_to_float(approx_s), global_id));
            }
        }

        // Cắt lấy Top-K cuối cùng
        final_candidates.truncate(top_k);
        for (k, &(s, id)) in final_candidates.iter().enumerate() {
            fs[qi * top_k + k] = s;
            fi[qi * top_k + k] = id;
        }
    }

    (fs, fi)
}

