use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// =============================================================================
// Quantization helpers (index-time)
// - SQ bits supported: 1 or 3 (TurboQuant 2b/4b)
// =============================================================================

pub fn tq_quantize_rotated(x_rot: &[f32], n: usize, d: usize, sq_centroids: &[f32], sq_bits: i32) -> Result<(Vec<u8>, Vec<u8>, Vec<f32>), String> {
    let cent = sq_centroids;

    let bits = sq_bits as usize;
    if bits != 1 && bits != 3 {
        return Err(format!("tq_quantize_rotated only supports sq_bits=1 or 3 (got {})", bits));
    }

    let k = 1usize << bits;
    if cent.len() != k {
        return Err(format!("sq_centroids length must be {} for sq_bits={}, got {}", k, bits, cent.len()));
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
                    let v = x_rot[row * d + j];
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
    }
    return Ok((sq_codes, qjl_signs, res_norms));
}

pub fn tq_kmeans_train(x_arr: &[f32], n: usize, d: usize, n_list: usize, iters: usize) -> Vec<f32> {
    
    if n < n_list { panic!("Points < n_list"); }
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.shuffle(&mut rng);
    let mut centroids = vec![0.0f32; n_list * d];
    for i in 0..n_list {
        let x_idx = idxs[i];
        for j in 0..d { centroids[i * d + j] = x_arr[x_idx * d + j]; }
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
        
        let x_view = ndarray::ArrayView2::from_shape((n, d), x_arr).unwrap();
        {
            let chunk_size = 1024; // Tối ưu hơn cho SIMD mà vẫn an toàn RAM
            assignments.par_chunks_mut(chunk_size).enumerate().for_each(|(c_idx, a_chunk)| {
                let start = c_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let x_chunk = x_view.slice(ndarray::s![start..end, ..]);
                
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
        }
        let mut next = vec![0.0f32; n_list * d];
        let mut counts = vec![0usize; n_list];
        for i in 0..n {
            let cid = assignments[i];
            for j in 0..d { next[cid*d+j] += x_arr[i * d + j]; }
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
    centroids
}

pub fn tq_assign_clusters(x_arr: &[f32], n: usize, d: usize, c_arr: &[f32], n_list: usize) -> Vec<i32> {
    let mut ass = vec![0i32; n];
    ass.par_chunks_mut(1024).enumerate().for_each(|(c_idx, a_chunk)| {
        let start = c_idx * 1024;
        let end = (start + 1024).min(n);
        a_chunk.iter_mut().enumerate().for_each(|(i, out)| {
            let row_idx = start + i;
            let mut ms = f32::MIN;
            let mut bc = 0usize;
            for cid in 0..n_list {
                let mut s = 0.0;
                for j in 0..d { s += x_arr[row_idx * d + j] * c_arr[cid * d + j]; }
                if s > ms { ms = s; bc = cid; }
            }
            *out = bc as i32;
        });
    });
    ass
}

pub fn tq_unified_search(
    queries_flat: &[f32],
    rot_op: &[f32],
    coarse_view: &[f32],
    offsets_sl: &[i32],
    vector_ids_sl: &[i64],
    sq_flat: &[u8],
    cent_sl: &[f32],
    norms_sl: &[f32],
    signs_flat: &[u8],
    res_sl: &[f32],
    qjl_scale: f32,
    dim: usize,
    mse_bits: usize,
    n_probe: usize,
    top_k: usize,
    allowed_set: Option<&std::collections::HashSet<i64>>,
) -> (Vec<f32>, Vec<i64>) {
    use std::sync::Mutex;
    use std::collections::{BinaryHeap, HashSet};
    use std::cmp::Reverse;

    let d = dim;
    let num_queries = 1;
    let num_centroids = offsets_sl.len() - 1;
    let qjl_dim = d;
    
    // Rotate query
    let mut qjl_queries_flat = vec![0.0f32; d];
    for i in 0..d {
        let mut s = 0.0;
        for j in 0..d { s += queries_flat[j] * rot_op[j * d + i]; }
        qjl_queries_flat[i] = s;
    }

    let packed_sq_d = if mse_bits == 1 { d / 8 } else { d / 2 };
    let packed_qjl_d = qjl_dim / 8;
    let scan_top_k = top_k; // No reranking for simple version

    // LUTs pre-computation
    let mut all_sq_luts = vec![0.0f32; d * 8 * 8];
    let mut all_qjl_luts = vec![0.0f32; (qjl_dim / 4) * 16 * 8];

    for k in 0..d {
        let qv = qjl_queries_flat[k];
        for b in 0..8 { all_sq_luts[(k * 8 + b) * 8] = qv * cent_sl[b]; }
    }

    for k in 0..(qjl_dim / 4) {
        for b in 0..16 {
            let mut s = 0.0f32;
            for v in 0..4 {
                let qv = qjl_queries_flat[k * 4 + v];
                let sign = if ((b >> v) & 1) == 1 { 1.0f32 } else { -1.0f32 };
                s += qv * sign;
            }
            all_qjl_luts[(k * 16 + b) * 8] = s;
        }
    }

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

    let mut global_heaps = Mutex::new(BinaryHeap::with_capacity(top_k));

    let mut query_to_clusters = vec![0usize; n_probe];
    let mut query_to_cluster_ip = vec![0.0f32; n_probe];

    let mut dists: Vec<(f32, usize)> = Vec::with_capacity(num_centroids);
    for ci in 0..num_centroids {
        let mut ip = 0.0;
        for j in 0..d { ip += queries_flat[j] * coarse_view[ci * d + j]; }
        dists.push((-ip, ci));
    }
    let actual_probe = n_probe.min(num_centroids);
    if actual_probe > 0 {
        dists.select_nth_unstable_by(actual_probe - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 0..actual_probe {
            query_to_clusters[i] = dists[i].1;
            query_to_cluster_ip[i] = -dists[i].0;
        }
    }

    let mut cluster_to_queries: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_centroids];
    for j in 0..actual_probe {
        let cid = query_to_clusters[j];
        let score = query_to_cluster_ip[j];
        cluster_to_queries[cid].push((0, score));
    }

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

        for qchunk in cluster_queries.chunks(8) {
            for buf in local_buffers.iter_mut() { buf.clear(); }
            thresholds = [f32::MIN; 8];
            let mut centroid_bias = [0.0f32; 8];

            for (lq, &(qi, score)) in qchunk.iter().enumerate() {
                centroid_bias[lq] = score;
            }

            unsafe {
                use std::arch::x86_64::*;
                let v_bias = _mm256_loadu_ps(centroid_bias.as_ptr());
                
                for i in start..end {
                    let global_id = vector_ids_sl[i];
                    if let Some(ref set) = allowed_set {
                        if !set.contains(&global_id) { continue; }
                    }

                    let rsq = i * packed_sq_d;
                    let rqj = i * packed_qjl_d;

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
                            v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8) * 8 + (s0 & 1)) * 8)));
                            v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 1) * 8 + ((s0 >> 1) & 1)) * 8)));
                            v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 2) * 8 + ((s0 >> 2) & 1)) * 8)));
                            v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 3) * 8 + ((s0 >> 3) & 1)) * 8)));
                            v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 4) * 8 + ((s0 >> 4) & 1)) * 8)));
                            v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 5) * 8 + ((s0 >> 5) & 1)) * 8)));
                            v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 6) * 8 + ((s0 >> 6) & 1)) * 8)));
                            v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 7) * 8 + ((s0 >> 7) & 1)) * 8)));

                            let b = signs_flat[rqj + k] as usize;
                            let b0 = b & 15;
                            let b1 = b >> 4;
                            v_qjl0 = _mm256_add_ps(v_qjl0, _mm256_loadu_ps(all_qjl_luts.as_ptr().add(((k * 2) * 16 + b0) * 8)));
                            v_qjl1 = _mm256_add_ps(v_qjl1, _mm256_loadu_ps(all_qjl_luts.as_ptr().add(((k * 2 + 1) * 16 + b1) * 8)));
                        }
                    } else {
                        for k in 0..(d / 8) {
                            let s_idx = rsq + k * 4;
                            let s0 = sq_flat[s_idx] as usize;
                            v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8) * 8 + (s0 & 7)) * 8)));
                            v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 1) * 8 + ((s0 >> 3) & 7)) * 8)));

                            let s1 = sq_flat[s_idx + 1] as usize;
                            v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 2) * 8 + (s1 & 7)) * 8)));
                            v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 3) * 8 + ((s1 >> 3) & 7)) * 8)));

                            let s2 = sq_flat[s_idx + 2] as usize;
                            v_sq0 = _mm256_add_ps(v_sq0, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 4) * 8 + (s2 & 7)) * 8)));
                            v_sq1 = _mm256_add_ps(v_sq1, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 5) * 8 + ((s2 >> 3) & 7)) * 8)));

                            let s3 = sq_flat[s_idx + 3] as usize;
                            v_sq2 = _mm256_add_ps(v_sq2, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 6) * 8 + (s3 & 7)) * 8)));
                            v_sq3 = _mm256_add_ps(v_sq3, _mm256_loadu_ps(all_sq_luts.as_ptr().add(((k * 8 + 7) * 8 + ((s3 >> 3) & 7)) * 8)));

                            let b = signs_flat[rqj + k] as usize;
                            let b0 = b & 15;
                            let b1 = b >> 4;
                            v_qjl0 = _mm256_add_ps(v_qjl0, _mm256_loadu_ps(all_qjl_luts.as_ptr().add(((k * 2) * 16 + b0) * 8)));
                            v_qjl1 = _mm256_add_ps(v_qjl1, _mm256_loadu_ps(all_qjl_luts.as_ptr().add(((k * 2 + 1) * 16 + b1) * 8)));
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
                let mut g_h = global_heaps.lock().unwrap();
                for &(s, id) in local_buffers[lq].iter() {
                    if g_h.len() < scan_top_k { g_h.push(Reverse((s, id))); }
                    else if s > g_h.peek().unwrap().0.0 { g_h.pop(); g_h.push(Reverse((s, id))); }
                }
            }
        }
    });

    let mut fs = vec![0.0f32; top_k];
    let mut fi = vec![-1i64; top_k];

    let mut h = global_heaps.lock().unwrap();
    let mut r = Vec::new();
    while let Some(Reverse(e)) = h.pop() { r.push(e); }
    r.reverse();

    for (k, &(s, id)) in r.iter().enumerate() {
        if k < top_k { fs[k] = ordered_u32_to_float(s); fi[k] = vector_ids_sl[id as usize]; }
    }

    (fs, fi)
}
