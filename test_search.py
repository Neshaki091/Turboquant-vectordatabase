import numpy as np
import requests
import time
import argparse

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    parser = argparse.ArgumentParser(description="Test TurboQuant Search Performance")
    parser.add_argument("--file", type=str, default=r"e:\ARQ-RAG\ARQ-RAG-turboquant-main\tq_java_test\Qasper_E5\corpus_embedded_norm.npy", help="Path to the .npy file")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:6333/collections/default/search", help="TurboQuant Server Search endpoint")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of queries to run for benchmarking")
    parser.add_argument("--top_k", type=int, default=10, help="Top K results to retrieve")
    parser.add_argument("--n_probe", type=int, default=10, help="Number of IVF clusters to probe (for 4-bit search)")
    parser.add_argument("--rerank", action="store_true", help="Enable server-side re-ranking using exact vectors")
    parser.add_argument("--rerank_factor", type=int, default=10, help="Retrieve top_k * rerank_factor candidates from ANN index for re-ranking")
    parser.add_argument("--with_vector", action="store_true", help="Return raw vectors in search response")
    parser.add_argument("--batch_size", type=int, default=1, help="If > 1, send queries in batches to /collections/default/search/batch")
    
    args = parser.parse_args()

    print(f"Loading queries from {args.file}...")
    try:
        data = np.load(args.file)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    queries = data[:args.num_queries]
    print(f"Prepared {len(queries)} queries. Top K = {args.top_k}, Dim = {queries.shape[1]}")
    print(f"Config: batch_size = {args.batch_size}, with_vector = {args.with_vector}")

    # ==========================================
    # 1. Test Flat Search (Chính xác 100%, không nén)
    # ==========================================
    print("\n" + "="*50)
    print(f">>> Bat dau Benchmark: FLAT SEARCH (Exact Match, batch_size={args.batch_size})")
    print("="*50)
    
    flat_results = []
    start_time = time.time()
    
    if args.batch_size > 1:
        batch_url = args.url.replace("/search", "/search/batch")
        for chunk in chunked(queries.tolist(), args.batch_size):
            payload = {
                "vectors": chunk,
                "top_k": args.top_k,
                "params": {
                    "exact": True,
                    "with_vector": args.with_vector
                }
            }
            res = requests.post(batch_url, json=payload)
            flat_results.extend(res.json())
    else:
        for q in queries:
            payload = {
                "vector": q.tolist(),
                "top_k": args.top_k,
                "params": {
                    "exact": True,
                    "with_vector": args.with_vector
                }
            }
            res = requests.post(args.url, json=payload)
            flat_results.append(res.json())
        
    flat_time = time.time() - start_time
    print(f"Xong Flat Search {len(queries)} queries trong {flat_time:.4f} giay")
    print(f"QPS (Queries Per Second): {len(queries) / flat_time:.1f}")

    # ==========================================
    # 2. Test TurboQuant 4-bit Search (Siêu tốc độ)
    # ==========================================
    print("\n" + "="*50)
    print(f">>> Bat dau Benchmark: TURBOQUANT 4-BIT (IVF, n_probe={args.n_probe}, rerank={args.rerank}, batch_size={args.batch_size})")
    print("="*50)
    
    tq_results = []
    start_time = time.time()
    
    if args.batch_size > 1:
        batch_url = args.url.replace("/search", "/search/batch")
        for chunk in chunked(queries.tolist(), args.batch_size):
            payload = {
                "vectors": chunk,
                "top_k": args.top_k,
                "params": {
                    "exact": False,
                    "n_probe": args.n_probe,
                    "rerank": args.rerank,
                    "rerank_factor": args.rerank_factor,
                    "with_vector": args.with_vector
                }
            }
            res = requests.post(batch_url, json=payload)
            tq_results.extend(res.json())
    else:
        for q in queries:
            payload = {
                "vector": q.tolist(),
                "top_k": args.top_k,
                "params": {
                    "exact": False,
                    "n_probe": args.n_probe,
                    "rerank": args.rerank,
                    "rerank_factor": args.rerank_factor,
                    "with_vector": args.with_vector
                }
            }
            res = requests.post(args.url, json=payload)
            tq_results.append(res.json())
        
    tq_time = time.time() - start_time
    print(f"Xong TurboQuant Search {len(queries)} queries trong {tq_time:.4f} giay")
    print(f"QPS (Queries Per Second): {len(queries) / tq_time:.1f}")
    
    print(f"\n>>> TOC DO CUA TURBOQUANT (JSON): NHANH HON {flat_time / tq_time:.1f} LAN!")

    # ==========================================
    # 3. Test TurboQuant Binary Search (Zero-Copy)
    # ==========================================
    print("\n" + "="*50)
    print(f">>> Bat dau Benchmark: TURBOQUANT BINARY (Zero-Copy, batch_size={args.batch_size})")
    print("="*50)
    
    import struct
    tq_bin_results = []
    start_time = time.time()
    
    bin_url = args.url.replace("/search", "/search/batch/bin")
    
    # Python struct format for request header:
    # [n_queries (u32)][top_k (u32)][n_probe (u32)][rerank (u8)][rerank_factor (u32)][with_vector (u8)]
    # Little endian format: <IIIbIb
    for chunk in chunked(queries.tolist(), args.batch_size):
        header = struct.pack("<IIIbIb", len(chunk), args.top_k, args.n_probe, 1 if args.rerank else 0, args.rerank_factor, 1 if args.with_vector else 0)
        query_bytes = np.array(chunk, dtype=np.float32).tobytes()
        payload = header + query_bytes
        
        res = requests.post(bin_url, data=payload, headers={"Content-Type": "application/octet-stream"})
        res_bytes = res.content
        
        # Parse binary response
        offset = 0
        for _ in range(len(chunk)):
            num_res = struct.unpack_from("<I", res_bytes, offset)[0]
            offset += 4
            query_hits = []
            for _ in range(num_res):
                vec_id, score, has_vector = struct.unpack_from("<QfB", res_bytes, offset)
                offset += 13
                vector = None
                if has_vector:
                    vector = np.frombuffer(res_bytes, dtype=np.float32, count=384, offset=offset).tolist()
                    offset += 384 * 4
                query_hits.append({
                    "id": vec_id,
                    "score": score,
                    "vector": vector
                })
            tq_bin_results.append(query_hits)
            
    tq_bin_time = time.time() - start_time
    print(f"Xong TurboQuant Binary Search {len(queries)} queries trong {tq_bin_time:.4f} giay")
    print(f"QPS (Queries Per Second): {len(queries) / tq_bin_time:.1f}")
    
    print(f"\n>>> TOC DO CUA TURBOQUANT BINARY: NHANH HON {flat_time / tq_bin_time:.1f} LAN!")
    print(f">>> SOI SANG: BINARY NHANH HON JSON {tq_time / tq_bin_time:.2f} LAN!")

    # ==========================================
    # 4. So sánh độ chính xác (Recall)
    # ==========================================
    print("\n" + "="*50)
    print("So sanh ket qua Query dau tien (Flat vs TurboQuant Binary)")
    print("="*50)
    
    if len(flat_results) > 0 and len(tq_bin_results) > 0:
        flat_top = flat_results[0]
        tq_top = tq_bin_results[0]
        
        print(">>> FLAT Ket qua (ID - Score):")
        for hit in flat_top[:10]:
            print(f"    ID: {hit['id']:<6} | Score: {hit['score']:.4f}")
            
        print("\n>>> TURBOQUANT BINARY Ket qua (ID - Score):")
        for hit in tq_top[:10]:
            print(f"    ID: {hit['id']:<6} | Score: {hit['score']:.4f}")
            
        # Tính các loại Recall
        total_intersection = 0
        total_top1_hit = 0
        
        for i in range(len(queries)):
            flat_ids = {hit['id'] for hit in flat_results[i]}
            tq_ids = {hit['id'] for hit in tq_bin_results[i]}
            
            # 1. Tỉ lệ trùng khớp tập hợp (Intersection K@K)
            intersection = flat_ids.intersection(tq_ids)
            total_intersection += len(intersection) / args.top_k
            
            # 2. ANN Recall@1@K (Có tìm thấy Best Match trong Top K không?)
            if len(flat_results[i]) > 0:
                exact_best_match_id = flat_results[i][0]['id']
                if exact_best_match_id in tq_ids:
                    total_top1_hit += 1
            
        avg_intersection = total_intersection / len(queries)
        recall_1_at_k = total_top1_hit / len(queries)
        
        print(f"\n>>> [1] Ti le trung khop tap hop (Top-{args.top_k} Overlap): {avg_intersection * 100:.2f}%")
        print(f">>> [2] ANN Recall@1@{args.top_k} (Best Match in Top K): {recall_1_at_k * 100:.2f}%")

if __name__ == "__main__":
    main()
