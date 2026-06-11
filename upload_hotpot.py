import numpy as np
import requests
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import threading

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

def upload_vector(i, vec, url):
    payload = {
        "id": i,
        "vector": vec.tolist(),
        "payload": {"source": "HotpotQA_E5", "index": i}
    }
    try:
        session = get_session()
        res = session.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        return False

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    parser = argparse.ArgumentParser(description="Upload HotpotQA E5 dataset to TurboQuant Server (Batch Mode)")
    parser.add_argument("--file", type=str, default=r"e:\ARQ-RAG\ARQ-RAG-turboquant-main\Benchmark\data\HotpotQA_E5\corpus_embedded_norm.npy", help="Path to the .npy file")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:6333/collections/default/points", help="TurboQuant Server API endpoint")
    parser.add_argument("--batch_size", type=int, default=10000, help="Number of vectors to send per batch request")
    parser.add_argument("--max_samples", type=int, default=50000, help="Max random samples for K-Means training")
    
    args = parser.parse_args()

    print(f"Loading data from {args.file}...")
    try:
        data = np.load(args.file)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    print(f"Loaded shape: {data.shape} (Type: {data.dtype})")
    num_vectors = data.shape[0]

    print(f"\n[1] Cau hinh TurboQuant Index (IVF, 4-bit, {args.max_samples} max samples)...")
    config_url = args.url.replace("/points", "/config")
    try:
        config_res = requests.post(config_url, json={"n_list": None, "quantize_bits": 4, "max_training_samples": args.max_samples})
        if config_res.status_code == 200:
            print(f">>> Cau hinh thanh cong: {config_res.text}")
        else:
            print(f">>> Loi cau hinh: {config_res.status_code}")
    except Exception as e:
        print(f">>> Khong the goi cau hinh: {e}")

    batch_url = args.url + "/batch"
    print(f"\n[2] Bat dau Upload {num_vectors} vectors toi {batch_url} (Batch Size: {args.batch_size})...")
    
    start_time = time.time()
    success_count = 0

    print("Sending batches...")

    with requests.Session() as session:
        for start_idx in range(0, num_vectors, args.batch_size):
            end_idx = min(start_idx + args.batch_size, num_vectors)
            chunk = []
            for i in range(start_idx, end_idx):
                chunk.append({
                    "id": i,
                    "vector": data[i].tolist(),
                    "payload": {"source": "HotpotQA_E5", "index": i}
                })
            
            payload = {"points": chunk}
            try:
                res = session.post(batch_url, json=payload, timeout=30)
                if res.status_code == 200:
                    success_count += len(chunk)
                else:
                    print(f"Batch Error: {res.status_code} - {res.text}")
            except Exception as e:
                print(f"Batch Exception: {e}")
                
            elapsed = time.time() - start_time
            rate = success_count / elapsed if elapsed > 0 else 0
            print(f"Progress: {success_count}/{num_vectors} ({(success_count/num_vectors)*100:.1f}%) - Rate: {rate:.1f} vec/s")

    end_time = time.time()
    print(f"\nUpload completed in {end_time - start_time:.2f} seconds.")
    print(f"Successfully uploaded {success_count}/{num_vectors} vectors.")

    print("\n[3] Kich hoat qua trinh Build Index (IVF 4-bit)...")
    try:
        search_payload = {
            "vector": data[0].tolist(),
            "top_k": 5,
            "params": {
                "exact": False,
                "n_probe": 10
            }
        }
        search_res = requests.post(args.url.replace("/points", "/search"), json=search_payload)
        if search_res.status_code == 200:
            print(">>> Index Build & Search hoan tat thanh cong!")
        else:
            print(f">>> Loi khi Search: {search_res.status_code}")
    except Exception as e:
        print(f">>> Loi khi goi Search: {e}")

    print("\n[4] Dang luu toan bo Database xuong O cung...")
    try:
        save_url = args.url.replace("/points", "/save")
        save_res = requests.post(save_url)
        if save_res.status_code == 200:
            print(f">>> Da luu thanh cong: {save_res.text}")
        else:
            print(f">>> Loi khi luu: {save_res.status_code}")
    except Exception as e:
        print(f">>> Loi khi goi Save: {e}")

if __name__ == "__main__":
    main()
