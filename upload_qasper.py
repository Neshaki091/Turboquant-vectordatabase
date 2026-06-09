import numpy as np
import requests
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def upload_vector(i, vec, url):
    payload = {
        "id": i,
        "vector": vec.tolist(),
        "payload": {"source": "Qasper_E5", "index": i}
    }
    try:
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            return True
        else:
            return False
    except Exception as e:
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload Qasper E5 dataset to TurboQuant Server")
    parser.add_argument("--file", type=str, default=r"e:\ARQ-RAG\ARQ-RAG-turboquant-main\tq_java_test\Qasper_E5\corpus_embedded_norm.npy", help="Path to the .npy file")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:6333/collections/default/points", help="TurboQuant Server API endpoint")
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent upload workers")
    
    args = parser.parse_args()

    print(f"Loading data from {args.file}...")
    try:
        data = np.load(args.file)
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    print(f"Loaded shape: {data.shape} (Type: {data.dtype})")
    num_vectors = data.shape[0]

    print(f"\n[1] Cau hinh TurboQuant Index (IVF, 4-bit)...")
    config_url = args.url.replace("/points", "/config")
    try:
        config_res = requests.post(config_url, json={"n_list": 256, "quantize_bits": 4})
        if config_res.status_code == 200:
            print(f">>> Cau hinh thanh cong: {config_res.text}")
        else:
            print(f">>> Loi cau hinh: {config_res.status_code}")
    except Exception as e:
        print(f">>> Khong the goi cau hinh: {e}")

    print(f"\n[2] Bat dau Upload {num_vectors} vectors toi {args.url} (Workers: {args.workers})...")
    
    start_time = time.time()
    success_count = 0

    # We use ThreadPoolExecutor to upload concurrently
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # submit all tasks
        futures = {executor.submit(upload_vector, i, data[i], args.url): i for i in range(num_vectors)}
        
        for idx, future in enumerate(as_completed(futures)):
            if future.result():
                success_count += 1
                
            # Print progress every 1000 items
            if (idx + 1) % 1000 == 0 or (idx + 1) == num_vectors:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                print(f"Progress: {idx + 1}/{num_vectors} ({((idx+1)/num_vectors)*100:.1f}%) - Rate: {rate:.1f} req/s")

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
