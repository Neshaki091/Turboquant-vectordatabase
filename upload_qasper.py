import numpy as np
import requests
import time
import argparse

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def main():
    parser = argparse.ArgumentParser(description="Upload Qasper E5 dataset to TurboQuant Server")
    parser.add_argument("--file", type=str, default=r"e:\ARQ-RAG\ARQ-RAG-turboquant-main\tq_java_test\Qasper_E5\corpus_embedded_norm.npy", help="Path to the .npy file")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:6333/collections/default/points/batch", help="TurboQuant Server API endpoint")
    parser.add_argument("--batch_size", type=int, default=5000, help="Number of vectors per batch")
    
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
    config_url = args.url.replace("/points/batch", "/config")
    try:
        config_res = requests.post(config_url, json={"n_list": None, "quantize_bits": 4})
        if config_res.status_code == 200:
            print(f">>> Cau hinh thanh cong: {config_res.text}")
        else:
            print(f">>> Loi cau hinh: {config_res.status_code}")
    except Exception as e:
        print(f">>> Khong the goi cau hinh: {e}")

    print(f"\n[2] Bat dau Upload {num_vectors} vectors toi {args.url} (Batch Size: {args.batch_size})...")
    
    start_time = time.time()
    
    session = requests.Session()
    print("Sending batches...")
    
    for start_idx in range(0, num_vectors, args.batch_size):
        end_idx = min(start_idx + args.batch_size, num_vectors)
        chunk = []
        for i in range(start_idx, end_idx):
            chunk.append({
                "id": i,
                "vector": data[i].tolist(),
                "payload": {"source": "Qasper_E5", "index": i}
            })
            
        payload = {"points": chunk}
        try:
            res = session.post(args.url, json=payload)
            if res.status_code == 200:
                elapsed = time.time() - start_time
                rate = end_idx / elapsed
                print(f"Progress: {end_idx}/{num_vectors} ({(end_idx/num_vectors)*100:.1f}%) - Rate: {rate:.1f} vec/s")
            else:
                print(f"Batch {start_idx}-{end_idx} failed: {res.text}")
        except Exception as e:
            print(f"Exception during batch {start_idx}-{end_idx}: {e}")

    end_time = time.time()
    print(f"\nUpload completed in {end_time - start_time:.2f} seconds.")

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
        search_res = session.post(args.url.replace("/points/batch", "/search"), json=search_payload)
        if search_res.status_code == 200:
            print(">>> Index Build & Search hoan tat thanh cong!")
        else:
            print(f">>> Loi khi Search: {search_res.status_code}")
    except Exception as e:
        print(f">>> Loi khi goi Search: {e}")

    print("\n[4] Dang luu toan bo Database xuong O cung...")
    try:
        save_url = args.url.replace("/points/batch", "/save")
        save_res = session.post(save_url)
        if save_res.status_code == 200:
            print(f">>> Da luu thanh cong: {save_res.text}")
        else:
            print(f">>> Loi khi luu: {save_res.status_code}")
    except Exception as e:
        print(f">>> Loi khi goi Save: {e}")


if __name__ == "__main__":
    main()
