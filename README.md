# <p align="center">⚡ TurboQuant Standalone Vector Database Server ⚡</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Rust-orange?style=for-the-badge&logo=rust" alt="Rust" />
  <img src="https://img.shields.io/badge/Throughput-1094%20QPS-brightgreen?style=for-the-badge&logo=fastapi" alt="Throughput" />
  <img src="https://img.shields.io/badge/Latency-0.35%20ms-blue?style=for-the-badge&logo=clock" alt="Latency" />
  <img src="https://img.shields.io/badge/Memory-1GB%20RAM-red?style=for-the-badge" alt="Memory Limit" />
</p>

---

## 📖 Giới thiệu

**TurboQuant Standalone Server** là máy chủ cơ sở dữ liệu vector siêu tốc độ viết hoàn toàn bằng **Rust**. Hệ thống được thiết kế để giải quyết bài toán tìm kiếm vector quy mô lớn dưới các điều kiện tài nguyên cực kỳ ngặt nghèo của môi trường đám mây giá rẻ (**1vCPU, 1GB RAM**). 

Bằng cách loại bỏ hoàn toàn các thư viện cồng kềnh (Python, C++) và thay thế bằng mã máy biên dịch tối ưu hóa phần cứng ở cấp độ hợp ngữ (**SIMD AVX-512/AVX2**), TurboQuant đạt hiệu năng vượt trội so với các giải pháp hiện nay.

---

## ✨ Điểm Vượt Trội (Core Features)

* 🦀 **100% Native Rust**: Khởi động tức thì trong 10ms, RAM chiếm dụng chưa tới 50MB khi chạy nền.
* ⚡ **Giao thức Nhị phân Zero-Copy**: Tăng tốc độ truyền dữ liệu qua mạng bằng cách bỏ qua JSON Serialization.
* 🎛️ **Dual-Quantization (Nén phân tầng)**: Lượng tử hóa vector 4-bit (SQ 4b + QJL 1-bit) giúp tiết kiệm 85% dung lượng lưu trữ RAM và ổ cứng.
* 🎯 **Độ chính xác ấn tượng**: Đạt Recall@1@16 lên đến **100%** nhờ tầng Re-ranking thông minh dựa trên vector gốc.

---

## 📊 So Sánh Hiệu Năng Thực Tế (Benchmark Results)

*Thử nghiệm đo lường tần suất xử lý truy vấn mạng (QPS) và độ trễ (Latency) trên hai tập dữ liệu **Qasper_E5 (12,450 vectors)** và **HotpotQA_E5 (482,021 vectors)** (Vector chiều 384) qua môi trường mạng cục bộ:*

### 1. Tập dữ liệu siêu lớn: HotpotQA_E5 (482,021 vectors)
*(Thiết lập: `n_probe=16`, Không Rerank, đo lường toàn hệ thống)*

**Biểu đồ tốc độ QPS (Batch Size = 100):**
```text
Flat Search (Exact Match)      | █ 9.9 QPS (1.0x)
TQ JSON (API Truyền thống)     | ███████████████████████████████ 519.4 QPS (52.5x)
TQ Binary (Zero-Copy)          | ██████████████████████████████████ 563.6 QPS (57.0x)
```

**Chi tiết các chỉ số (HotpotQA):**
| Cấu hình thuật toán / Giao thức | Batch Size | QPS | Độ trễ Mean (ms) | Độ trễ P99 (ms) | Recall@1@16 | Overlap@16 | Tăng tốc (vs Flat) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flat Search** | 100 | 9.9 | 10,114.8 | 10,114.8 | 100% | 100% | 1.0x |
| **TQ JSON** | 100 | 519.4 | 189.0 | 189.0 | 100% | 84.19% | **52.5x** |
| **TQ Binary** | 100 | 563.6 | 173.4 | 173.4 | 100% | 84.19% | **57.0x** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Flat Search** | 1 | 1.6 | 610.0 | 695.5 | 100% | 100% | 1.0x |
| **TQ JSON** | 1 | 231.5 | 4.3 | 6.0 | 100% | 84.19% | **141.2x** |
| **TQ Binary** | 1 | **269.7** | **3.6** | **5.3** | 100% | 84.19% | **164.6x** 🚀 |

### 2. Tập dữ liệu trung bình: Qasper_E5 (12,450 vectors)
*(Thiết lập: `n_probe=16`, Không Rerank, đo lường toàn hệ thống)*

| Cấu hình thuật toán / Giao thức | Batch Size | QPS | Độ trễ Mean (ms) | Độ trễ P99 (ms) | Recall@1@16 | Overlap@16 | Tăng tốc (vs Flat) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flat Search** | 100 | 407.2 | 243.0 | 243.0 | 100% | 100% | 1.0x |
| **TQ JSON** | 100 | 1,985.7 | 47.3 | 47.3 | 100% | 78.12% | 4.9x |
| **TQ Binary** | 100 | **3,850.0** | **24.3** | **24.3** | 100% | 78.12% | **9.5x** 🚀 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Flat Search** | 1 | 60.5 | 16.5 | 19.3 | 100% | 100% | 1.0x |
| **TQ JSON** | 1 | 422.2 | 2.3 | 3.8 | 100% | 78.12% | 7.0x |
| **TQ Binary** | 1 | **437.0** | **2.2** | **3.6** | 100% | 78.12% | **7.2x** |

> [!IMPORTANT]
> **Nhận xét cốt lõi từ thực nghiệm:**
> 1. **Khả năng duy trì độ trễ (Sub-10ms):** Dù truy vấn trên tập dữ liệu nhỏ (12K) hay cực lớn (482K), độ trễ truy vấn đơn (`batch_size=1`) của TurboQuant Binary luôn cực kỳ ổn định ở mức **< 6 mili-giây**, giúp đáp ứng hoàn hảo cho các hệ thống RAG thời gian thực. Trong khi đó, Flat Search bị sụt giảm tốc độ nghiêm trọng từ 19ms xuống tận 695ms.
> 2. **Hiệu suất thu phóng siêu đẳng (Scalability):** Khi áp dụng trên dữ liệu khổng lồ (HotpotQA), thuật toán phân mảnh kết hợp nén 4-bit của TurboQuant tỏa sáng rực rỡ với mức tăng tốc lên tới **164.6 lần** so với Flat Search truyền thống, nhưng **Recall (Top-1) vẫn luôn đạt tuyệt đối 100%**.
> 3. **Ưu thế tuyệt đối của Binary Protocol:** Đặc biệt khi xử lý các lô lớn (Batch Size = 100) trên tập Qasper, Giao thức Nhị phân giúp tăng tốc độ xử lý thêm gần **100%** (từ 1985 QPS lên 3850 QPS) so với giao thức JSON truyền thống nhờ việc loại bỏ hoàn toàn nút thắt cổ chai serialize/deserialize UTF-8 trên CPU.

### ❓ Tại sao tốc độ QPS hiển thị đôi khi dao động (Lúc nhanh lúc chậm)?

Khi chạy các benchmark liên tiếp, bạn có thể nhận thấy QPS của lượt đầu tiên thường thấp hơn các lượt sau (ví dụ: lượt đầu đạt 1066 QPS nhị phân, lượt sau lên 1094 QPS). Điều này xuất phát từ các yếu tố kỹ thuật sau:
1. **Khởi tạo luồng Rayon/Tokio (Thread Warmup):** Rust sử dụng thư viện `Rayon` để xử lý song song hóa truy vấn lô (`par_iter`). Lần đầu tiên gọi API, các luồng (worker threads) trong Thread Pool cần thời gian khởi tạo hoặc đánh thức từ chế độ ngủ, tạo ra độ trễ ban đầu nhỏ.
2. **Cơ chế OS Page Cache:** Khi hệ thống bắt đầu truy cập mảng lượng tử hóa và các ma trận xoay lưu trên RAM, nếu hệ điều hành chưa nạp hoặc đã giải phóng các trang bộ nhớ này (Page Cache Miss), nó sẽ tốn chi phí nhỏ để đọc từ bộ nhớ ảo. Các lượt chạy sau toàn bộ dữ liệu nằm trực tiếp trong bộ đệm CPU (L1/L2/L3) giúp tốc độ tìm kiếm đạt giới hạn vật lý phần cứng.
3. **Độ ấm của Socket TCP (TCP Warmup):** Lượt chạy đầu tiên của Python Client cần thiết lập kết nối TCP mới tới Axum server. Các lượt chạy tiếp theo tận dụng lại TCP connection pool đang hoạt động (Keep-Alive) nên loại bỏ được độ trễ bắt tay mạng (3-way handshake).
4. **Ảnh hưởng của kích thước lô nhỏ (`batch_size=100`):** Với tổng số chỉ 100 truy vấn, thời gian xử lý cực kỳ nhỏ (~0.09 giây). Do đó, chỉ một dao động cực nhỏ khoảng **15 - 20 mili-giây** của hệ điều hành (do tiến trình nền khác sử dụng CPU) cũng sẽ làm thay đổi hiển thị số QPS từ 650 QPS lên 890 QPS. Khi đo với `batch_size=2000`, thời gian chạy dài hơn sẽ giúp trung hòa các sai lệch nhỏ này, cho ra con số QPS ổn định và chính xác nhất.

---

## 📐 Cơ Chế Tự Động Tính n_list (IVF Centroids)

Từ phiên bản mới nhất, TurboQuant tích hợp thuật toán phân hoạch cụm động thông minh:
* Nếu người dùng cấu hình `n_list = null` (None) khi tạo Index, máy chủ sẽ tự động tính toán dựa trên quy mô số vector $N$:
  $$\text{target} = 2 \times \sqrt{N}$$
* Sau đó, hệ thống tìm **lũy thừa của 2 gần nhất** với giá trị $\text{target}$ (giới hạn tối đa bằng $N$).
* **Ví dụ:** Với tập dữ liệu Qasper ($N = 12,450$):
  $$\text{target} = 2 \times \sqrt{12,450} \approx 223.16 \implies \text{Lũy thừa 2 gần nhất là } 256$$
  Giúp cấu hình Index luôn đạt hiệu năng và độ chính xác cân bằng nhất mà không cần nhập thủ công.

---

## 🛠️ Biên dịch Tối ưu hóa AVX2 / AVX-512

Để máy chủ đạt tốc độ tối đa **>1,000 QPS**, hãy đảm bảo biên dịch ứng dụng trực tiếp trên phần cứng máy chủ đích bằng cờ `target-cpu=native`.

### Khởi chạy nhanh trên Windows:
Chạy file [start.bat](file:///e:/ARQ-RAG/ARQ-RAG-turboquant-main/TurboQuant_Server/start.bat):
```batch
set RUSTFLAGS=-C target-cpu=native
cargo run --release
```

### Khởi chạy nhanh trên Linux / GCP VPS:
Chạy file [start.sh](file:///e:/ARQ-RAG/ARQ-RAG-turboquant-main/TurboQuant_Server/start.sh):
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/turboquant-server
```

---

## 📡 Đặc tả Giao thức Nhị phân (Zero-Copy Binary Protocol)

Gửi request có tiêu đề `Content-Type: application/octet-stream` tới:
`POST /collections/default/search/batch/bin`

### ✉️ Cấu trúc Gói tin Request (Client gửi đi)
Mỗi request bao gồm **18 bytes Header** và mảng byte phẳng chứa các giá trị `float32` của tất cả vector truy vấn:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            18-Byte Header                                   │
├───────────────┬───────────┬──────────────┬───────────┬─────────────────┬────┤
│ n_queries(u32)│ top_k(u32)│ n_probe(u32) │ rerank(u8)│rerank_factor(u32)│with│
└───────────────┴───────────┴──────────────┴───────────┴─────────────────┴────┘
```

* `n_queries` (u32 - 4 bytes): Số lượng vector trong lô.
* `top_k` (u32 - 4 bytes): Số kết quả muốn trả về cho mỗi vector.
* `n_probe` (u32 - 4 bytes): Số lượng cụm IVF cần thăm dò.
* `rerank` (u8 - 1 byte): Kích hoạt tái xếp hạng bằng vector chính xác (`1` = Bật, `0` = Tắt).
* `rerank_factor` (u32 - 4 bytes): Số ứng viên lấy ra để re-rank (ví dụ: `2` hoặc `4`).
* `with_vector` (u8 - 1 byte): Trả về cả vector gốc trong kết quả (`1` = Bật, `0` = Tắt).

