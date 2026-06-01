# TurboQuant Standalone VectorDB

Đây là phiên bản lõi máy chủ (Server) của TurboQuant được viết 100% bằng Rust.
Nó hoạt động tương tự như Qdrant: Không cần Python, Không cần Java. Nó tự mở cổng REST API HTTP (cổng 6333) để các ngôn ngữ khác gọi vào.

## Cách sử dụng

Chỉ cần click đúp vào file `start.bat` (trên Windows) hoặc gõ `cargo run --release` trên Terminal.

## API Endpoints (Tương tự Qdrant)

### 1. Thêm Vector & Payload (POST /collections/default/points)
```json
{
  "id": 1,
  "vector": [0.1, 0.2, 0.3],
  "payload": {"category": "tech"}
}
```

### 2. Tìm kiếm (POST /collections/default/search)
```json
{
  "vector": [0.1, 0.2, 0.3],
  "top_k": 5
}
```
