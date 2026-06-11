mod engine;
mod turboquant;

use axum::{
    routing::{get, post, delete},
    Router, Json, extract::{State, Path, DefaultBodyLimit},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use engine::TQEngine;
use tower_http::cors::CorsLayer;

// Trạng thái được chia sẻ giữa các luồng (Thread-safe)
#[derive(Clone)]
struct AppState {
    engine: Arc<RwLock<TQEngine>>,
}

#[derive(Deserialize)]
struct AddPointReq {
    id: u64,
    vector: Vec<f32>,
    payload: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct BatchAddPointReq {
    points: Vec<AddPointReq>,
}

#[derive(Deserialize)]
pub struct ConfigReq {
    pub n_list: Option<usize>,
    pub quantize_bits: usize,
    pub max_training_samples: Option<usize>,
}

#[derive(Deserialize, Clone)]
pub struct SearchParams {
    pub exact: Option<bool>,
    pub n_probe: Option<usize>,
    pub rerank: Option<bool>,
    pub rerank_factor: Option<usize>,
    pub with_vector: Option<bool>,
}

#[derive(Deserialize)]
pub struct SearchReq {
    pub vector: Vec<f32>,
    pub top_k: usize,
    pub params: Option<SearchParams>,
}

#[derive(Deserialize)]
pub struct BatchSearchReq {
    pub vectors: Vec<Vec<f32>>,
    pub top_k: usize,
    pub params: Option<SearchParams>,
}

#[derive(Serialize)]
pub struct SearchResult {
    id: u64,
    score: f32,
    vector: Option<Vec<f32>>,
    payload: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct PointDetail {
    id: u64,
    vector: Vec<f32>,
    payload: Option<serde_json::Value>,
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--bench") {
        println!("🔥 Running pure in-memory QPS benchmark...");
        let mut engine = TQEngine::new();
        engine.load_from_disk("data/tq_index").expect("Failed to load index");
        
        let npy_path = "e:\\ARQ-RAG\\ARQ-RAG-turboquant-main\\tq_java_test\\Qasper_E5\\corpus_embedded_norm.npy";
        let matrix: ndarray::Array2<f32> = ndarray_npy::read_npy(npy_path).expect("Failed to load npy file");
        
        let mut queries = vec![];
        for i in 0..100.min(matrix.shape()[0]) {
            queries.push(matrix.row(i).to_vec());
        }

        // Warmup
        for q in &queries {
            let _ = engine.search(q, 16, Some(SearchParams {
                exact: Some(false),
                n_probe: Some(16),
                rerank: Some(true),
                rerank_factor: Some(10),
                with_vector: Some(false),
            }));
        }

        let start = std::time::Instant::now();
        let loops = 100;
        for _ in 0..loops {
            let _ = engine.search_batch(&queries, 16, Some(SearchParams {
                exact: Some(false),
                n_probe: Some(16),
                rerank: Some(true),
                rerank_factor: Some(10),
                with_vector: Some(false),
            }));
        }
        let duration = start.elapsed();
        let total_queries = loops * queries.len();
        let qps = (total_queries as f64) / duration.as_secs_f64();
        println!("\n=============================================");
        println!("🔥 PURE ENGINE BENCHMARK (In-Memory, No HTTP/JSON)");
        println!("=============================================");
        println!("Pure Engine QPS: {:.2} QPS", qps);
        println!("Duration for {} queries: {:?}", total_queries, duration);
        println!("Average latency per query: {:.4} ms", duration.as_secs_f64() * 1000.0 / (total_queries as f64));
        println!("=============================================\n");
        return;
    }

    println!("🚀 Khởi động TurboQuant Server (Phiên bản Độc Lập)...");
    #[cfg(target_feature = "avx512f")]
    println!("⚡ SIMD Hardware Optimization: AVX-512 Active");
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    println!("⚡ SIMD Hardware Optimization: AVX2 Active");
    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    println!("⚠️ SIMD Hardware Optimization: None (Running in fallback scalar mode - make sure target-cpu=native is active)");
    
    
    let mut init_engine = TQEngine::new();
    if std::path::Path::new("data/tq_index").exists() {
        match init_engine.load_from_disk("data/tq_index") {
            Ok(_) => println!("✅ Đã load toàn bộ dữ liệu & lượng tử (data/tq_index)!"),
            Err(e) => println!("❌ Lỗi khi load dữ liệu: {:?}", e),
        }
    } else {
        println!("ℹ️ Không tìm thấy thư mục data/tq_index, khởi tạo DB mới.");
    }
    
    let state = AppState {
        engine: Arc::new(RwLock::new(init_engine)),
    };

    // Định tuyến API theo phong cách Qdrant
    let app = Router::new()
        .route("/", get(|| async { axum::response::Redirect::permanent("/dashboard") }))
        .route("/dashboard", get(|| async { axum::response::Html(include_str!("dashboard.html")) }))
        .route("/guide", get(|| async { axum::response::Html(include_str!("guide.html")) }))
        .route("/collections/default/points", get(list_points).post(add_point).delete(clear_points))
        .route("/collections/default/points/batch", post(add_batch_points))
        .route("/collections/default/search", post(search_points))
        .route("/collections/default/search/batch", post(search_batch_points))
        .route("/collections/default/search/batch/bin", post(search_batch_points_bin))
        .route("/collections/default/save", post(save_points))
        .route("/collections/default/config", post(configure_index))
        .route("/collections/default/points/:id", delete(delete_point))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 100)) // Giới hạn 100MB để chống DDoS
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:6333").await.unwrap();
    println!("✅ Server đang lắng nghe tại http://0.0.0.0:6333");
    
    axum::serve(listener, app).await.unwrap();
}

async fn add_point(State(state): State<AppState>, Json(req): Json<AddPointReq>) -> Json<&'static str> {
    let mut engine = state.engine.write().await;
    engine.add(req.id, req.vector, req.payload);
    Json("Success")
}

async fn add_batch_points(State(state): State<AppState>, Json(req): Json<BatchAddPointReq>) -> Json<&'static str> {
    let mut engine = state.engine.write().await;
    for point in req.points {
        engine.add(point.id, point.vector, point.payload);
    }
    Json("Success")
}

async fn search_points(State(state): State<AppState>, Json(req): Json<SearchReq>) -> Json<Vec<SearchResult>> {
    {
        let mut engine = state.engine.write().await;
        // Qdrant Optimizer Logic: Tự động Build Index nếu số vector chưa nén > 1000
        // Hoặc khi chưa từng có Index nhưng đã có dữ liệu.
        if engine.unindexed_ids.len() > 1000 || (engine.indexed_segment.is_none() && !engine.unindexed_ids.is_empty()) {
            engine.build_index();
        }
    }
    let engine = state.engine.read().await; // RwLock cho phép nhiều truy vấn đọc đồng thời
    let results = engine.search(&req.vector, req.top_k, req.params);
    Json(results)
}

async fn search_batch_points(State(state): State<AppState>, Json(req): Json<BatchSearchReq>) -> Json<Vec<Vec<SearchResult>>> {
    {
        let mut engine = state.engine.write().await;
        if engine.unindexed_ids.len() > 1000 || (engine.indexed_segment.is_none() && !engine.unindexed_ids.is_empty()) {
            engine.build_index();
        }
    }
    let engine = state.engine.read().await;
    let results = engine.search_batch(&req.vectors, req.top_k, req.params);
    Json(results)
}

async fn search_batch_points_bin(State(state): State<AppState>, body: axum::body::Bytes) -> axum::response::Response {
    use std::convert::TryInto;
    let bytes = body.to_vec();
    if bytes.len() < 18 {
        return axum::response::Response::builder()
            .status(400)
            .body(axum::body::Body::from("Invalid binary payload length"))
            .unwrap();
    }
    
    let n_queries = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let top_k = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let n_probe = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let rerank = bytes[12] != 0;
    let rerank_factor = u32::from_le_bytes(bytes[13..17].try_into().unwrap()) as usize;
    let with_vector = bytes[17] != 0;

    let float_bytes_start = 18;
    let expected_floats_len = n_queries * 384 * 4;
    if bytes.len() < float_bytes_start + expected_floats_len {
        return axum::response::Response::builder()
            .status(400)
            .body(axum::body::Body::from("Binary payload truncated"))
            .unwrap();
    }

    let mut vectors = Vec::with_capacity(n_queries);
    let mut offset = float_bytes_start;
    for _ in 0..n_queries {
        let mut query_vec = vec![0.0f32; 384];
        for i in 0..384 {
            query_vec[i] = f32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
            offset += 4;
        }
        vectors.push(query_vec);
    }

    {
        let mut engine = state.engine.write().await;
        if engine.unindexed_ids.len() > 1000 || (engine.indexed_segment.is_none() && !engine.unindexed_ids.is_empty()) {
            engine.build_index();
        }
    }

    let engine = state.engine.read().await;
    let search_params = Some(SearchParams {
        exact: Some(false),
        n_probe: Some(n_probe),
        rerank: Some(rerank),
        rerank_factor: Some(rerank_factor),
        with_vector: Some(with_vector),
    });
    
    let results = engine.search_batch(&vectors, top_k, search_params);

    let mut response_bytes = Vec::new();
    for q_res in results {
        let num_res = q_res.len() as u32;
        response_bytes.extend_from_slice(&num_res.to_le_bytes());
        for res in q_res {
            response_bytes.extend_from_slice(&res.id.to_le_bytes());
            response_bytes.extend_from_slice(&res.score.to_le_bytes());
            if let Some(vec) = res.vector {
                response_bytes.push(1);
                for val in vec {
                    response_bytes.extend_from_slice(&val.to_le_bytes());
                }
            } else {
                response_bytes.push(0);
            }
        }
    }

    axum::response::Response::builder()
        .header("content-type", "application/octet-stream")
        .body(axum::body::Body::from(response_bytes))
        .unwrap()
}

async fn delete_point(State(state): State<AppState>, Path(id): Path<u64>) -> Json<&'static str> {
    let mut engine = state.engine.write().await;
    engine.delete(id);
    Json("Deleted")
}

async fn clear_points(State(state): State<AppState>) -> Json<&'static str> {
    let mut engine = state.engine.write().await;
    engine.clear();
    Json("Cleared All Data")
}

async fn save_points(State(state): State<AppState>) -> Json<&'static str> {
    let engine = state.engine.read().await;
    
    // Đảm bảo thư mục data/ tồn tại
    if let Err(e) = std::fs::create_dir_all("data/tq_index") {
        eprintln!("❌ Lỗi khi tạo thư mục data/tq_index: {}", e);
        return Json("Failed to create data directory");
    }
    
    match engine.save_to_disk("data/tq_index") {
        Ok(_) => Json("Saved successfully to disk"),
        Err(e) => {
            eprintln!("❌ Lỗi khi lưu đĩa: {}", e);
            Json("Failed to save to disk")
        }
    }
}

async fn configure_index(State(state): State<AppState>, Json(req): Json<ConfigReq>) -> Json<&'static str> {
    let mut engine = state.engine.write().await;
    engine.index_config.n_list = req.n_list;
    engine.index_config.quantize_bits = req.quantize_bits;
    if req.max_training_samples.is_some() {
        engine.index_config.max_training_samples = req.max_training_samples;
    }
    
    // Đánh sập Index cũ và đẩy toàn bộ dữ liệu vào Buffer để ép Build lại với cấu hình mới
    engine.reset_index();
    
    Json("Index configured successfully")
}

async fn list_points(State(state): State<AppState>) -> Json<Vec<PointDetail>> {
    let engine = state.engine.read().await;
    Json(engine.get_all())
}
