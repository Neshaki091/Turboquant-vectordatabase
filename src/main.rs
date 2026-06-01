mod engine;

use axum::{
    routing::{get, post, delete},
    Router, Json, extract::{State, Path},
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
struct SearchReq {
    vector: Vec<f32>,
    top_k: usize,
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
    println!("🚀 Khởi động TurboQuant Server (Phiên bản Độc Lập)...");
    
    let state = AppState {
        engine: Arc::new(RwLock::new(TQEngine::new())),
    };

    // Định tuyến API theo phong cách Qdrant
    let app = Router::new()
        .route("/", get(|| async { axum::response::Redirect::permanent("/dashboard") }))
        .route("/dashboard", get(|| async { axum::response::Html(include_str!("dashboard.html")) }))
        .route("/collections/default/points", get(list_points).post(add_point).delete(clear_points))
        .route("/collections/default/search", post(search_points))
        .route("/collections/default/points/:id", delete(delete_point))
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

async fn search_points(State(state): State<AppState>, Json(req): Json<SearchReq>) -> Json<Vec<SearchResult>> {
    let engine = state.engine.read().await; // RwLock cho phép nhiều truy vấn đọc đồng thời
    let results = engine.search(&req.vector, req.top_k);
    Json(results)
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

async fn list_points(State(state): State<AppState>) -> Json<Vec<PointDetail>> {
    let engine = state.engine.read().await;
    Json(engine.get_all())
}
