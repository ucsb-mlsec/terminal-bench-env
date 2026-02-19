use axum::{
    Router,
    routing::get,
    Json,
};
use serde::{Serialize, Deserialize};
use std::net::SocketAddr;

#[derive(Serialize, Deserialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[derive(Serialize, Deserialize)]
struct InfoResponse {
    name: String,
    description: String,
}

async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: "1.0.0".to_string(),
    })
}

async fn info() -> Json<InfoResponse> {
    Json(InfoResponse {
        name: "web-server".to_string(),
        description: "A simple web server built with Axum".to_string(),
    })
}

async fn root() -> &'static str {
    "Welcome to the web server!"
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health_check))
        .route("/info", get(info));

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    
    println!("Server starting on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to address");
    
    axum::serve(listener, app)
        .await
        .expect("Server failed to start");
}