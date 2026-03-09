pub mod model;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use crate::streaming_quantum_scheduler::QuantumHybridBatcher;

// ── Shared types used across the crate ─────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Wire format for POST /v1/chat/completions.
/// Matches the example request:
/// { "model": "...", "backend": "hybrid", "quantization": "quantum_nf4",
///   "quantum_provider": "ibm", "messages": [...] }
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub backend: Option<String>,
    pub quantization: Option<String>,
    pub quantum_provider: Option<String>,
    pub messages: Vec<Message>,
}

#[derive(Debug, Serialize, Default)]
pub struct ChatResponse {
    pub token_stream: Vec<String>,
}

// ── Axum application state ──────────────────────────────────────────────────

pub struct AppState {
    pub metrics_tx: broadcast::Sender<String>,
    pub batcher: Arc<QuantumHybridBatcher>,
}

// ── Server entry-point ──────────────────────────────────────────────────────

pub async fn start_server(
    metrics_tx: broadcast::Sender<String>,
    batcher: Arc<QuantumHybridBatcher>,
) {
    let state = Arc::new(AppState { metrics_tx, batcher });

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/v1/models", get(model::list_models))
        .with_state(state);

    axum::Server::bind(&"0.0.0.0:8080".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

// ── Route handlers ──────────────────────────────────────────────────────────

async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let backend = crate::inference::scheduler::Backend::from_str(
        req.backend.as_deref().unwrap_or("hybrid"),
    );
    let quantization = req.quantization.as_deref().unwrap_or("").to_string();

    let tokens = crate::session::SessionManager::route_session(
        &req.model,
        backend,
        &quantization,
        req.quantum_provider.as_deref(),
        &req.messages,
    )
    .await;

    let _ = state.metrics_tx.send(format!(
        "{{\"model\":\"{}\",\"tokens\":{}}}",
        req.model,
        tokens.len()
    ));

    Json(ChatResponse { token_stream: tokens })
}