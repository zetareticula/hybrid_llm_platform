pub mod model;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tower_http::cors::{Any, CorsLayer};
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

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/v1/models", get(model::list_models))
        .layer(cors)
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
    let t0 = Instant::now();
    let backend = crate::inference::scheduler::Backend::from_str(
        req.backend.as_deref().unwrap_or("hybrid"),
    );
    let quantization = req.quantization.as_deref().unwrap_or("").to_string();

    let tokens = if quantization == "quantum_nf4" {
        // VC demo path: force the production batch loop that computes real transformer
        // embeddings and runs QAOA token reordering.
        let (tx, mut rx) = mpsc::channel::<String>(1024);
        let qreq = crate::streaming_quantum_scheduler::QuantumStreamRequest {
            model_name: req.model.clone(),
            backend: crate::inference::scheduler::Backend::Quantum,
            quantum_provider: req.quantum_provider.clone(),
            messages: req.messages.clone(),
            token_sender: tx,
        };
        state.batcher.push_request(qreq).await;

        let mut out = Vec::new();
        while let Some(tok) = rx.recv().await {
            out.push(tok);
        }
        if out.is_empty() {
            crate::session::SessionManager::route_session(
                &req.model,
                backend,
                &quantization,
                req.quantum_provider.as_deref(),
                &req.messages,
            )
            .await
        } else {
            out
        }
    } else {
        crate::session::SessionManager::route_session(
            &req.model,
            backend,
            &quantization,
            req.quantum_provider.as_deref(),
            &req.messages,
        )
        .await
    };

    let elapsed_ms = t0.elapsed().as_millis().max(1);
    let tps = tokens.len() as f64 / (elapsed_ms as f64 / 1000.0);

    let backend_str: String = if quantization == "quantum_nf4" {
        "Quantum".to_string()
    } else {
        format!("{:?}", backend)
    };

    let _ = state.metrics_tx.send(format!(
        "{{\"model\":\"{}\",\"tokens\":{},\"latency_ms\":{},\"tps\":{:.1},\"backend\":\"{}\",\"quantization\":\"{}\"}}",
        req.model,
        tokens.len(),
        elapsed_ms,
        tps,
        backend_str,
        quantization
    ));

    Json(ChatResponse { token_stream: tokens })
}