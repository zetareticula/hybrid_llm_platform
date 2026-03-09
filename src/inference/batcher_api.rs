use std::sync::Arc;
use tokio::sync::oneshot;
use crate::inference::batcher::{DynamicBatcher, InferenceRequest};
use crate::api::{Message, ChatResponse};

pub async fn handle_chat_with_batcher(
    batcher: Arc<DynamicBatcher>,
    model: String,
    backend: String,
    quantum_provider: Option<String>,
    messages: Vec<Message>,
) -> ChatResponse {
    let backend_enum = match backend.as_str() {
        "gpu" => crate::inference::scheduler::Backend::GPU,
        "cpu" => crate::inference::scheduler::Backend::CPU,
        _ => crate::inference::scheduler::Backend::Hybrid,
    };

    let (tx, rx) = oneshot::channel();
    let req = InferenceRequest {
        model_name: model,
        backend: backend_enum,
        quantum_provider,
        messages,
        responder: tx,
    };

    batcher.push_request(req).await;

    let token_stream = rx.await.unwrap_or_default();
    ChatResponse { token_stream }
}