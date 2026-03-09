use tokio::sync::mpsc::Receiver;
use crate::api::Message;
use crate::inference::scheduler::Backend;

/// Stream chat responses token-by-token as they are generated.
/// Derived from interleave_api.rust — corrected extension and module paths.
pub async fn handle_chat_stream(
    model: String,
    backend: String,
    messages: Vec<Message>,
    quantum_provider: Option<String>,
) -> Receiver<String> {
    let backend_enum = Backend::from_str(&backend);

    crate::inference::scheduler::schedule_inference_stream(
        &model,
        backend_enum,
        &messages,
        quantum_provider.as_deref(),
    )
    .await
}
