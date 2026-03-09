use crate::inference::gpu::gpu_infer;
use crate::inference::cpu::cpu_infer;
use crate::quantization::quantum::quantum_infer_blockwise;
use crate::api::Message;
use std::path::PathBuf;
use tokio::task;

/// Backend selection
#[derive(Clone, Copy)]
pub enum Backend {
    GPU,
    CPU,
    Hybrid,
}

/// Parallel hybrid scheduling
pub async fn schedule_inference(
    model_name: &str,
    backend: Backend,
    messages: &[Message],
    provider: Option<&str>,
) -> Vec<String> {
    let model_path = PathBuf::from(format!("models/{}.npy", model_name));

    match backend {
        Backend::GPU => gpu_infer(&model_path, messages).await,
        Backend::CPU => cpu_infer(&model_path, messages).await,
        Backend::Hybrid => {
            // Spawn GPU and Quantum inference tasks concurrently
            let gpu_task = task::spawn(gpu_infer(&model_path, messages));
            let quantum_task = task::spawn(quantum_infer_blockwise(&model_path, messages, provider));

            // Wait for both to complete
            let (gpu_tokens, quantum_tokens) = tokio::join!(gpu_task, quantum_task);

            let mut tokens = gpu_tokens.unwrap_or_default();
            let q_tokens = quantum_tokens.unwrap_or_default();

            // Merge streams intelligently (e.g., append or interleave)
            tokens.extend(q_tokens);

            tokens
        }
    }
}