use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio::task;
use crate::api::Message;
use crate::inference::gpu::gpu_infer;
use crate::inference::cpu::cpu_infer;
use crate::quantization::quantum::quantum_infer_blockwise;

/// Execution backend for a given inference request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    GPU,
    CPU,
    Hybrid,
    Quantum,
}

impl Backend {
    pub fn from_str(s: &str) -> Self {
        match s {
            "gpu"     => Backend::GPU,
            "cpu"     => Backend::CPU,
            "quantum" => Backend::Quantum,
            _         => Backend::Hybrid,
        }
    }
}

/// Batch-oriented inference: returns a complete token list.
/// GPU and Quantum paths run concurrently in Hybrid mode (from parallel_scheduler.rs).
pub async fn schedule_inference(
    model_name: &str,
    backend: Backend,
    messages: &[Message],
    provider: Option<&str>,
) -> Vec<String> {
    let model_path = resolve_model_path(model_name);

    match backend {
        Backend::GPU => gpu_infer(&model_path, messages).await,
        Backend::CPU => cpu_infer(&model_path, messages).await,
        Backend::Quantum => quantum_infer_blockwise(&model_path, messages, provider).await,
        Backend::Hybrid => {
            let path_gpu = model_path.clone();
            let path_q   = model_path.clone();
            let msgs_gpu: Vec<Message> = messages.to_vec();
            let msgs_q:   Vec<Message> = messages.to_vec();
            let prov = provider.map(|s| s.to_string());

            let gpu_task     = task::spawn(async move { gpu_infer(&path_gpu, &msgs_gpu).await });
            let quantum_task = task::spawn(async move {
                quantum_infer_blockwise(&path_q, &msgs_q, prov.as_deref()).await
            });

            let (gpu_res, q_res) = tokio::join!(gpu_task, quantum_task);
            let mut tokens = gpu_res.unwrap_or_default();
            tokens.extend(q_res.unwrap_or_default());
            tokens
        }
    }
}

/// Streaming inference: returns a channel that emits tokens as they are produced.
/// Tokens from GPU and Quantum are interleaved in Hybrid mode (from interleave_scheduler.rs).
pub async fn schedule_inference_stream(
    model_name: &str,
    backend: Backend,
    messages: &[Message],
    provider: Option<&str>,
) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(1024);
    let messages_owned = messages.to_vec();
    let model_name_owned = model_name.to_string();
    let provider_owned = provider.map(|s| s.to_string());

    tokio::spawn(async move {
        match backend {
            Backend::GPU => {
                let model_path = resolve_model_path(&model_name_owned);
                for tok in gpu_infer(&model_path, &messages_owned).await {
                    if tx.send(tok).await.is_err() { break; }
                }
            }
            Backend::CPU => {
                let model_path = resolve_model_path(&model_name_owned);
                for tok in cpu_infer(&model_path, &messages_owned).await {
                    if tx.send(tok).await.is_err() { break; }
                }
            }
            Backend::Quantum => {
                let model_path = resolve_model_path(&model_name_owned);
                for tok in quantum_infer_blockwise(&model_path, &messages_owned, provider_owned.as_deref()).await {
                    if tx.send(tok).await.is_err() { break; }
                }
            }
            Backend::Hybrid => {
                let path_gpu = resolve_model_path(&model_name_owned);
                let path_q   = resolve_model_path(&model_name_owned);
                let msgs_gpu = messages_owned.clone();
                let msgs_q   = messages_owned.clone();
                let prov     = provider_owned.clone();

                let gpu_handle = task::spawn(async move { gpu_infer(&path_gpu, &msgs_gpu).await });
                let q_handle   = task::spawn(async move {
                    quantum_infer_blockwise(&path_q, &msgs_q, prov.as_deref()).await
                });

                let (gpu_res, q_res) = tokio::join!(gpu_handle, q_handle);
                let gpu_tokens = gpu_res.unwrap_or_default();
                let q_tokens   = q_res.unwrap_or_default();
                let max_len    = gpu_tokens.len().max(q_tokens.len());

                for i in 0..max_len {
                    if i < gpu_tokens.len() {
                        if tx.send(gpu_tokens[i].clone()).await.is_err() { break; }
                    }
                    if i < q_tokens.len() {
                        if tx.send(q_tokens[i].clone()).await.is_err() { break; }
                    }
                }
            }
        }
    });

    rx
}

/// Build a model path from a name, checking supported formats in priority order.
/// Handles .safetensors, .gguf, .bin, .pt, and .npy weight files.
pub fn resolve_model_path(model_name: &str) -> PathBuf {
    let extensions = ["safetensors", "gguf", "bin", "pt", "npy"];
    for ext in &extensions {
        let candidate = PathBuf::from(format!("models/{}.{}", model_name, ext));
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from(format!("models/{}.bin", model_name))
}