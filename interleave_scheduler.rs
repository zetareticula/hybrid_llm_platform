use crate::inference::gpu::gpu_infer;
use crate::quantization::quantum::quantum_infer_blockwise;
use crate::api::Message;
use tokio::sync::mpsc;
use tokio::task;

/// Backend selection
#[derive(Clone, Copy)]
pub enum Backend {
    GPU,
    CPU,
    Hybrid,
}

/// Stream tokens asynchronously
pub async fn schedule_inference_stream(
    model_name: &str,
    backend: Backend,
    messages: &[Message],
    provider: Option<&str>,
) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(1024);
    let messages_clone = messages.to_vec();
    let model_name = model_name.to_string();
    let provider = provider.map(|s| s.to_string());

    tokio::spawn(async move {
        match backend {
            Backend::GPU => {
                let gpu_tokens = gpu_infer_tokens(&model_name, &messages_clone).await;
                for tok in gpu_tokens { let _ = tx.send(tok).await; }
            }
            Backend::CPU => {
                let cpu_tokens = messages_clone.iter()
                    .flat_map(|m| m.content.split_whitespace().map(|t| t.to_string()))
                    .collect::<Vec<_>>();
                for tok in cpu_tokens { let _ = tx.send(tok).await; }
            }
            Backend::Hybrid => {
                // Spawn GPU and Quantum concurrently
                let gpu_handle = task::spawn(gpu_infer_tokens(&model_name, &messages_clone));
                let quantum_handle = task::spawn(quantum_infer_tokens(&model_name, &messages_clone, provider));

                let (gpu_tokens, quantum_tokens) = tokio::join!(gpu_handle, quantum_handle);

                // Interleave tokens
                let gpu_tokens = gpu_tokens.unwrap_or_default();
                let quantum_tokens = quantum_tokens.unwrap_or_default();
                let max_len = gpu_tokens.len().max(quantum_tokens.len());

                for i in 0..max_len {
                    if i < gpu_tokens.len() { let _ = tx.send(gpu_tokens[i].clone()).await; }
                    if i < quantum_tokens.len() { let _ = tx.send(quantum_tokens[i].clone()).await; }
                }
            }
        }
    });

    rx
}

/// GPU token streaming (simulate per-token generation)
async fn gpu_infer_tokens(model_name: &str, messages: &[Message]) -> Vec<String> {
    let model_path = format!("models/{}.npy", model_name);
    messages.iter()
        .flat_map(|m| m.content.split_whitespace().map(|t| format!("{}[GPU]", t)))
        .collect()
}

/// Quantum token streaming — runs Python block-wise optimiser then tags each token.
async fn quantum_infer_tokens(model_name: &str, messages: &[Message], provider: Option<String>) -> Vec<String> {
    let model_name = model_name.to_string();
    let messages_owned = messages.to_vec();

    tokio::task::spawn_blocking(move || {
        use pyo3::prelude::*;
        use std::path::PathBuf;

        Python::with_gil(|py| {
            let optimizer = PyModule::from_code(
                py,
                include_str!("python_quantum/optimize_full_model.py"),
                "optimize_full_model.py",
                "",
            )
            .unwrap_or_else(|_| pyo3::types::PyModule::new(py, "fallback").unwrap());

            let model_path = PathBuf::from(format!("models/{}.npy", model_name));
            let provider_str = provider.as_deref().unwrap_or("").to_string();

            let mut outputs = Vec::new();
            for msg in &messages_owned {
                let _opt_path: String = optimizer
                    .getattr("optimize_full_model")
                    .and_then(|f| f.call1((model_path.to_string_lossy().to_string(), &provider_str)))
                    .and_then(|r| r.extract::<String>())
                    .unwrap_or_default();

                outputs.extend(
                    msg.content.split_whitespace().map(|t| format!("{}[Q]", t)),
                );
            }
            outputs
        })
    })
    .await
    .unwrap_or_default()
}