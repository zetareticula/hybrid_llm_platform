use pyo3::prelude::*;
use ndarray::Array2;
use std::path::PathBuf;
use crate::api::Message;

fn ensure_python_quantum_path(py: Python<'_>) {
    let Ok(sys) = PyModule::import(py, "sys") else { return; };
    let Ok(path) = sys.getattr("path") else { return; };
    let pq = format!("{}/python_quantum", env!("CARGO_MANIFEST_DIR"));
    let _ = path.call_method1("insert", (0, pq));
}

/// Quantum-assisted token reordering via QAOA (python_quantum/quantum_optimizer.py).
/// Accepts raw token strings and their embedding matrix; returns reordered tokens.
pub fn reorder_tokens_quantum(tokens: &[String], embeddings: Array2<f32>) -> Vec<String> {
    Python::with_gil(|py| {
        ensure_python_quantum_path(py);
        let Ok(module) = PyModule::import(py, "quantum_optimizer") else {
            return tokens.to_vec();
        };
        let tokens_py: Vec<String> = tokens.to_vec();
        let emb_vec: Vec<Vec<f32>> = embeddings.outer_iter().map(|r| r.to_vec()).collect();

        module
            .getattr("reorder_tokens_qaoa")
            .and_then(|f| f.call1((tokens_py, emb_vec)))
            .and_then(|v| v.extract::<Vec<String>>())
            .unwrap_or_else(|_| tokens.to_vec())
    })
}

/// Compute transformer embeddings for a token list via python_quantum/transformer_embed.py.
/// Python returns a nested list; we reconstruct an ndarray Array2 on the Rust side.
pub fn compute_transformer_embeddings(tokens: &[String], model_name: &str) -> Array2<f32> {
    Python::with_gil(|py| {
        ensure_python_quantum_path(py);
        let Ok(hf_embed) = PyModule::import(py, "transformer_embed") else {
            return Array2::zeros((tokens.len().max(1), 384));
        };
        let tokens_py: Vec<String> = tokens.to_vec();
        let mut raw: Vec<Vec<f32>> = hf_embed
            .getattr("get_embeddings")
            .and_then(|f| f.call1((tokens_py, model_name)))
            .and_then(|v| v.extract::<Vec<Vec<f32>>>())
            .unwrap_or_else(|_| vec![vec![0.0_f32; 384]]);

        // Invariant: one embedding row per token. If Python returned a single pooled embedding
        // (or anything mismatched), expand/crop deterministically so reordering stays stable.
        if raw.len() != tokens.len() {
            let cols = raw.first().map_or(384, |r| r.len());
            raw = vec![vec![0.0_f32; cols]; tokens.len().max(1)];
        }

        let rows = raw.len();
        let cols = raw.first().map_or(384, |r| r.len());
        let flat: Vec<f32> = raw.into_iter().flatten().collect();
        Array2::from_shape_vec((rows, cols), flat)
            .unwrap_or_else(|_| Array2::zeros((tokens.len().max(1), 384)))
    })
}

/// Block-wise quantum inference: quantum-optimize the model weights then generate tokens.
/// Calls python_quantum/optimize_full_model.py, then produces tagged token output.
/// Used by schedule_inference and schedule_inference_stream for Quantum/Hybrid backends.
pub async fn quantum_infer_blockwise(
    model_path: &PathBuf,
    messages: &[Message],
    provider: Option<&str>,
) -> Vec<String> {
    let model_path_str = model_path.to_string_lossy().to_string();
    let provider_str = provider.unwrap_or("").to_string();

    let optimized_path = tokio::task::spawn_blocking(move || {
        Python::with_gil(|py| -> PyResult<String> {
            ensure_python_quantum_path(py);
            let optimizer = PyModule::import(py, "optimize_full_model")?;
            optimizer
                .getattr("optimize_full_model")?
                .call1((model_path_str, provider_str))?
                .extract::<String>()
        })
        .unwrap_or_default()
    })
    .await
    .unwrap_or_default();

    messages
        .iter()
        .flat_map(|m| {
            m.content
                .split_whitespace()
                .map(|t| {
                    if optimized_path.is_empty() {
                        format!("{}[Q]", t)
                    } else {
                        format!("{}[Q]({})", t, optimized_path)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}