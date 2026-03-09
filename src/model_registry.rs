pub mod artifact;
pub mod ranked_artifact;
pub mod ranker;
pub mod curated_model;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::sync::RwLock;
use once_cell::sync::Lazy;
use anyhow::Result;
use tokio::fs;
use tokio::process::Command;
use crate::model_registry::artifact::ModelArtifact;

/// Supported model file extensions in priority order for the Rust core loader.
const MODEL_EXTENSIONS: &[&str] = &["safetensors", "gguf", "bin", "pt", "npy"];

pub static MODEL_CACHE: Lazy<RwLock<HashMap<String, PathBuf>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Scan `model_dir` for all supported model file formats and populate the cache.
pub async fn init_registry(model_dir: &str) -> Result<()> {
    let mut cache = MODEL_CACHE.write().await;
    let dir = Path::new(model_dir);
    if dir.exists() {
        let mut entries = fs::read_dir(dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if MODEL_EXTENSIONS.contains(&ext) {
                    let model_name = path.file_stem().unwrap().to_string_lossy().to_string();
                    cache.entry(model_name).or_insert(path);
                }
            }
        }
    }
    Ok(())
}

/// Resolve a model path, checking all supported formats. On first access,
/// runs quantum block-wise optimization via Python and caches the result.
pub async fn get_model(name: &str, provider: Option<&str>) -> Result<PathBuf> {
    {
        let cache = MODEL_CACHE.read().await;
        if let Some(path) = cache.get(name) {
            return Ok(path.clone());
        }
    }

    let raw_model_path = find_model_file(name)
        .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in models/ directory", name))?;

    let optimized_path = optimize_model_blockwise(&raw_model_path, provider).await?;
    let mut cache = MODEL_CACHE.write().await;
    cache.insert(name.to_string(), optimized_path.clone());
    Ok(optimized_path)
}

/// Return all on-disk artifacts for a model name as typed ModelArtifact variants.
/// Used by the ranker to select the best backend.
pub async fn get_model_artifacts(name: &str) -> Vec<ModelArtifact> {
    MODEL_EXTENSIONS
        .iter()
        .filter_map(|ext| {
            let p = PathBuf::from(format!("models/{}.{}", name, ext));
            if p.exists() { ModelArtifact::from_path(p) } else { None }
        })
        .collect()
}

/// Try each supported extension and return the first file that exists.
fn find_model_file(name: &str) -> Option<PathBuf> {
    MODEL_EXTENSIONS.iter().find_map(|ext| {
        let p = PathBuf::from(format!("models/{}.{}", name, ext));
        if p.exists() { Some(p) } else { None }
    })
}

/// Invoke python_quantum/optimize_full_model.py to run quantum block-wise compression.
async fn optimize_model_blockwise(raw_model_path: &PathBuf, provider: Option<&str>) -> Result<PathBuf> {
    let provider_arg = provider.unwrap_or("");
    let model_path_str = raw_model_path.to_string_lossy().to_string();
    let output = Command::new("python3")
        .arg("-c")
        .arg(format!(
            "import sys; sys.path.insert(0,'python_quantum'); \
             from optimize_full_model import optimize_full_model; \
             print(optimize_full_model('{}', '{}'))",
            model_path_str, provider_arg
        ))
        .output()
        .await?;

    if !output.status.success() {
        anyhow::bail!(
            "Quantum optimization failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let optimized_path_str = String::from_utf8(output.stdout)?.trim().to_string();
    Ok(PathBuf::from(optimized_path_str))
}