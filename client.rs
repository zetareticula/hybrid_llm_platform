use tokio::sync::mpsc;
use crate::streaming_quantum_scheduler::{QuantumHybridBatcher, QuantumStreamRequest};
use crate::gpu_shard::GpuManager;
use crate::api::Message;
use std::sync::Arc;
use crate::inference::scheduler::Backend;

let gpu_manager = Arc::new(GpuManager::new());
gpu_manager.register_shard("llama-3-70b".into(), 0, 2000).await;
gpu_manager.register_shard("llama-3-70b".into(), 1, 2000).await;

let batcher = Arc::new(QuantumHybridBatcher::new(20, gpu_manager.clone()));
batcher.start().await;

let (tx, mut rx) = mpsc::channel(1024);
batcher.push_request(QuantumStreamRequest {
    model_name: "llama-3-70b".into(),
    backend: Backend::Hybrid,
    quantum_provider: Some("ibm".into()),
    messages: vec![Message { role: "user".into(), content: "Hello world from hybrid LLM".into() }],
    token_sender: tx,
}).await;

// Stream tokens to UI
while let Some(token) = rx.recv().await {
    print!("{} ", token);
}