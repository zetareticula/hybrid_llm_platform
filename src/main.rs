use hybrid_llm_platform::metrics_ws::{init_metrics_channel, start_ws_server};
use hybrid_llm_platform::streaming_quantum_scheduler::QuantumHybridBatcher;
use hybrid_llm_platform::gpu_shard::GpuManager;
use hybrid_llm_platform::api;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let metrics_tx = init_metrics_channel();
    
    // GPU shard manager
    let gpu_mgr = Arc::new(GpuManager::new());
    gpu_mgr.register_shard("llama-3-7b".into(), 0, 2000).await;
    
    // Scheduler
    let batcher = Arc::new(QuantumHybridBatcher::new(20, gpu_mgr.clone()));
    batcher.clone().start().await;
    
    // WebSocket metrics (clone sender — broadcast::Sender is Clone)
    tokio::spawn(start_ws_server(metrics_tx.clone()));

    // Start REST / WS API
    api::start_server(metrics_tx, batcher).await;
}