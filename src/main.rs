#![allow(dead_code, unused_imports)]

mod api;
mod metrics_ws;
mod streaming_quantum_scheduler;
mod gpu_shard;
mod gpu_batcher;
mod quantization;
mod inference;
mod model_registry;
mod session;
mod scheduler;

use metrics_ws::init_metrics_channel;
use streaming_quantum_scheduler::QuantumHybridBatcher;
use std::sync::Arc;
use gpu_shard::GpuManager;

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
    tokio::spawn(metrics_ws::start_ws_server(metrics_tx.clone()));

    // Start REST / WS API
    api::start_server(metrics_tx, batcher).await;
}