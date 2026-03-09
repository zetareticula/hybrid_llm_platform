use std::collections::HashMap;
use tokio::sync::Mutex;
use std::sync::Arc;

#[derive(Clone)]
pub struct GpuShard {
    pub model_name: String,
    pub gpu_id: usize,
    pub max_batch_tokens: usize,
    pub current_load: usize,
}

pub struct GpuManager {
    shards: Arc<Mutex<HashMap<String, Vec<GpuShard>>>>,
}

impl GpuManager {
    pub fn new() -> Self {
        Self {
            shards: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn register_shard(&self, model_name: String, gpu_id: usize, max_batch_tokens: usize) {
        let mut shards = self.shards.lock().await;
        shards.entry(model_name)
            .or_default()
            .push(GpuShard {
                model_name,
                gpu_id,
                max_batch_tokens,
                current_load: 0,
            });
    }

    /// Select GPU shard based on memory and load
    pub async fn select_shard(&self, model_name: &str, request_tokens: usize) -> Option<GpuShard> {
        let mut shards = self.shards.lock().await;
        if let Some(shard_list) = shards.get_mut(model_name) {
            // Find shard with enough available batch space
            shard_list.sort_by_key(|s| s.current_load);
            for shard in shard_list.iter_mut() {
                if shard.current_load + request_tokens <= shard.max_batch_tokens {
                    shard.current_load += request_tokens;
                    return Some(shard.clone());
                }
            }
            // No shard has enough memory, pick least-loaded and let it split batch
            shard_list.first_mut().map(|s| {
                s.current_load += request_tokens;
                s.clone()
            })
        } else { None }
    }

    pub async fn release_load(&self, model_name: &str, gpu_id: usize, tokens: usize) {
        let mut shards = self.shards.lock().await;
        if let Some(shard_list) = shards.get_mut(model_name) {
            if let Some(shard) = shard_list.iter_mut().find(|s| s.gpu_id == gpu_id) {
                shard.current_load = shard.current_load.saturating_sub(tokens);
            }
        }
    }
}