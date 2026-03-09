use crate::api::Message;
use crate::gpu_shard::GpuManager;
use crate::inference::scheduler::{Backend, schedule_inference_stream};
use tokio::sync::mpsc;
use std::sync::Arc;

/// A streaming request with its per-client token sender channel.
pub struct StreamRequest {
    pub model_name: String,
    pub backend: Backend,
    pub quantum_provider: Option<String>,
    pub messages: Vec<Message>,
    pub token_sender: mpsc::Sender<String>,
}

/// Memory-aware dynamic GPU batching with per-shard load tracking.
pub struct GpuBatcher {
    queue: Arc<tokio::sync::Mutex<Vec<StreamRequest>>>,
    max_batch_delay_ms: u64,
    gpu_manager: Arc<GpuManager>,
}

impl GpuBatcher {
    pub fn new(max_batch_delay_ms: u64, gpu_manager: Arc<GpuManager>) -> Self {
        Self {
            queue: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            max_batch_delay_ms,
            gpu_manager,
        }
    }

    pub async fn push_request(&self, req: StreamRequest) {
        let mut q = self.queue.lock().await;
        q.push(req);
    }

    pub async fn start(self: Arc<Self>) {
        let queue_ref = self.queue.clone();
        let gpu_mgr = self.gpu_manager.clone();
        let delay = tokio::time::Duration::from_millis(self.max_batch_delay_ms);

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(delay).await;

                let batch_requests: Vec<StreamRequest>;
                {
                    let mut queue = queue_ref.lock().await;
                    if queue.is_empty() {
                        continue;
                    }
                    batch_requests = queue.drain(..).collect();
                }

                for req in batch_requests {
                    let request_tokens: usize = req
                        .messages
                        .iter()
                        .map(|m| m.content.split_whitespace().count())
                        .sum();

                    if let Some(shard) =
                        gpu_mgr.select_shard(&req.model_name, request_tokens).await
                    {
                        let sender = req.token_sender.clone();
                        let messages = req.messages.clone();
                        let model_name = req.model_name.clone();
                        let provider = req.quantum_provider.clone();
                        let gpu_mgr_inner = gpu_mgr.clone();

                        tokio::spawn(async move {
                            let mut token_stream = schedule_inference_stream(
                                &model_name,
                                req.backend,
                                &messages,
                                provider.as_deref(),
                            )
                            .await;

                            while let Some(token) = token_stream.recv().await {
                                let _ = sender.send(token).await;
                            }

                            gpu_mgr_inner
                                .release_load(&model_name, shard.gpu_id, request_tokens)
                                .await;
                        });
                    } else {
                        eprintln!(
                            "[GpuBatcher] No available shard for model {}",
                            req.model_name
                        );
                    }
                }
            }
        });
    }
}
