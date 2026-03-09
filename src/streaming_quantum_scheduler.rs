use tokio::sync::{mpsc, Mutex};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use crate::api::Message;
use crate::gpu_shard::GpuManager;
use crate::inference::scheduler::Backend;
use crate::quantization::quantum::{compute_transformer_embeddings, reorder_tokens_quantum};

/// A single streaming inference request routed through the quantum hybrid batcher.
#[derive(Clone)]
pub struct QuantumStreamRequest {
    pub model_name: String,
    pub backend: Backend,
    pub quantum_provider: Option<String>,
    pub messages: Vec<Message>,
    pub token_sender: mpsc::Sender<String>,
}

/// Batches requests and processes them with quantum-assisted token reordering.
pub struct QuantumHybridBatcher {
    pub batch_size: usize,
    pub queue: Arc<Mutex<VecDeque<QuantumStreamRequest>>>,
    pub gpu_manager: Arc<GpuManager>,
}

impl QuantumHybridBatcher {
    pub fn new(batch_size: usize, gpu_manager: Arc<GpuManager>) -> Self {
        Self {
            batch_size,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            gpu_manager,
        }
    }

    /// Push a new request into the queue.
    pub async fn push_request(&self, req: QuantumStreamRequest) {
        let mut queue = self.queue.lock().await;
        queue.push_back(req);
    }

    /// Start the production batch processing loop.
    pub async fn start(self: Arc<Self>) {
        let queue_clone = self.queue.clone();
        let batch_size = self.batch_size;

        tokio::spawn(async move {
            loop {
                let mut batch = Vec::new();
                {
                    let mut queue = queue_clone.lock().await;
                    while batch.len() < batch_size && !queue.is_empty() {
                        if let Some(req) = queue.pop_front() {
                            batch.push(req);
                        }
                    }
                }

                if !batch.is_empty() {
                    let mut handles = vec![];
                    for req in batch {
                        handles.push(tokio::spawn(Self::process_request(req)));
                    }
                    futures::future::join_all(handles).await;
                } else {
                    sleep(Duration::from_millis(10)).await;
                }
            }
        });
    }

    /// Process a single request: compute embeddings, quantum-reorder tokens, stream output.
    async fn process_request(req: QuantumStreamRequest) {
        let tokens: Vec<String> = req
            .messages
            .iter()
            .flat_map(|m| m.content.split_whitespace().map(|t| t.to_string()))
            .collect();

        let embeddings = tokio::task::spawn_blocking({
            let tokens = tokens.clone();
            let model = req.model_name.clone();
            move || compute_transformer_embeddings(&tokens, &model)
        })
        .await
        .unwrap_or_else(|_| ndarray::Array2::zeros((1, 384)));

        let optimized_tokens = tokio::task::spawn_blocking(move || {
            reorder_tokens_quantum(&tokens, embeddings)
        })
        .await
        .unwrap_or_default();

        for tok in optimized_tokens {
            if req.token_sender.send(tok).await.is_err() {
                break;
            }
        }
    }
}
