use tokio::sync::{mpsc, Mutex};
use std::collections::VecDeque;
use std::sync::Arc;
use crate::quantization::{compute_transformer_embeddings, reorder_tokens_quantum};
use tokio::time::{sleep, Duration};

#[derive(Clone)]
pub struct QuantumStreamRequest {
    pub model_name: String,
    pub messages: Vec<String>,
    pub token_sender: mpsc::Sender<String>,
}

pub struct QuantumHybridBatcher {
    pub batch_size: usize,
    pub queue: Arc<Mutex<VecDeque<QuantumStreamRequest>>>,
}

impl QuantumHybridBatcher {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Push a new request into the queue
    pub async fn push_request(&self, req: QuantumStreamRequest) {
        let mut queue = self.queue.lock().await;
        queue.push_back(req);
    }

    /// Start the production batch processing loop
    pub async fn start(self: Arc<Self>) {
        let queue_clone = self.queue.clone();
        tokio::spawn(async move {
            loop {
                let mut batch = Vec::new();
                {
                    let mut queue = queue_clone.lock().await;
                    while batch.len() < self.batch_size && !queue.is_empty() {
                        if let Some(req) = queue.pop_front() {
                            batch.push(req);
                        }
                    }
                }

                if !batch.is_empty() {
                    // Process batch in parallel
                    let mut handles = vec![];
                    for req in batch {
                        handles.push(tokio::spawn(Self::process_request(req)));
                    }
                    // Wait for all batch requests to finish
                    futures::future::join_all(handles).await;
                } else {
                    // Queue is empty, sleep briefly
                    sleep(Duration::from_millis(10)).await;
                }
            }
        });
    }

    /// Process a single request: compute embeddings, quantum reorder, stream tokens
    async fn process_request(req: QuantumStreamRequest) {
        // Step 1: Compute real transformer embeddings
        let embeddings = compute_transformer_embeddings(&req.messages, &req.model_name);

        // Step 2: Quantum-assisted token reordering
        let optimized_tokens =
            reorder_tokens_quantum(&req.messages, embeddings.to_owned().to_vec());

        // Step 3: Stream tokens to client session
        for tok in optimized_tokens {
            if req.token_sender.send(tok).await.is_err() {
                // Client disconnected
                break;
            }
        }
    }
}