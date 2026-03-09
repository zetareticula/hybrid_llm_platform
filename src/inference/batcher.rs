use crate::api::Message;
use crate::inference::scheduler::{schedule_inference, Backend};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{Duration, Instant};

/// A single request for inference
pub struct InferenceRequest {
    pub model_name: String,
    pub backend: Backend,
    pub quantum_provider: Option<String>,
    pub messages: Vec<Message>,
    pub responder: tokio::sync::oneshot::Sender<Vec<String>>,
}

/// Dynamic batching scheduler
pub struct DynamicBatcher {
    queue: Arc<Mutex<Vec<InferenceRequest>>>,
    max_batch_delay: Duration,
}

impl DynamicBatcher {
    pub fn new(max_batch_delay_ms: u64) -> Self {
        Self {
            queue: Arc::new(Mutex::new(Vec::new())),
            max_batch_delay: Duration::from_millis(max_batch_delay_ms),
        }
    }

    /// Push a new request into the batch queue
    pub async fn push_request(&self, req: InferenceRequest) {
        let mut queue = self.queue.lock().await;
        queue.push(req);
    }

    /// Start the batching loop
    pub async fn start(self: Arc<Self>) {
        let queue_ref = self.queue.clone();
        let max_delay = self.max_batch_delay;

        tokio::spawn(async move {
            loop {
                let _start_time = Instant::now();
                tokio::time::sleep(max_delay).await;

                let batch_requests: Vec<InferenceRequest>;
                {
                    let mut queue = queue_ref.lock().await;
                    if queue.is_empty() {
                        continue;
                    }
                    batch_requests = queue.drain(..).collect();
                }

                // Group requests by model + backend for efficiency
                let mut groups: std::collections::HashMap<(String, Backend), Vec<InferenceRequest>> = std::collections::HashMap::new();
                for req in batch_requests {
                    groups.entry((req.model_name.clone(), req.backend))
                        .or_default()
                        .push(req);
                }

                for ((model_name, backend), requests) in groups {
                    let combined_messages: Vec<Message> = requests.iter().flat_map(|r| r.messages.clone()).collect();
                    let provider = requests.get(0).and_then(|r| r.quantum_provider.clone());

                    // Schedule inference for the entire batch
                    let tokens = schedule_inference(&model_name, backend, &combined_messages, provider.as_deref()).await;

                    // Split results back to original requests
                    let mut offset = 0;
                    for req in requests {
                        let len = req.messages.len();
                        let resp_tokens = tokens[offset..offset+len].to_vec();
                        let _ = req.responder.send(resp_tokens);
                        offset += len;
                    }
                }
            }
        });
    }
}