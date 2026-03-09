use crate::api::Message;
use std::path::PathBuf;

pub async fn gpu_infer(_model_path: &PathBuf, messages: &[Message]) -> Vec<String> {
    // Simulate token streaming per message
    messages
        .iter()
        .map(|m| m.content.split_whitespace().map(|t| format!("{}[GPU]", t)).collect::<Vec<_>>().join(" "))
        .collect()
}