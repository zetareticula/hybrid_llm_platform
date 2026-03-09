use crate::api::Message;
use std::path::PathBuf;

pub async fn cpu_infer(_model_path: &PathBuf, messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|m| m.content.split_whitespace().map(|t| format!("{}[CPU]", t)).collect::<Vec<_>>().join(" "))
        .collect()
}