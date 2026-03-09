//! Hybrid LLM Platform — public library surface for examples, tests, and integration.
//!
//! Causal inference pipeline:
//!   artifact → quantization → backend → batching → streaming
#![allow(dead_code, unused_imports)]

pub mod api;
pub mod metrics_ws;
pub mod streaming_quantum_scheduler;
pub mod gpu_shard;
pub mod gpu_batcher;
pub mod quantization;
pub mod inference;
pub mod model_registry;
pub mod session;
pub mod scheduler;
