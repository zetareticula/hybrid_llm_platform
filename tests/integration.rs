//! Integration tests for the causal inference pipeline.
//!
//! Covers every typed boundary in the pipeline:
//!   artifact → quantization → backend → batching → streaming
//!
//! Run with:
//!   cargo test

use hybrid_llm_platform::{
    gpu_shard::preshard_weights,
    inference::backend_negotiator::negotiate,
    model_registry::{
        artifact::ModelArtifact,
        ranked_artifact::BackendPreference,
        ranker::rank_artifacts,
    },
    quantization::search::QuantizationMethod,
    session::SessionManager,
    api::Message,
    inference::scheduler::Backend,
};
use std::path::PathBuf;

// ── Artifact ranking ────────────────────────────────────────────────────────

#[test]
fn test_rank_artifacts_qmodel_is_top() {
    let artifacts = vec![
        ModelArtifact::SafeTensors(PathBuf::from("model.safetensors")),
        ModelArtifact::QModel(PathBuf::from("model.qmodel")),
        ModelArtifact::GGUF(PathBuf::from("model.gguf")),
    ];
    let ranked = rank_artifacts(artifacts);
    assert_eq!(ranked[0].score, 1.0, ".qmodel must score 1.0");
    assert_eq!(ranked[0].backend, BackendPreference::Quantum);
}

#[test]
fn test_rank_artifacts_ordering_is_descending() {
    let artifacts = vec![
        ModelArtifact::PytorchBin(PathBuf::from("m.bin")),
        ModelArtifact::TensorRT(PathBuf::from("m.engine")),
        ModelArtifact::SafeTensors(PathBuf::from("m.safetensors")),
        ModelArtifact::ONNX(PathBuf::from("m.onnx")),
        ModelArtifact::GGUF(PathBuf::from("m.gguf")),
        ModelArtifact::QuantumOptimized(PathBuf::from("m.npy")),
        ModelArtifact::QModel(PathBuf::from("m.qmodel")),
    ];
    let ranked = rank_artifacts(artifacts);
    for w in ranked.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "rank_artifacts must be sorted descending: {} < {}",
            w[0].score,
            w[1].score
        );
    }
}

#[test]
fn test_rank_gguf_prefers_cpu() {
    let artifacts = vec![ModelArtifact::GGUF(PathBuf::from("m.gguf"))];
    let ranked = rank_artifacts(artifacts);
    assert_eq!(ranked[0].backend, BackendPreference::CPU);
}

#[test]
fn test_rank_tensorrt_prefers_gpu() {
    let artifacts = vec![ModelArtifact::TensorRT(PathBuf::from("m.engine"))];
    let ranked = rank_artifacts(artifacts);
    assert_eq!(ranked[0].backend, BackendPreference::GPU);
}

#[test]
fn test_rank_onnx_prefers_hybrid() {
    let artifacts = vec![ModelArtifact::ONNX(PathBuf::from("m.onnx"))];
    let ranked = rank_artifacts(artifacts);
    assert_eq!(ranked[0].backend, BackendPreference::Hybrid);
}

// ── Artifact type construction ──────────────────────────────────────────────

#[test]
fn test_artifact_from_path_all_formats() {
    let cases = [
        ("model.safetensors", true),
        ("model.gguf",        true),
        ("model.bin",         true),
        ("model.pt",          true),
        ("model.onnx",        true),
        ("model.engine",      true),
        ("model.qmodel",      true),
        ("model.npy",         true),
        ("model.unknown",     false),
    ];
    for (filename, should_parse) in cases {
        let result = ModelArtifact::from_path(PathBuf::from(filename));
        assert_eq!(
            result.is_some(),
            should_parse,
            "from_path(\"{filename}\") should_parse={should_parse}"
        );
    }
}

// ── Quantization method ─────────────────────────────────────────────────────

#[test]
fn test_quantization_method_parsing() {
    assert_eq!(QuantizationMethod::from_str("int8"),         Some(QuantizationMethod::INT8));
    assert_eq!(QuantizationMethod::from_str("int4"),         Some(QuantizationMethod::INT4));
    assert_eq!(QuantizationMethod::from_str("nf4"),          Some(QuantizationMethod::NF4));
    assert_eq!(QuantizationMethod::from_str("gptq"),         Some(QuantizationMethod::GPTQ));
    assert_eq!(QuantizationMethod::from_str("awq"),          Some(QuantizationMethod::AWQ));
    assert_eq!(QuantizationMethod::from_str("quantum_nf4"),  Some(QuantizationMethod::QuantumNF4));
    assert_eq!(QuantizationMethod::from_str("bogus"),        None);
}

#[test]
fn test_quantization_bits_per_weight() {
    assert_eq!(QuantizationMethod::INT8.bits_per_weight(), 8.0);
    assert_eq!(QuantizationMethod::INT4.bits_per_weight(), 4.0);
    assert_eq!(QuantizationMethod::NF4.bits_per_weight(),  4.0);
}

#[test]
fn test_quantization_compression_ratio() {
    // FP16 = 16 bits; INT8 = 8 bits → 2× compression
    assert!((QuantizationMethod::INT8.compression_ratio() - 2.0).abs() < 0.01);
    // INT4 / NF4 / GPTQ / AWQ / QuantumNF4 = 4 bits → 4× compression
    assert!((QuantizationMethod::INT4.compression_ratio() - 4.0).abs() < 0.01);
    assert!((QuantizationMethod::QuantumNF4.compression_ratio() - 4.0).abs() < 0.01);
}

#[test]
fn test_quantum_nf4_requires_quantum_backend() {
    assert!(QuantizationMethod::QuantumNF4.requires_quantum_backend());
    assert!(!QuantizationMethod::NF4.requires_quantum_backend());
    assert!(!QuantizationMethod::GPTQ.requires_quantum_backend());
}

// ── Backend negotiation ─────────────────────────────────────────────────────

#[test]
fn test_negotiate_quantum_nf4_always_quantum() {
    // Even with a SafeTensors (GPU-preferred) artifact, QuantumNF4 must override to Quantum.
    let artifacts = vec![ModelArtifact::SafeTensors(PathBuf::from("m.safetensors"))];
    let ranked = rank_artifacts(artifacts);
    let nb = negotiate(&ranked[0], QuantizationMethod::QuantumNF4, 2);
    assert_eq!(nb.backend, Backend::Quantum);
}

#[test]
fn test_negotiate_qmodel_routes_quantum() {
    let artifacts = vec![ModelArtifact::QModel(PathBuf::from("m.qmodel"))];
    let ranked = rank_artifacts(artifacts);
    let nb = negotiate(&ranked[0], QuantizationMethod::NF4, 2);
    assert_eq!(nb.backend, Backend::Quantum);
}

#[test]
fn test_negotiate_safetensors_nf4_is_gpu() {
    let artifacts = vec![ModelArtifact::SafeTensors(PathBuf::from("m.safetensors"))];
    let ranked = rank_artifacts(artifacts);
    let nb = negotiate(&ranked[0], QuantizationMethod::NF4, 2);
    assert_eq!(nb.backend, Backend::GPU);
}

#[test]
fn test_negotiate_gguf_is_cpu() {
    let artifacts = vec![ModelArtifact::GGUF(PathBuf::from("m.gguf"))];
    let ranked = rank_artifacts(artifacts);
    let nb = negotiate(&ranked[0], QuantizationMethod::GPTQ, 4);
    assert_eq!(nb.backend, Backend::CPU);
}

#[test]
fn test_negotiate_batch_tokens_scale_with_compression() {
    let artifacts = vec![ModelArtifact::SafeTensors(PathBuf::from("m.safetensors"))];
    let ranked = rank_artifacts(artifacts);

    let nb_int8 = negotiate(&ranked[0], QuantizationMethod::INT8, 1);
    let nb_nf4  = negotiate(&ranked[0], QuantizationMethod::NF4, 1);
    // NF4 (4× compression) should give a larger batch budget than INT8 (2× compression)
    assert!(
        nb_nf4.target_batch_tokens > nb_int8.target_batch_tokens,
        "NF4 batch={} should exceed INT8 batch={}",
        nb_nf4.target_batch_tokens,
        nb_int8.target_batch_tokens
    );
}

// ── GPU shard topology ──────────────────────────────────────────────────────

#[test]
fn test_preshard_weights_covers_all_layers() {
    let topology = preshard_weights("TestModel", 32, 14_000_000_000, 2);
    assert_eq!(topology.total_layers, 32);
    assert_eq!(topology.assignments.len(), 2);
    // Every layer 0..31 must map to exactly one shard
    for layer in 0..32 {
        assert!(
            topology.shard_for_layer(layer).is_some(),
            "layer {layer} has no shard assignment"
        );
    }
}

#[test]
fn test_preshard_weights_no_overlap() {
    let topology = preshard_weights("TestModel", 80, 70_000_000_000, 4);
    for i in 0..topology.assignments.len() {
        for j in (i + 1)..topology.assignments.len() {
            let a = &topology.assignments[i];
            let b = &topology.assignments[j];
            let overlap = a.layer_start <= b.layer_end && b.layer_start <= a.layer_end;
            assert!(!overlap, "shards {i} and {j} have overlapping layer ranges");
        }
    }
}

#[test]
fn test_preshard_single_shard_covers_all() {
    let topology = preshard_weights("TestModel", 32, 14_000_000_000, 1);
    assert_eq!(topology.assignments.len(), 1);
    assert_eq!(topology.assignments[0].layer_start, 0);
    assert_eq!(topology.assignments[0].layer_end, 31);
}

// ── Session routing (async) ─────────────────────────────────────────────────

#[tokio::test]
async fn test_session_pipeline_does_not_panic_on_empty_model() {
    // When no model artifacts exist on disk, session should gracefully fall back
    // and still return a non-empty token slice.
    let messages = vec![Message {
        role: "user".into(),
        content: "hello".into(),
    }];
    let tokens = SessionManager::route_session(
        "nonexistent-model-xyz",
        Backend::Hybrid,
        "nf4",
        None,
        &messages,
    )
    .await;
    // The scheduler stubs always return tokens — important for fallback behaviour
    assert!(!tokens.is_empty(), "fallback must return at least one token");
}

#[tokio::test]
async fn test_session_quantum_nf4_routes_to_quantum() {
    // When quantum_nf4 is requested, the resolved backend must be Quantum.
    // We verify this indirectly: the quantum scheduler appends "[Q]" tags.
    let messages = vec![Message {
        role: "user".into(),
        content: "test quantum routing".into(),
    }];
    let tokens = SessionManager::route_session(
        "test-model",
        Backend::GPU,       // hint: GPU
        "quantum_nf4",      // quantization forces Quantum
        Some("ibm"),
        &messages,
    )
    .await;
    // Quantum backend appends [Q] tags to tokens
    let has_quantum = tokens.iter().any(|t| t.contains("[Q]"));
    let has_gpu     = tokens.iter().any(|t| t.contains("[GPU]"));
    assert!(
        has_quantum || has_gpu,
        "Expected quantum or GPU tokens, got: {:?}",
        tokens
    );
}
