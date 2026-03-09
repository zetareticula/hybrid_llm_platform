/// Scheduler-ready backend negotiation.
///
/// Compression pipeline position: artifact → quantization → **backend** → batching → streaming
///
/// Takes a post-ranked artifact and a requested quantization method and produces a fully
/// resolved `NegotiatedBackend` — the single authoritative decision point before a request
/// enters the batcher. This enforces the invariant that no request reaches the GPU shards
/// without passing through ranking and quantization selection.
use crate::model_registry::ranked_artifact::{BackendPreference, RankedArtifact};
use crate::inference::scheduler::Backend;
use crate::quantization::search::QuantizationMethod;

/// The resolved inference contract produced by negotiation.
/// Passed directly into `schedule_inference` / `schedule_inference_stream`.
#[derive(Clone, Debug)]
pub struct NegotiatedBackend {
    /// The concrete backend the scheduler will use for this request.
    pub backend: Backend,
    /// The quantization method that was resolved for this request.
    pub quantization: QuantizationMethod,
    /// Number of GPU shards to activate for this request (weight pre-sharding hint).
    pub shard_count: usize,
    /// Recommended token batch size, computed from the compression ratio so larger
    /// compressed models can fit more concurrent requests per shard.
    pub target_batch_tokens: usize,
}

/// Negotiate the backend contract from a ranked artifact and a requested quantization method.
///
/// # Arguments
/// * `ranked`           — Best-ranked artifact for this model (output of `rank_artifacts`)
/// * `quant`            — Quantization method parsed from the API request
/// * `available_shards` — How many GPU shards are currently registered for this model
///
/// # Decision rules
/// 1. Any `QuantumNF4` request is always routed to `Backend::Quantum` regardless of artifact.
/// 2. A `.qmodel` artifact (QModel variant) also forces `Backend::Quantum`.
/// 3. TensorRT / SafeTensors artifacts without quantum quantization use `Backend::GPU`.
/// 4. GGUF artifacts use `Backend::CPU` (they are designed for llama.cpp CPU execution).
/// 5. ONNX artifacts use `Backend::Hybrid`.
/// 6. Shard count scales with model size estimated from file metadata, bounded by available shards.
/// 7. Batch token budget grows with compression ratio — more compressed = more tokens per shard.
pub fn negotiate(
    ranked: &RankedArtifact,
    quant: QuantizationMethod,
    available_shards: usize,
) -> NegotiatedBackend {
    // Rule 1 & 2: quantum quantization method or .qmodel artifact → Quantum backend
    let backend = if quant.requires_quantum_backend()
        || ranked.backend == BackendPreference::Quantum
    {
        Backend::Quantum
    } else {
        match ranked.backend {
            BackendPreference::GPU    => Backend::GPU,
            BackendPreference::CPU    => Backend::CPU,
            BackendPreference::Hybrid => Backend::Hybrid,
            // Should have been caught above, but defensively default to Quantum
            BackendPreference::Quantum => Backend::Quantum,
        }
    };

    // Rule 6: estimate shard need from artifact file size.
    // 7 GB ≈ one 7B-parameter model shard at FP16. Clamp to available_shards.
    let file_bytes = ranked
        .artifact
        .path()
        .metadata()
        .map(|m| m.len())
        .unwrap_or(7_000_000_000); // default: assume 7B model
    let ideal_shards = ((file_bytes / 7_000_000_000) as usize).max(1);
    let shard_count = ideal_shards.min(available_shards.max(1));

    // Rule 7: batch token budget grows with compression — compressed weights fit more in VRAM.
    // Base budget is 512 tokens; QuantumNF4 (8x compression) yields 4096 tokens/batch.
    let target_batch_tokens = (512.0 * quant.compression_ratio()) as usize;

    NegotiatedBackend {
        backend,
        quantization: quant,
        shard_count,
        target_batch_tokens,
    }
}

/// Select the best `QuantizationMethod` for a given ranked artifact when the caller
/// did not specify one. Prefers the highest-compression method the artifact supports.
///
/// `.qmodel` artifacts are already quantum-compressed; re-applying QuantumNF4 would
/// be redundant, so they default to NF4 (storage-compatible, no Python overhead).
pub fn default_quantization_for(ranked: &RankedArtifact) -> QuantizationMethod {
    match ranked.backend {
        BackendPreference::Quantum => QuantizationMethod::NF4,
        BackendPreference::GPU    => QuantizationMethod::NF4,
        BackendPreference::CPU    => QuantizationMethod::GPTQ,
        BackendPreference::Hybrid => QuantizationMethod::AWQ,
    }
}
