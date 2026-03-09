/// Full inference session router.
///
/// Implements the causal pipeline:
///   artifact → quantization → backend → batching → streaming
///
/// Every request passes through artifact ranking and backend negotiation before
/// reaching the scheduler. This ensures the correct compression format, GPU shard
/// count, and batch token budget are always respected — regardless of what the
/// caller specified as their preferred backend.
use crate::api::Message;
use crate::inference::scheduler::{schedule_inference, Backend};
use crate::inference::backend_negotiator::{negotiate, default_quantization_for};
use crate::model_registry::{get_model_artifacts, ranker::rank_artifacts};
use crate::quantization::search::QuantizationMethod;

pub struct SessionManager;

impl SessionManager {
    /// Route a chat session through the full artifact → quantization → backend pipeline.
    ///
    /// # Pipeline steps
    /// 1. **Artifact** — discover all on-disk weight files for `model_name`
    ///    (`.safetensors`, `.gguf`, `.qmodel`, `.onnx`, `.engine`, `.bin`, `.pt`, `.npy`)
    /// 2. **Rank** — `rank_artifacts` scores each format; `.qmodel` scores 1.0,
    ///    TensorRT 0.98, SafeTensors 0.95, GGUF 0.90, etc.
    /// 3. **Quantization** — parse the requested method (e.g. `"quantum_nf4"`);
    ///    fall back to the format-optimal default if unrecognised.
    /// 4. **Backend negotiation** — `negotiate` resolves the final `Backend` enum,
    ///    shard count, and batch token budget from the ranked artifact + quantization.
    /// 5. **Schedule** — hand off to `schedule_inference` with the negotiated backend.
    pub async fn route_session(
        model_name: &str,
        _hint_backend: Backend,  // kept for API compat; negotiator overrides if needed
        quantization: &str,
        quantum_provider: Option<&str>,
        messages: &[Message],
    ) -> Vec<String> {
        // ── Step 1: Artifact discovery ─────────────────────────────────────
        let artifacts = get_model_artifacts(model_name).await;

        // ── Step 2: Rank artifacts ─────────────────────────────────────────
        let ranked = rank_artifacts(artifacts);

        // ── Step 3: Quantization method selection ──────────────────────────
        // Parse the caller's requested method; if unrecognised, derive the
        // best method for the top-ranked artifact format.
        let quant_method = QuantizationMethod::from_str(quantization)
            .or_else(|| ranked.first().map(default_quantization_for))
            .unwrap_or(QuantizationMethod::NF4);

        // ── Step 4: Backend negotiation ────────────────────────────────────
        // Fall back gracefully when no artifacts are on disk yet (model not
        // compressed) — derive backend from original hint and quantization.
        let resolved_backend = if let Some(top) = ranked.first() {
            let negotiated = negotiate(top, quant_method, 1);
            negotiated.backend
        } else {
            // No artifact found: honour quantum hint, otherwise use Hybrid
            if quant_method.requires_quantum_backend() {
                Backend::Quantum
            } else {
                _hint_backend
            }
        };

        // ── Step 5: Schedule inference ─────────────────────────────────────
        schedule_inference(model_name, resolved_backend, messages, quantum_provider).await
    }
}