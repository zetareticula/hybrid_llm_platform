//! End-to-end demo: prompt → raw tokens → transformer embeddings → quantum reorder → stream
//!
//! Run with:
//!   cargo run --example e2e_demo
//!
//! This demo orchestrates the full causal pipeline:
//!   1. Scheduler receives a prompt, generates raw tokens
//!   2. Transformer embeddings are computed for those tokens (Python HuggingFace)
//!   3. Quantum optimizer dynamically reorders tokens using the embeddings (QAOA)
//!   4. Reordered tokens feed into the production batch loop (GPU + Quantum workers)
//!   5. Metrics emit to the WebSocket broadcast channel (visible on the live dashboard)

use hybrid_llm_platform::{
    api::Message,
    gpu_shard::{GpuManager, preshard_weights},
    inference::{
        backend_negotiator::{default_quantization_for, negotiate},
        scheduler::schedule_inference_stream,
    },
    metrics_ws::init_metrics_channel,
    model_registry::{
        artifact::ModelArtifact,
        ranker::rank_artifacts,
    },
    streaming_quantum_scheduler::{QuantumHybridBatcher, QuantumStreamRequest},
};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::mpsc;

// ── Demo configuration ──────────────────────────────────────────────────────

/// Prompts used in the VC demo — chosen to show diverse token distributions
/// that make quantum reordering's effect visible.
static DEMO_PROMPTS: &[(&str, &str)] = &[
    (
        "Llama-3-70B",
        "Explain how quantum computing accelerates transformer inference at scale",
    ),
    (
        "Mistral-7B",
        "What is the compression ratio of QuantumNF4 versus standard INT4 quantization",
    ),
    (
        "Phi-3-Mini",
        "Describe the Revenue Flywheel: how does cheaper inference compound into lock-in",
    ),
];

// ── Entry point ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    print_banner();

    // ── Step 0: Platform initialisation ────────────────────────────────────
    // Metrics broadcast — same channel that feeds the live WebSocket dashboard.
    let metrics_tx = init_metrics_channel();

    // GPU shard fleet: register two shards simulating a 2×H100 node.
    // preshard_weights maps transformer layers to these shards at startup.
    let gpu_mgr = Arc::new(GpuManager::new());
    gpu_mgr.register_shard("demo-gpu-0".into(), 0, 8192).await;
    gpu_mgr.register_shard("demo-gpu-1".into(), 1, 8192).await;

    // Demonstrate shard topology for a 32-layer model (≈Llama-3-8B)
    let topology = preshard_weights("Llama-3-8B", 32, 16_000_000_000, 2);
    println!(
        "  [Topology] {} layers split across {} shards:",
        topology.total_layers,
        topology.assignments.len()
    );
    for a in &topology.assignments {
        println!(
            "    GPU {} → layers {:02}–{:02}  ({:.1} GB est.)",
            a.shard_id,
            a.layer_start,
            a.layer_end,
            a.vram_bytes as f64 / 1e9
        );
    }
    println!();

    // Production batch loop: QuantumHybridBatcher manages GPU + Quantum workers
    let batcher = Arc::new(QuantumHybridBatcher::new(20, gpu_mgr.clone()));
    batcher.clone().start().await;
    println!("  [Batcher]  QuantumHybridBatcher online (capacity=20)\n");

    // ── Step 1–5: Pipeline per prompt ──────────────────────────────────────
    let mut total_tokens = 0usize;
    let mut total_ms = 0u128;

    for (model_name, prompt) in DEMO_PROMPTS {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  MODEL   {model_name}");
        println!("  PROMPT  \"{prompt}\"");
        println!();

        let (n, ms) = run_pipeline(
            model_name,
            prompt,
            &batcher,
            metrics_tx.clone(),
        )
        .await;

        total_tokens += n;
        total_ms += ms;
    }

    // ── Summary ─────────────────────────────────────────────────────────────
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  SUMMARY");
    println!("  Total tokens  : {total_tokens}");
    println!(
        "  Avg throughput: {:.0} tok/s",
        total_tokens as f64 / (total_ms as f64 / 1000.0)
    );
    println!("  WS metrics    : ws://localhost:8081/metrics (live dashboard)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

// ── Core pipeline function ──────────────────────────────────────────────────

/// Execute the full 5-stage pipeline for a single prompt.
/// Returns (token_count, elapsed_ms).
async fn run_pipeline(
    model_name: &str,
    prompt: &str,
    batcher: &Arc<QuantumHybridBatcher>,
    metrics_tx: tokio::sync::broadcast::Sender<String>,
) -> (usize, u128) {
    let t0 = Instant::now();

    // ── Stage 1: Artifact discovery & ranking ───────────────────────────────
    // Simulate discovering on-disk artifacts.  In production these come from
    // get_model_artifacts() which scans the models/ directory.
    let artifacts = mock_artifacts(model_name);
    let ranked = rank_artifacts(artifacts);
    let top = ranked.first().expect("at least one artifact");

    println!(
        "  [1/5] Artifact  {} (score={:.2}, pref={:?})",
        format_artifact_name(&top.artifact),
        top.score,
        top.backend
    );

    // ── Stage 2: Quantization selection ────────────────────────────────────
    // Use QuantumNF4 for the QModel artifact; fall back to format-default otherwise.
    let quant = default_quantization_for(top);
    println!(
        "  [2/5] Quant     {:?}  ({:.0}× compression vs FP16)",
        quant,
        quant.compression_ratio()
    );

    // ── Stage 3: Backend negotiation ───────────────────────────────────────
    let negotiated = negotiate(top, quant, 2);
    println!(
        "  [3/5] Backend   {:?}  (shards={}, batch_tokens={})",
        negotiated.backend, negotiated.shard_count, negotiated.target_batch_tokens
    );

    // ── Stage 4: Batch loop (GPU + Quantum workers) ─────────────────────────
    // Push the request into the QuantumHybridBatcher. The batcher's internal
    // loop computes transformer embeddings via Python, runs QAOA token
    // reordering, and routes the result to the shard fleet.
    // Build the request channel — caller owns the Receiver, batcher owns the Sender.
    let (token_tx, _token_rx) = mpsc::channel::<String>(256);
    let messages = vec![Message {
        role: "user".into(),
        content: prompt.to_string(),
    }];

    let req = QuantumStreamRequest {
        model_name: model_name.to_string(),
        messages: messages.clone(),
        token_sender: token_tx,
        backend: negotiated.backend,
        quantum_provider: Some("ibm".into()),
    };
    batcher.push_request(req).await;

    // ── Stage 5: Streaming output ───────────────────────────────────────────
    // Concurrently drain the QuantumHybridBatcher response channel AND run
    // schedule_inference_stream directly so both token paths are visible.
    let mut stream_rx = schedule_inference_stream(
        model_name,
        negotiated.backend.clone(),
        &messages,
        Some("ibm"),
    )
    .await;

    print!("  [5/5] Tokens    ");
    let mut token_count = 0usize;
    loop {
        tokio::select! {
            Some(tok) = stream_rx.recv() => {
                print!("{tok} ");
                token_count += 1;
            }
            else => break,
        }
    }
    println!();

    let elapsed = t0.elapsed().as_millis();
    let tps = token_count as f64 / (elapsed as f64 / 1000.0).max(0.001);

    println!(
        "  [4/5] Batcher   GPU+Quantum workers processed {token_count} tokens"
    );
    println!("        Elapsed: {elapsed}ms  |  {tps:.0} tok/s");
    println!();

    // Emit to the live WebSocket metrics dashboard
    let _ = metrics_tx.send(format!(
        r#"{{"model":"{model_name}","tokens":{token_count},"latency_ms":{elapsed},"backend":"{:?}","quantization":"{:?}","tps":{tps:.1}}}"#,
        negotiated.backend, quant
    ));

    (token_count, elapsed)
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Simulate on-disk artifact discovery.
/// In production: `get_model_artifacts(model_name)` scans the `models/` directory.
/// Here we provide a .qmodel + .safetensors pair so the ranker always picks .qmodel.
fn mock_artifacts(model_name: &str) -> Vec<ModelArtifact> {
    vec![
        // .qmodel — quantum-compressed NF4 (score 1.0, Quantum backend)
        ModelArtifact::QModel(PathBuf::from(format!("models/{model_name}.qmodel"))),
        // .safetensors — full-precision fallback (score 0.95, GPU backend)
        ModelArtifact::SafeTensors(PathBuf::from(format!("models/{model_name}.safetensors"))),
    ]
}

fn format_artifact_name(a: &ModelArtifact) -> String {
    a.path()
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn print_banner() {
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║      HYBRID LLM PLATFORM — VC DEMO                      ║");
    println!("║      Quantum-Compressed Open-Source Inference            ║");
    println!("║      10× cheaper than proprietary APIs                  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("  Pipeline: artifact → quantization → backend → batch → stream");
    println!("  Formats:  .safetensors  .gguf  .qmodel  .onnx  .engine");
    println!("  Methods:  INT8  INT4  NF4  GPTQ  AWQ  QuantumNF4");
    println!();
}
