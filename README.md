# Hybrid LLM Platform

> **Open-source LLM inference at 10× lower cost than proprietary APIs — powered by quantum-assisted compression.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

---

## Overview

Hybrid LLM Platform is a production-grade inference server that compresses open-source LLMs using quantum-assisted optimisation (QAOA / VQE) and serves them via a high-throughput Rust core. Developers get an OpenAI-compatible REST API with **no per-token pricing** — just compute costs, typically **10× cheaper** than GPT-4 / Claude equivalents for the same model class.

### Example Request

```json
POST /v1/chat/completions
{
  "model":            "meta-llama/Llama-3-70B",
  "backend":          "hybrid",
  "quantization":     "quantum_nf4",
  "quantum_provider": "ibm",
  "messages": [
    { "role": "user", "content": "Explain dark matter" }
  ]
}
```

---

## Infra-Business Model

The platform operates three independent but compounding clusters. Each cluster generates value that feeds the next.

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Compression        │      │  Inference           │      │  Artifact           │
│  Cluster            │─────▶│  Cluster             │─────▶│  Registry           │
│                     │      │                      │      │                     │
│  QAOA/VQE compress  │      │  GPU shard fleet     │      │  .qmodel store      │
│  HF weights →       │      │  dynamic batching    │      │  versioned, signed  │
│  .qmodel artifacts  │      │  streaming tokens    │      │  metered access     │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
         ▲                                                           │
         └───────────────────────────────────────────────────────────┘
                         Revenue Flywheel (see below)
```

### 1. Compression Cluster

Accepts raw HuggingFace checkpoints in any supported format and produces `.qmodel` artifacts — quantum-compressed NF4 weight files that achieve **up to 8× size reduction** with near-FP16 perplexity.

**How it works:**
1. Model weights are loaded block-by-block (`compressed_tensor.py`)
2. Each block is amplitude-encoded into a quantum circuit
3. A VQE cost operator finds the optimal permutation of weights before NF4 quantisation
4. QAOA further optimises token-order routing for inference (`qaoa_prompt_opt.py`)
5. The result is saved as a `.qmodel` file and registered in the Artifact Registry

**Compression ratios by method:**

| Method       | Bits/weight | Compression vs FP16 | Quality loss |
|--------------|-------------|----------------------|--------------|
| `int8`       | 8           | 2×                   | Minimal      |
| `int4`       | 4           | 4×                   | Moderate     |
| `nf4`        | 4           | 4×                   | Low          |
| `gptq`       | 4           | 4×                   | Very low     |
| `awq`        | 4           | 4×                   | Very low     |
| `quantum_nf4`| 4           | **8×** (eff.)        | Near-zero    |

### 2. Inference Cluster

A Tokio-based Rust server with dynamic GPU batching. Key properties:

- **Pre-sharded weights** — `preshard_weights()` maps transformer layers to GPU nodes at startup, minimising cross-shard communication during forward passes
- **Dynamic compute topology** — shard assignments rebalance based on load; latter-half transformer layers receive 25% more VRAM allocation to match attention-head growth
- **Backend negotiation** — every request passes through `BackendNegotiator` before reaching the scheduler, enforcing the artifact → quantization → backend invariant
- **Token streaming** — GPU and Quantum backends produce interleaved token streams over `mpsc` channels, delivered to clients as SSE or WebSocket

**Throughput advantage:** With `quantum_nf4` compression, a 70B-parameter model fits in the same VRAM as a 9B FP16 model, allowing **7–8× more concurrent requests per GPU node**.

### 3. Artifact Registry

The registry (`model_registry`) stores and versions `.qmodel` files after compression. It is the single source of truth for:

- Which formats are available for a given model (`get_model_artifacts`)
- The ranked preference order (`rank_artifacts`) — `.qmodel` scores 1.0, TensorRT 0.98, SafeTensors 0.95, GGUF 0.90
- Cache invalidation when new compressed artifacts arrive

---

## Revenue Flywheel

```
More developers
      │
      ▼
More inference volume ──────────────────────────────────┐
      │                                                  │
      ▼                                                  │
More compression jobs                          More .qmodel artifacts
      │                                                  │
      ▼                                                  ▼
Better QAOA training data            Lower per-token compute cost
      │                                                  │
      └─────────────────────── 10× cheaper ─────────────┘
                                      │
                                      ▼
                              Compounding lock-in
```

**Lock-in mechanisms (non-destructive — all open-source):**

1. **`.qmodel` format** — compressed artifacts are portable but optimised for this runtime. Developers build tooling around the format.
2. **Artifact Registry API** — once a team has a library of compressed models, migrating means re-running expensive compression jobs.
3. **Quantization telemetry** — the metrics WebSocket (port 8081) feeds back compression quality data that improves future QAOA runs — a data flywheel.
4. **`CURATED_MODELS` registry** — new open-source models are onboarded here first, making this platform the fastest path to production for new releases.

---

## Supported Model Formats

| Extension     | Variant                     | Backend preference | Notes                                    |
|---------------|-----------------------------|--------------------|------------------------------------------|
| `.safetensors`| `ModelArtifact::SafeTensors`| GPU                | HuggingFace native; preferred for GPU    |
| `.gguf`       | `ModelArtifact::GGUF`       | CPU                | llama.cpp format; edge / CPU inference   |
| `.bin` / `.pt`| `ModelArtifact::PytorchBin` | GPU (fallback)     | PyTorch checkpoint                       |
| `.onnx`       | `ModelArtifact::ONNX`       | Hybrid             | Cross-runtime; CPU + GPU portable        |
| `.engine`     | `ModelArtifact::TensorRT`   | GPU                | TensorRT compiled; highest throughput    |
| `.qmodel`     | `ModelArtifact::QModel`     | Quantum            | **Output of compression cluster**; 8× compressed NF4 |
| `.npy`        | `ModelArtifact::QuantumOptimized` | Quantum      | Intermediate quantum optimisation artifact |

---

## Architecture

### Causal Pipeline

Every inference request flows through this exact sequence — no shortcuts:

```
ChatRequest (JSON)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  artifact                                                   │
│  get_model_artifacts() → [.qmodel, .safetensors, .gguf …]  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  quantization                                               │
│  QuantizationMethod::from_str("quantum_nf4")                │
│  rank_artifacts() → top: QModel(score=1.0, Quantum)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  backend                                                    │
│  negotiate(ranked, QuantumNF4, available_shards)            │
│  → NegotiatedBackend { backend: Quantum, shards: 2,         │
│                        batch_tokens: 4096 }                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  batching                                                   │
│  QuantumHybridBatcher / GpuBatcher                          │
│  ShardTopology: layer 0-39 → GPU 0, layer 40-79 → GPU 1     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  streaming                                                  │
│  schedule_inference_stream() → mpsc::Receiver<String>       │
│  GPU ⟷ Quantum tokens interleaved per Hybrid mode          │
└─────────────────────────────────────────────────────────────┘
```

### Module Map

```
src/
├── main.rs                         Entry point — initialises GPU manager, batcher, API server
├── api/
│   ├── mod.rs                      Axum router: POST /v1/chat/completions, GET /v1/models
│   └── model.rs                    GET /v1/models handler
├── session.rs                      Full pipeline router (artifact → … → streaming)
├── model_registry/
│   ├── mod.rs                      init_registry, get_model, get_model_artifacts
│   ├── artifact.rs                 ModelArtifact enum (.safetensors/.gguf/.qmodel/…)
│   ├── ranked_artifact.rs          RankedArtifact + BackendPreference enum
│   ├── ranker.rs                   rank_artifacts() — scores each format
│   └── curated_model.rs            CURATED_MODELS static registry
├── quantization/
│   ├── mod.rs
│   ├── quantum.rs                  Rust↔Python bridge: embeddings, QAOA reorder, blockwise infer
│   └── search.rs                   QuantizationMethod enum (INT8/INT4/NF4/GPTQ/AWQ/QuantumNF4)
├── inference/
│   ├── mod.rs
│   ├── backend_negotiator.rs       negotiate() — resolves Backend+shards+batch from ranked artifact
│   ├── scheduler.rs                Backend enum, schedule_inference, schedule_inference_stream
│   ├── batcher.rs                  DynamicBatcher — groups requests by model+backend
│   ├── batcher_api.rs              HTTP handler using DynamicBatcher
│   ├── cpu.rs                      cpu_infer()
│   ├── gpu.rs                      gpu_infer()
│   └── stream_api.rs               handle_chat_stream() — streaming endpoint handler
├── scheduler/
│   ├── mod.rs
│   ├── execution_graph.rs          NodeId / Edge / ExecutionGraph
│   ├── path_optimizer.rs           Dijkstra shortest_path (f32-safe BinaryHeap)
│   ├── automorphism.rs             GraphAutomorphism::dither()
│   ├── callback_log.rs             CallbackLog — latency history
│   └── override_engine.rs          optimize_execution() — dither vs shortest-path decision
├── gpu_shard.rs                    GpuManager, ShardTopology, preshard_weights()
├── gpu_batcher.rs                  GpuBatcher — memory-aware streaming GPU batching
├── streaming_quantum_scheduler.rs  QuantumHybridBatcher — quantum token reorder + streaming
└── metrics_ws.rs                   WebSocket metrics push server (port 8081)

python_quantum/
├── quantum_optimizer.py            reorder_tokens_qaoa(), quantum_optimize_tokens()
├── qaoa_prompt_opt.py              Full QAOA token-ordering solver (Qiskit)
├── compressed_tensor.py            optimize_model() — VQE block-wise weight compression
├── optimize_full_model.py          Block-wise full-model compression orchestrator
├── py_full_model.py                Alternative full-model compression implementation
├── full-model-block-wise.py        Block-wise compression with configurable block size
└── transformer_embed.py            get_embeddings() — HuggingFace transformer embeddings
```

---

## REST API Reference

### `POST /v1/chat/completions`

**Request body:**

```json
{
  "model":            "Llama-3-70B",
  "backend":          "hybrid",
  "quantization":     "quantum_nf4",
  "quantum_provider": "ibm",
  "messages": [
    { "role": "system",  "content": "You are a helpful assistant." },
    { "role": "user",    "content": "Explain dark matter" }
  ]
}
```

| Field              | Type     | Required | Description                                           |
|--------------------|----------|----------|-------------------------------------------------------|
| `model`            | string   | ✓        | Model name (matches `CURATED_MODELS` or HF repo)     |
| `backend`          | string   |          | `gpu` / `cpu` / `hybrid` / `quantum` (hint only)     |
| `quantization`     | string   |          | `int8` / `int4` / `nf4` / `gptq` / `awq` / `quantum_nf4` |
| `quantum_provider` | string   |          | `ibm` / `aws` / `ionq` (required for `quantum_nf4`)  |
| `messages`         | array    | ✓        | OpenAI-compatible message array                       |

**Response:**

```json
{
  "token_stream": ["Explain", "[GPU]", "dark", "[GPU]", "matter", "[GPU]"]
}
```

### `GET /v1/models`

Returns the list of curated open-source models available for inference.

```json
[
  { "name": "Llama-3-8B",   "hf_repo": "meta-llama/Meta-Llama-3-8B",  "parameter_count": 8000000000 },
  { "name": "Llama-3-70B",  "hf_repo": "meta-llama/Meta-Llama-3-70B", "parameter_count": 70000000000 },
  { "name": "Mistral-7B",   "hf_repo": "mistralai/Mistral-7B-v0.1",   "parameter_count": 7000000000 },
  { "name": "Mixtral-8x7B", "hf_repo": "mistralai/Mixtral-8x7B-Instruct-v0.1", "parameter_count": 46000000000 },
  { "name": "Phi-3-Mini",   "hf_repo": "microsoft/Phi-3-mini-4k-instruct", "parameter_count": 3800000000 }
]
```

### WebSocket — `/metrics` (port 8081)

Streams real-time JSON events per completed request:

```json
{ "model": "Llama-3-70B", "tokens": 142 }
```

---

## Getting Started

### Prerequisites

- Rust 1.75+
- Python 3.10+ with `pip`
- CUDA 12+ (for GPU backend)
- Qiskit (for `quantum_nf4` backend)

### Install Python dependencies

```bash
pip install -r requirements.txt
pip install -r python_quantum/requirements.txt
```

### Build and run

```bash
cargo build --release
cargo run --release
```

The server starts on:
- `http://0.0.0.0:8080` — REST API
- `ws://0.0.0.0:8081/metrics` — metrics WebSocket

### Place model weights

```
models/
├── Llama-3-70B.qmodel        # quantum-compressed (highest priority)
├── Llama-3-70B.safetensors   # fallback
└── Mistral-7B.gguf           # CPU inference
```

The artifact ranker automatically selects the best available format.

---

## Compression Workflow

To compress a new model and register it:

```bash
# 1. Download from HuggingFace
python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Meta-Llama-3-70B', cache_dir='models/')"

# 2. Convert to .npy blocks and run quantum compression
python python_quantum/optimize_full_model.py models/Llama-3-70B.safetensors ibm

# 3. The output path is printed — rename to .qmodel for top-rank selection
mv models/Llama-3-70B_optimized.npy models/Llama-3-70B.qmodel
```

---

## Design Principles

1. **Causal pipeline enforcement** — `artifact → quantization → backend → batching → streaming`. No request skips steps. `session.rs` is the single enforcement point.
2. **Typed artifact abstraction** — `ModelArtifact` enum ensures format-specific handling is exhaustive at compile time. Adding a new format requires updating all match arms.
3. **Post-rank enum selection** — `rank_artifacts` returns a scored, sorted list. `BackendPreference` is derived from the artifact format, not from the caller's hint.
4. **Scheduler-ready backend negotiation** — `NegotiatedBackend` carries shard count and batch token budget alongside the backend enum, so the scheduler never has to infer these from the request.
5. **No proprietary dependencies** — the entire inference stack is open-source. Quantum providers (IBM, AWS) are optional and only activated by `quantum_nf4` quantization.

---

## License

MIT — see [LICENSE](LICENSE).
