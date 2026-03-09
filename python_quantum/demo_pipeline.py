#!/usr/bin/env python3
"""
Hybrid LLM Platform — Python Quantum Pipeline Demo
===================================================

Demonstrates the full Python-side quantum inference pipeline:

  1. Tokenize a prompt
  2. Compute real transformer embeddings (HuggingFace)
  3. Quantum QAOA token reordering using those embeddings
  4. Full QAOA optimisation pass (if Qiskit is available)
  5. Timing and throughput metrics

Run with:
    cd python_quantum
    python demo_pipeline.py

Requirements:
    pip install -r requirements.txt   (torch, transformers, numpy)
    pip install qiskit                (optional — for full QAOA pass)
"""

import sys
import time
import numpy as np

# ── Demo prompts ──────────────────────────────────────────────────────────────

DEMO_PROMPTS = [
    "quantum computing accelerates transformer inference at scale",
    "QuantumNF4 achieves eight times compression with near FP16 perplexity",
    "the revenue flywheel compounds lock in through cheaper inference",
]

DEMO_MODELS = ["Llama-3-70B", "Mistral-7B", "Phi-3-Mini"]

# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_tokenize(prompt: str) -> list[str]:
    """Step 1: Simple whitespace tokenisation (mirrors Rust scheduler stub)."""
    return prompt.split()


def step_compute_embeddings(tokens: list[str]) -> np.ndarray:
    """Step 2: Compute real transformer embeddings via HuggingFace.

    Falls back to random embeddings if torch/transformers are unavailable
    so the demo runs even without a GPU.
    """
    try:
        from transformer_embed import get_embeddings
        embeddings = get_embeddings(tokens)
        return np.array(embeddings)
    except Exception as e:
        print(f"    [warn] HuggingFace unavailable ({e}); using random embeddings")
        # Reproducible fallback: deterministic pseudo-embedding from token hashes
        rng = np.random.default_rng(seed=abs(hash(" ".join(tokens))) % (2**32))
        return rng.standard_normal((len(tokens), 768)).astype(np.float32)


def step_quantum_reorder(tokens: list[str], embeddings: np.ndarray) -> list[str]:
    """Step 3: QAOA-inspired token reordering using transformer embeddings.

    Uses the fast numpy implementation from quantum_optimizer.py, which mirrors
    what the Rust core calls via PyO3 during inference.
    """
    from quantum_optimizer import reorder_tokens_qaoa
    return reorder_tokens_qaoa(tokens, embeddings)


def step_full_qaoa(embeddings: np.ndarray) -> list[int]:
    """Step 4: Full QAOA optimisation pass (Qiskit required).

    Returns an optimal token ordering as a list of indices.
    Falls back to the fast numpy sort if Qiskit is not installed.
    """
    from quantum_optimizer import quantum_optimize_tokens
    return quantum_optimize_tokens(embeddings.tolist())


def step_simulate_inference(reordered_tokens: list[str], backend: str) -> list[str]:
    """Step 5: Simulate the scheduler producing tagged output tokens.

    In production this is handled by schedule_inference_stream in Rust.
    """
    tag = {"GPU": "[GPU]", "Quantum": "[Q]", "Hybrid": "[GPU]|[Q]"}.get(backend, "[?]")
    return [f"{tok}{tag}" for tok in reordered_tokens]


# ── Full pipeline run ─────────────────────────────────────────────────────────

def run_pipeline(model_name: str, prompt: str, backend: str = "Hybrid") -> dict:
    """Execute all 5 pipeline stages and return timing + token data."""
    print(f"\n  MODEL   {model_name}")
    print(f"  PROMPT  \"{prompt}\"")
    print(f"  BACKEND {backend}")
    print()

    results = {}
    t_start = time.perf_counter()

    # Stage 1: Tokenize
    t0 = time.perf_counter()
    tokens = step_tokenize(prompt)
    results["tokenize_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  [1/5] Tokenise      {tokens}")

    # Stage 2: Transformer embeddings
    t0 = time.perf_counter()
    embeddings = step_compute_embeddings(tokens)
    results["embed_ms"] = (time.perf_counter() - t0) * 1000
    print(
        f"  [2/5] Embeddings    shape={embeddings.shape}  "
        f"({results['embed_ms']:.0f}ms)"
    )
    # Print a small slice of the first token's embedding to show real values
    print(f"         token[0] emb[:5] = {embeddings[0, :5].tolist()}")

    # Stage 3: Quantum QAOA token reorder
    t0 = time.perf_counter()
    reordered = step_quantum_reorder(tokens, embeddings)
    results["reorder_ms"] = (time.perf_counter() - t0) * 1000
    print(
        f"  [3/5] QAOA reorder  {reordered}  ({results['reorder_ms']:.0f}ms)"
    )
    delta = sum(1 for a, b in zip(tokens, reordered) if a != b)
    print(f"         {delta}/{len(tokens)} tokens moved by quantum optimiser")

    # Stage 4: Full QAOA (optional)
    t0 = time.perf_counter()
    opt_order = step_full_qaoa(embeddings)
    results["qaoa_ms"] = (time.perf_counter() - t0) * 1000
    print(
        f"  [4/5] Full QAOA     optimal_order={opt_order}  ({results['qaoa_ms']:.0f}ms)"
    )

    # Stage 5: Simulated inference stream
    t0 = time.perf_counter()
    output_tokens = step_simulate_inference(reordered, backend)
    results["infer_ms"] = (time.perf_counter() - t0) * 1000
    results["total_ms"] = (time.perf_counter() - t_start) * 1000
    tps = len(output_tokens) / max(results["total_ms"] / 1000, 1e-6)

    print(f"  [5/5] Token stream  {output_tokens}")
    print()
    print(
        f"  ✓ {len(output_tokens)} tokens in {results['total_ms']:.0f}ms  "
        f"({tps:.0f} tok/s)"
    )

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  HYBRID LLM PLATFORM — Python Quantum Pipeline Demo     ║")
    print("║  artifact → quantization → backend → batch → stream     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def main():
    print_banner()

    backends = ["GPU", "Quantum", "Hybrid"]
    all_results = []

    for i, (model, prompt) in enumerate(zip(DEMO_MODELS, DEMO_PROMPTS)):
        print(f"{'━'*60}")
        backend = backends[i % len(backends)]
        result = run_pipeline(model, prompt, backend)
        all_results.append(result)

    print(f"{'━'*60}")
    print("  SUMMARY")
    avg_total = np.mean([r["total_ms"] for r in all_results])
    avg_embed = np.mean([r["embed_ms"] for r in all_results])
    avg_qaoa  = np.mean([r["qaoa_ms"]  for r in all_results])
    print(f"  Avg total latency      : {avg_total:.0f}ms")
    print(f"  Avg embedding compute  : {avg_embed:.0f}ms")
    print(f"  Avg QAOA reorder time  : {avg_qaoa:.0f}ms")
    print(f"  QAOA overhead fraction : {avg_qaoa/avg_total*100:.1f}%")
    print()
    print("  Connect the Rust server to see live metrics:")
    print("    cargo run --release")
    print("    ws://localhost:8081/metrics")
    print(f"{'━'*60}")


if __name__ == "__main__":
    main()
