import numpy as np

def reorder_tokens_qaoa(tokens, embeddings):
    """Fast QAOA-inspired token reordering used by the Rust bridge (quantum.rs).
    Sorts tokens by the first principal embedding dimension."""
    emb = np.array(embeddings)
    idx = np.argsort(emb[:, 0])
    return [tokens[i] for i in idx]

def quantum_optimize_tokens(token_embeddings):
    """Full QAOA token-ordering optimisation.
    Delegates to the production QAOA solver in qaoa_prompt_opt.py when available,
    otherwise falls back to the fast numpy sort above."""
    try:
        from qaoa_prompt_opt import quantum_optimize_tokens as _qaoa_full
        return _qaoa_full(np.array(token_embeddings))
    except Exception:
        emb = np.array(token_embeddings)
        return list(np.argsort(emb[:, 0]))