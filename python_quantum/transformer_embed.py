import numpy as np

MODEL_CACHE = {}

def _deterministic_fallback(tokens: list, dim: int = 384):
    seed = abs(hash(" ".join(tokens))) % (2**32)
    rng = np.random.default_rng(seed=seed)
    return rng.standard_normal((len(tokens), dim)).astype(np.float32)

def get_embeddings(tokens: list, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except Exception:
        return _deterministic_fallback(tokens).tolist()

    default_model = "sentence-transformers/all-MiniLM-L6-v2"
    chosen = model_name or default_model

    if chosen not in MODEL_CACHE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(chosen)
            model = AutoModel.from_pretrained(chosen)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(default_model)
                model = AutoModel.from_pretrained(default_model)
            except Exception:
                return _deterministic_fallback(tokens).tolist()
        model.eval()
        MODEL_CACHE[chosen] = (tokenizer, model)

    tokenizer, model = MODEL_CACHE[chosen]

    # Compute per-token embeddings by running the encoder on each token.
    # This is slower than a single pass, but keeps a 1:1 mapping between tokens
    # and embedding rows (required for token reordering in Rust).
    vecs = []
    for tok in tokens:
        try:
            inputs = tokenizer(tok, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)  # [1, hidden]
            vecs.append(emb.squeeze(0).cpu().numpy())
        except Exception:
            return _deterministic_fallback(tokens).tolist()

    arr = np.stack(vecs, axis=0).astype(np.float32)
    # Return a nested list so Rust/PyO3 can extract Vec<Vec<f32>> without pyo3-numpy.
    return arr.tolist()