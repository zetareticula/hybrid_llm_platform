from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

MODEL_CACHE = {}

def get_embeddings(tokens: list, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if model_name not in MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        MODEL_CACHE[model_name] = (tokenizer, model)
    else:
        tokenizer, model = MODEL_CACHE[model_name]

    text = " ".join(tokens)
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()