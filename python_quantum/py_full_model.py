import numpy as np
from pathlib import Path
from compressed_tensor import optimize_model  # single-block quantum VQE optimizer

def optimize_full_model(model_path: str, provider: str = "") -> str:
    """
    Block-wise full-model quantum optimization.
    """
    weights = np.load(model_path)
    block_size = 8  # feasible for small quantum circuits
    n_weights = weights.size

    optimized_weights = np.zeros_like(weights)

    for i in range(0, n_weights, block_size):
        block = weights[i:i+block_size]
        temp_block_path = f"{model_path}.block_{i}.npy"
        np.save(temp_block_path, block)
        optimized_block_path = optimize_model(temp_block_path, provider, block_size)
        optimized_block = np.load(optimized_block_path)
        optimized_weights[i:i+block_size] = optimized_block
        Path(temp_block_path).unlink()
        Path(optimized_block_path).unlink()

    optimized_model_path = f"{model_path}.quantum_optimized_full.npy"
    np.save(optimized_model_path, optimized_weights)
    return optimized_model_path