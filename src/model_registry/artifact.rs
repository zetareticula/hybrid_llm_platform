use std::path::PathBuf;

/// Typed model artifact discriminated by on-disk format.
///
/// Compression pipeline position: **artifact** → quantization → backend → batching → streaming
///
/// Supported formats:
///   .safetensors — HuggingFace native, full precision, GPU-preferred
///   .gguf        — llama.cpp format, CPU/Hybrid inference
///   .bin / .pt   — PyTorch checkpoint, GPU fallback
///   .onnx        — cross-runtime, Hybrid inference
///   .engine      — TensorRT compiled, GPU-optimised highest throughput
///   .qmodel      — Quantum-compressed NF4 weight file produced by the compression cluster
///   .npy         — NumPy weight array used during quantum optimisation passes
#[derive(Clone, Debug)]
pub enum ModelArtifact {
    /// HuggingFace safetensors — preferred for GPU inference
    SafeTensors(PathBuf),
    /// llama.cpp GGUF — preferred for CPU / edge inference
    GGUF(PathBuf),
    /// TensorRT compiled engine — highest GPU throughput, hardware-locked
    TensorRT(PathBuf),
    /// ONNX cross-runtime — Hybrid CPU+GPU, portable
    ONNX(PathBuf),
    /// Quantum-compressed NF4 model (.qmodel) — output of the compression cluster.
    /// Achieves up to 8x compression with near-FP16 perplexity via QAOA/VQE weight
    /// permutation before NF4 quantisation. Requires the Quantum backend.
    QModel(PathBuf),
    /// NumPy weight array produced during block-wise quantum optimisation passes
    QuantumOptimized(PathBuf),
    /// PyTorch binary checkpoint (.bin / .pt)
    PytorchBin(PathBuf),
}

impl ModelArtifact {
    /// Return the filesystem path for this artifact.
    pub fn path(&self) -> &PathBuf {
        match self {
            ModelArtifact::SafeTensors(p)      => p,
            ModelArtifact::GGUF(p)             => p,
            ModelArtifact::TensorRT(p)         => p,
            ModelArtifact::ONNX(p)             => p,
            ModelArtifact::QModel(p)           => p,
            ModelArtifact::QuantumOptimized(p) => p,
            ModelArtifact::PytorchBin(p)       => p,
        }
    }

    /// Construct the correct variant from a file extension.
    pub fn from_path(path: PathBuf) -> Option<Self> {
        match path.extension()?.to_str()? {
            "safetensors" => Some(ModelArtifact::SafeTensors(path)),
            "gguf"        => Some(ModelArtifact::GGUF(path)),
            "bin" | "pt"  => Some(ModelArtifact::PytorchBin(path)),
            "onnx"        => Some(ModelArtifact::ONNX(path)),
            "engine"      => Some(ModelArtifact::TensorRT(path)),
            // .qmodel — quantum-compressed NF4 output from the compression cluster
            "qmodel"      => Some(ModelArtifact::QModel(path)),
            "npy"         => Some(ModelArtifact::QuantumOptimized(path)),
            _             => None,
        }
    }
}
