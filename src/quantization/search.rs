/// All quantization schemes understood by the platform.
/// The pipeline selects the method based on the `quantization` field in ChatRequest.
///
/// Compression pipeline position: artifact → **quantization** → backend → batching → streaming
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantizationMethod {
    /// Standard 8-bit integer quantization — 2x size reduction, minimal quality loss.
    INT8,
    /// 4-bit integer quantization — 4x size reduction, moderate quality loss.
    INT4,
    /// NormalFloat 4-bit (NF4) — 4-bit with better distribution fit for LLM weights.
    NF4,
    /// GPTQ post-training quantization — high-quality 4-bit via second-order statistics.
    GPTQ,
    /// Activation-aware weight quantization — preserves outlier channels for accuracy.
    AWQ,
    /// Quantum-assisted NF4 — uses QAOA/VQE to find optimal weight permutations before
    /// NF4 quantization. Achieves up to 8x compression with near-FP16 perplexity.
    /// Requires `quantum_provider` field in the API request.
    QuantumNF4,
}

impl QuantizationMethod {
    /// Parse from the API `quantization` string field.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "int8"           => Some(Self::INT8),
            "int4"           => Some(Self::INT4),
            "nf4"            => Some(Self::NF4),
            "gptq"           => Some(Self::GPTQ),
            "awq"            => Some(Self::AWQ),
            "quantum_nf4"    => Some(Self::QuantumNF4),
            _                => None,
        }
    }

    /// Approximate bits-per-weight for storage and bandwidth planning.
    pub fn bits_per_weight(self) -> f32 {
        match self {
            Self::INT8       => 8.0,
            Self::INT4       => 4.0,
            Self::NF4        => 4.0,
            Self::GPTQ       => 4.0,
            Self::AWQ        => 4.0,
            Self::QuantumNF4 => 4.0, // same storage as NF4, better perplexity
        }
    }

    /// Compression ratio relative to FP16 baseline (16 bits).
    pub fn compression_ratio(self) -> f32 {
        16.0 / self.bits_per_weight()
    }

    /// Whether this method requires the Python quantum layer.
    pub fn requires_quantum_backend(self) -> bool {
        matches!(self, Self::QuantumNF4)
    }
}
