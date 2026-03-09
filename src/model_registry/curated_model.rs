use serde::Serialize;

/// A curated open-source LLM available for compression and inference.
/// All models are served 10x cheaper than proprietary APIs via quantum compression.
#[derive(Clone, Debug, Serialize)]
pub struct CuratedModel {
    /// Short display name (e.g. "Llama-3-70B")
    pub name: &'static str,
    /// Hugging Face repository identifier
    pub hf_repo: &'static str,
    /// Total parameter count (used for shard planning and pricing)
    pub parameter_count: u64,
}

/// Registry of curated open-source models supported by the compression cluster.
/// Extend this list to add new models without changing any other code.
pub static CURATED_MODELS: &[CuratedModel] = &[
    CuratedModel {
        name: "Llama-3-8B",
        hf_repo: "meta-llama/Meta-Llama-3-8B",
        parameter_count: 8_000_000_000,
    },
    CuratedModel {
        name: "Llama-3-70B",
        hf_repo: "meta-llama/Meta-Llama-3-70B",
        parameter_count: 70_000_000_000,
    },
    CuratedModel {
        name: "Mistral-7B",
        hf_repo: "mistralai/Mistral-7B-v0.1",
        parameter_count: 7_000_000_000,
    },
    CuratedModel {
        name: "Mixtral-8x7B",
        hf_repo: "mistralai/Mixtral-8x7B-Instruct-v0.1",
        parameter_count: 46_000_000_000,
    },
    CuratedModel {
        name: "Phi-3-Mini",
        hf_repo: "microsoft/Phi-3-mini-4k-instruct",
        parameter_count: 3_800_000_000,
    },
];
