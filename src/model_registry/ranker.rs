use super::{artifact::ModelArtifact, ranked_artifact::{RankedArtifact, BackendPreference}};

pub fn rank_artifacts(artifacts: Vec<ModelArtifact>) -> Vec<RankedArtifact> {
    let mut ranked: Vec<RankedArtifact> = artifacts
        .into_iter()
        .map(|a| {
            let (score, backend) = match &a {
                // .qmodel — quantum-compressed NF4; top rank, routes to Quantum backend
                ModelArtifact::QModel(_)           => (1.00, BackendPreference::Quantum),
                // TensorRT engine — highest GPU throughput when hardware is available
                ModelArtifact::TensorRT(_)         => (0.98, BackendPreference::GPU),
                // .npy — intermediate quantum optimisation artifact
                ModelArtifact::QuantumOptimized(_) => (0.97, BackendPreference::Quantum),
                // SafeTensors — HF native, full precision GPU
                ModelArtifact::SafeTensors(_)      => (0.95, BackendPreference::GPU),
                // ONNX — portable cross-runtime
                ModelArtifact::ONNX(_)             => (0.92, BackendPreference::Hybrid),
                // GGUF — CPU-optimised llama.cpp format
                ModelArtifact::GGUF(_)             => (0.90, BackendPreference::CPU),
                // PyTorch checkpoint — GPU fallback
                ModelArtifact::PytorchBin(_)       => (0.80, BackendPreference::GPU),
            };

            RankedArtifact {
                artifact: a,
                score,
                backend,
            }
        })
        .collect();

    ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    ranked
}