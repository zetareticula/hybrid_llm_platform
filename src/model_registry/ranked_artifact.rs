use super::artifact::ModelArtifact;

/// Backend preference resolved by the ranker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendPreference {
    GPU,
    CPU,
    Hybrid,
    Quantum,
}

/// A ranked model artifact with a quality score and backend recommendation.
#[derive(Clone, Debug)]
pub struct RankedArtifact {
    pub artifact: ModelArtifact,
    /// Quality / efficiency score in [0.0, 1.0]; higher is better.
    pub score: f32,
    pub backend: BackendPreference,
}
