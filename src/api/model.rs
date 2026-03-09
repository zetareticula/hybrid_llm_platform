use axum::Json;
use crate::model_registry::curated_model::{CURATED_MODELS, CuratedModel};

/// GET /v1/models — returns the curated list of supported open-source LLMs.
pub async fn list_models() -> Json<Vec<&'static CuratedModel>> {
    Json(CURATED_MODELS.iter().collect())
}
