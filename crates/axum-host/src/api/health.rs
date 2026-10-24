//! `GET /health` route — lightweight liveness check that reports
//! workspace scope, kernel health, and AIGC health.

use axum::Json;
use axum::extract::State;

use crate::dto::HealthResponse;
use crate::error::AxumHostResult;
use crate::state::AxumHostState;

pub async fn get(State(state): State<AxumHostState>) -> AxumHostResult<Json<HealthResponse>> {
    Ok(Json(HealthResponse::from_workspace(state.workspace())))
}
