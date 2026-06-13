//! `GET /health` route — lightweight liveness check that reports the
//! workspace scope.

use axum::Json;
use axum::extract::State;

use crate::dto::HealthResponse;
use crate::error::AxumHostResult;
use crate::state::AxumHostState;

pub async fn get(State(state): State<AxumHostState>) -> AxumHostResult<Json<HealthResponse>> {
    Ok(Json(HealthResponse::ok(
        state.workspace().workspace_scope().as_str(),
    )))
}
