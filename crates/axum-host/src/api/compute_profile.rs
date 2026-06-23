//! `GET /compute-profile` route — workspace capability discovery
//! surface.
//!
//! Returns the app-host DTO projection of the workspace's most
//! recent compute profile (see
//! [`WorkspaceHost::compute_profile_dto`](reimagine_app_host::WorkspaceHost::compute_profile_dto)).
//! The wire shape never carries backend-native device handles,
//! tensors, or loaded model structs.

use axum::Json;
use axum::extract::State;

use crate::dto::ComputeProfileDto;
use crate::error::AxumHostResult;
use crate::state::AxumHostState;

pub async fn get(State(state): State<AxumHostState>) -> AxumHostResult<Json<ComputeProfileDto>> {
    Ok(Json(state.workspace().compute_profile_dto()))
}
