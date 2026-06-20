//! `GET /nodes` route — host catalog projection.
//!
//! The route is deliberately thin: node metadata comes from
//! `WorkspaceHost::list_node_defs`, which in turn reads the app-host
//! `NodeCatalogService`. Axum does not maintain a separate node list.

use axum::Json;
use axum::extract::State;

use crate::dto::{NodeCatalogResponse, NodeDefDto};
use crate::error::AxumHostResult;
use crate::state::AxumHostState;

pub async fn list(State(state): State<AxumHostState>) -> AxumHostResult<Json<NodeCatalogResponse>> {
    let nodes = state
        .workspace()
        .list_node_defs()
        .into_iter()
        .map(NodeDefDto::from)
        .collect();
    Ok(Json(NodeCatalogResponse { nodes }))
}
