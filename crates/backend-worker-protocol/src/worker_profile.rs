use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::BackendInstanceId;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerProfile {
    pub instances: Vec<WorkerInstanceProfile>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerInstanceProfile {
    pub backend_instance_id: BackendInstanceId,
    pub device_label: String,
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub operation_options: Value,
}
