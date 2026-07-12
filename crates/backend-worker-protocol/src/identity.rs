use serde::{Deserialize, Serialize};

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }
    };
}

string_id!(BackendInstanceId);
string_id!(WorkerInstallationId);
string_id!(WorkerIncarnationId);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerIdentity {
    pub backend_instance_id: BackendInstanceId,
    pub installation_id: WorkerInstallationId,
    pub incarnation_id: WorkerIncarnationId,
    pub worker_version: String,
    pub backend_kind: String,
    pub target: String,
    pub manifest_digest: String,
}
