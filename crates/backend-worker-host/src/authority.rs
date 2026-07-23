use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use reimagine_backend_worker_protocol::WorkerIncarnationId;
use reimagine_inference::BackendPayloadKey;

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum WorkerAuthorityError {
    StaleIncarnation {
        expected: WorkerIncarnationId,
        actual: WorkerIncarnationId,
    },
    UnknownPayloadKey(BackendPayloadKey),
}

impl std::fmt::Display for WorkerAuthorityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleIncarnation { expected, actual } => write!(
                formatter,
                "worker payload belongs to incarnation `{}` rather than `{}`",
                expected.0, actual.0
            ),
            Self::UnknownPayloadKey(key) => {
                write!(formatter, "unknown worker payload key `{}`", key.as_str())
            }
        }
    }
}

pub(crate) struct WorkerAuthorityTable {
    incarnation_id: WorkerIncarnationId,
    next_key: AtomicU64,
    worker_tokens: Mutex<HashMap<BackendPayloadKey, String>>,
}

impl WorkerAuthorityTable {
    pub(crate) fn new(incarnation_id: WorkerIncarnationId) -> Self {
        Self {
            incarnation_id,
            next_key: AtomicU64::new(1),
            worker_tokens: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn register(&self, worker_token: String) -> BackendPayloadKey {
        let sequence = self.next_key.fetch_add(1, Ordering::Relaxed);
        let host_key = BackendPayloadKey::new(format!(
            "worker:{}:handle:{sequence}",
            self.incarnation_id.0
        ));
        self.worker_tokens
            .lock()
            .expect("worker authority table poisoned")
            .insert(host_key.clone(), worker_token);
        host_key
    }

    pub(crate) fn resolve(
        &self,
        incarnation_id: &WorkerIncarnationId,
        host_key: &BackendPayloadKey,
    ) -> Result<String, WorkerAuthorityError> {
        if incarnation_id != &self.incarnation_id {
            return Err(WorkerAuthorityError::StaleIncarnation {
                expected: self.incarnation_id.clone(),
                actual: incarnation_id.clone(),
            });
        }
        self.worker_tokens
            .lock()
            .expect("worker authority table poisoned")
            .get(host_key)
            .cloned()
            .ok_or_else(|| WorkerAuthorityError::UnknownPayloadKey(host_key.clone()))
    }

    pub(crate) fn incarnation_id(&self) -> &WorkerIncarnationId {
        &self.incarnation_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_resolution_is_bound_to_worker_incarnation() {
        let incarnation = WorkerIncarnationId::from("inc-1");
        let authority = WorkerAuthorityTable::new(incarnation.clone());
        let key = authority.register("worker-token-1".to_owned());

        assert_eq!(
            authority.resolve(&incarnation, &key).unwrap(),
            "worker-token-1"
        );
        assert!(matches!(
            authority.resolve(&WorkerIncarnationId::from("inc-2"), &key),
            Err(WorkerAuthorityError::StaleIncarnation { .. })
        ));
    }

    #[test]
    fn payload_key_from_replaced_incarnation_never_resolves_in_new_worker() {
        let old = WorkerAuthorityTable::new(WorkerIncarnationId::from("inc-old"));
        let old_key = old.register("old-token".to_owned());
        let replacement = WorkerAuthorityTable::new(WorkerIncarnationId::from("inc-new"));
        let replacement_key = replacement.register("new-token".to_owned());

        assert_ne!(old_key, replacement_key);
        assert!(matches!(
            replacement.resolve(replacement.incarnation_id(), &old_key),
            Err(WorkerAuthorityError::UnknownPayloadKey(_))
        ));
    }
}
