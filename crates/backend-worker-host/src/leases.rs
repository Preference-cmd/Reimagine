use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Duration;

use reimagine_core::model::RunId;
use tokio::sync::Notify;
use tokio::time::{Instant, timeout_at};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerAdmissionState {
    Ready,
    Draining,
}

#[derive(Debug)]
pub struct WorkerRunLeaseError {
    run_id: RunId,
}

impl std::fmt::Display for WorkerRunLeaseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "worker is draining and rejects the first lease for run `{}`",
            self.run_id
        )
    }
}

impl std::error::Error for WorkerRunLeaseError {}

#[derive(Debug)]
struct RunLeaseState {
    admission: WorkerAdmissionState,
    owned_runs: HashSet<RunId>,
}

#[derive(Debug)]
pub struct WorkerRunLeases {
    state: Mutex<RunLeaseState>,
    changed: Notify,
}

impl Default for WorkerRunLeases {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerRunLeases {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(RunLeaseState {
                admission: WorkerAdmissionState::Ready,
                owned_runs: HashSet::new(),
            }),
            changed: Notify::new(),
        }
    }

    pub fn acquire(&self, run_id: &RunId) -> Result<bool, WorkerRunLeaseError> {
        let mut state = self.state.lock().expect("worker run leases poisoned");
        if state.owned_runs.contains(run_id) {
            return Ok(false);
        }
        if state.admission == WorkerAdmissionState::Draining {
            return Err(WorkerRunLeaseError {
                run_id: run_id.clone(),
            });
        }
        state.owned_runs.insert(run_id.clone());
        Ok(true)
    }

    pub fn release(&self, run_id: &RunId) -> bool {
        let released = self
            .state
            .lock()
            .expect("worker run leases poisoned")
            .owned_runs
            .remove(run_id);
        if released {
            self.changed.notify_waiters();
        }
        released
    }

    pub fn begin_draining(&self) {
        self.state
            .lock()
            .expect("worker run leases poisoned")
            .admission = WorkerAdmissionState::Draining;
    }

    pub fn restore_ready(&self) {
        self.state
            .lock()
            .expect("worker run leases poisoned")
            .admission = WorkerAdmissionState::Ready;
    }

    pub fn admission(&self) -> WorkerAdmissionState {
        self.state
            .lock()
            .expect("worker run leases poisoned")
            .admission
    }

    pub fn owned_run_count(&self) -> usize {
        self.state
            .lock()
            .expect("worker run leases poisoned")
            .owned_runs
            .len()
    }

    pub fn owns(&self, run_id: &RunId) -> bool {
        self.state
            .lock()
            .expect("worker run leases poisoned")
            .owned_runs
            .contains(run_id)
    }

    pub fn owned_run_ids(&self) -> Vec<RunId> {
        self.state
            .lock()
            .expect("worker run leases poisoned")
            .owned_runs
            .iter()
            .cloned()
            .collect()
    }

    pub async fn wait_until_empty(&self, deadline: Duration) -> bool {
        let deadline = Instant::now() + deadline;
        loop {
            let changed = self.changed.notified();
            if self.owned_run_count() == 0 {
                return true;
            }
            if timeout_at(deadline, changed).await.is_err() {
                return self.owned_run_count() == 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup_releases_only_an_owned_run_once() {
        let leases = WorkerRunLeases::new();
        let owned = RunId::new("owned");
        let never_routed = RunId::new("never-routed");

        assert!(leases.acquire(&owned).expect("acquire"));
        assert!(!leases.release(&never_routed));
        assert!(leases.release(&owned));
        assert!(!leases.release(&owned));
    }
}
