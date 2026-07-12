use std::collections::{HashMap, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{CorrelationId, ProgressFrame, RequestId, TerminalFrame, WorkerIncarnationId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelDisposition {
    Requested,
    AlreadyRequested,
    AlreadyTerminal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LifecycleError {
    DuplicateRequest(RequestId),
    UnknownRequest(RequestId),
    CorrelationMismatch {
        request_id: RequestId,
        expected: CorrelationId,
        actual: CorrelationId,
    },
    DuplicateTerminal(RequestId),
    ProgressAfterTerminal(RequestId),
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "worker request lifecycle violation: {self:?}")
    }
}

impl std::error::Error for LifecycleError {}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TransportLost {
    pub incarnation_id: WorkerIncarnationId,
    pub pending_request_ids: Vec<RequestId>,
}

#[derive(Debug)]
struct RequestState {
    correlation_id: CorrelationId,
    cancel_requested: bool,
}

const DEFAULT_TERMINAL_RETENTION: usize = 1024;

#[derive(Debug)]
pub struct RequestTracker {
    active_requests: HashMap<RequestId, RequestState>,
    terminal_tombstones: HashMap<RequestId, CorrelationId>,
    terminal_order: VecDeque<RequestId>,
    terminal_retention: usize,
}

impl Default for RequestTracker {
    fn default() -> Self {
        Self {
            active_requests: HashMap::new(),
            terminal_tombstones: HashMap::new(),
            terminal_order: VecDeque::new(),
            terminal_retention: DEFAULT_TERMINAL_RETENTION,
        }
    }
}

impl RequestTracker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        request_id: RequestId,
        correlation_id: CorrelationId,
    ) -> Result<(), LifecycleError> {
        if self.active_requests.contains_key(&request_id)
            || self.terminal_tombstones.contains_key(&request_id)
        {
            return Err(LifecycleError::DuplicateRequest(request_id));
        }
        self.active_requests.insert(
            request_id,
            RequestState {
                correlation_id,
                cancel_requested: false,
            },
        );
        Ok(())
    }

    pub fn record_progress(&self, progress: &ProgressFrame) -> Result<(), LifecycleError> {
        if let Some(expected) = self.terminal_tombstones.get(&progress.request_id) {
            Self::verify_correlation_value(
                &progress.request_id,
                &progress.correlation_id,
                expected,
            )?;
            return Err(LifecycleError::ProgressAfterTerminal(
                progress.request_id.clone(),
            ));
        }
        self.state_for(&progress.request_id, &progress.correlation_id)?;
        Ok(())
    }

    pub fn record_terminal(&mut self, terminal: &TerminalFrame) -> Result<(), LifecycleError> {
        if let Some(expected) = self.terminal_tombstones.get(&terminal.request_id) {
            Self::verify_correlation_value(
                &terminal.request_id,
                &terminal.correlation_id,
                expected,
            )?;
            return Err(LifecycleError::DuplicateTerminal(
                terminal.request_id.clone(),
            ));
        }
        self.state_for(&terminal.request_id, &terminal.correlation_id)?;
        self.active_requests
            .remove(&terminal.request_id)
            .expect("active request was verified immediately before removal");
        self.terminal_tombstones
            .insert(terminal.request_id.clone(), terminal.correlation_id.clone());
        self.terminal_order.push_back(terminal.request_id.clone());
        self.evict_terminal_tombstones();
        Ok(())
    }

    pub fn request_cancel(
        &mut self,
        request_id: &RequestId,
        correlation_id: &CorrelationId,
    ) -> Result<CancelDisposition, LifecycleError> {
        if let Some(expected) = self.terminal_tombstones.get(request_id) {
            Self::verify_correlation_value(request_id, correlation_id, expected)?;
            return Ok(CancelDisposition::AlreadyTerminal);
        }
        let state = self.state_for_mut(request_id, correlation_id)?;
        if state.cancel_requested {
            return Ok(CancelDisposition::AlreadyRequested);
        }
        state.cancel_requested = true;
        Ok(CancelDisposition::Requested)
    }

    #[must_use]
    pub fn transport_lost(&self, incarnation_id: WorkerIncarnationId) -> TransportLost {
        let mut pending_request_ids = self.active_requests.keys().cloned().collect::<Vec<_>>();
        pending_request_ids.sort();
        TransportLost {
            incarnation_id,
            pending_request_ids,
        }
    }

    fn state_for(
        &self,
        request_id: &RequestId,
        correlation_id: &CorrelationId,
    ) -> Result<&RequestState, LifecycleError> {
        let state = self
            .active_requests
            .get(request_id)
            .ok_or_else(|| LifecycleError::UnknownRequest(request_id.clone()))?;
        Self::verify_correlation(request_id, correlation_id, state)?;
        Ok(state)
    }

    fn state_for_mut(
        &mut self,
        request_id: &RequestId,
        correlation_id: &CorrelationId,
    ) -> Result<&mut RequestState, LifecycleError> {
        let state = self
            .active_requests
            .get_mut(request_id)
            .ok_or_else(|| LifecycleError::UnknownRequest(request_id.clone()))?;
        Self::verify_correlation(request_id, correlation_id, state)?;
        Ok(state)
    }

    fn verify_correlation(
        request_id: &RequestId,
        correlation_id: &CorrelationId,
        state: &RequestState,
    ) -> Result<(), LifecycleError> {
        Self::verify_correlation_value(request_id, correlation_id, &state.correlation_id)
    }

    fn verify_correlation_value(
        request_id: &RequestId,
        correlation_id: &CorrelationId,
        expected: &CorrelationId,
    ) -> Result<(), LifecycleError> {
        if *expected == *correlation_id {
            return Ok(());
        }
        Err(LifecycleError::CorrelationMismatch {
            request_id: request_id.clone(),
            expected: expected.clone(),
            actual: correlation_id.clone(),
        })
    }

    fn evict_terminal_tombstones(&mut self) {
        while self.terminal_order.len() > self.terminal_retention {
            if let Some(request_id) = self.terminal_order.pop_front() {
                self.terminal_tombstones.remove(&request_id);
            }
        }
    }
}
