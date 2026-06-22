use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use reimagine_core::model::RunId;
use reimagine_inference::{
    Backend, BackendInstance, BackendInstanceObservation, BackendInstanceRuntimeHooks,
    BackendInstanceSnapshot, BackendRunLifecycle, BackendRunLifecycleReport,
    BackendRunLifecycleRequest, CompositeBackendInstanceRuntimeHooks, InferenceError,
};

#[derive(Debug)]
struct RecordingHooks {
    backend_instance: BackendInstance,
    backend: Backend,
    begin_count: AtomicUsize,
    cleanup_count: AtomicUsize,
}

impl RecordingHooks {
    fn new(instance: &str) -> Self {
        Self {
            backend_instance: BackendInstance::new(instance),
            backend: Backend::new("fake"),
            begin_count: AtomicUsize::new(0),
            cleanup_count: AtomicUsize::new(0),
        }
    }

    fn begin_count(&self) -> usize {
        self.begin_count.load(Ordering::SeqCst)
    }

    fn cleanup_count(&self) -> usize {
        self.cleanup_count.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl BackendRunLifecycle for RecordingHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
    }

    async fn begin_run(
        &self,
        _request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError> {
        self.begin_count.fetch_add(1, Ordering::SeqCst);
        Ok(BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics: Vec::new(),
        })
    }

    async fn cleanup_run(
        &self,
        _request: BackendRunLifecycleRequest,
    ) -> Result<BackendRunLifecycleReport, InferenceError> {
        self.cleanup_count.fetch_add(1, Ordering::SeqCst);
        Ok(BackendRunLifecycleReport {
            backend_instance: self.backend_instance.clone(),
            diagnostics: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl BackendInstanceObservation for RecordingHooks {
    fn backend_instance(&self) -> &BackendInstance {
        &self.backend_instance
    }

    async fn snapshot(&self) -> BackendInstanceSnapshot {
        let mut observations = BTreeMap::new();
        observations.insert("label".to_owned(), self.backend_instance.to_string());
        BackendInstanceSnapshot {
            backend_instance: self.backend_instance.clone(),
            backend: self.backend.clone(),
            plugin: None,
            extension: None,
            device: None,
            observations,
            diagnostics: Vec::new(),
        }
    }
}

#[tokio::test]
async fn single_hook_snapshots_uses_trait_default() {
    let hook = Arc::new(RecordingHooks::new("fake:single"));
    let snapshots = BackendInstanceObservation::snapshots(hook.as_ref()).await;

    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].backend_instance.to_string(), "fake:single");
}

#[tokio::test]
async fn composite_backend_instance_runtime_hooks_broadcast_lifecycle() {
    let first = Arc::new(RecordingHooks::new("fake:first"));
    let second = Arc::new(RecordingHooks::new("fake:second"));
    let composite = CompositeBackendInstanceRuntimeHooks::new(vec![
        first.clone() as Arc<dyn BackendInstanceRuntimeHooks>,
        second.clone() as Arc<dyn BackendInstanceRuntimeHooks>,
    ]);

    let request = BackendRunLifecycleRequest {
        run_id: RunId::new("run-composite"),
    };

    let begin_report = composite.begin_run(request.clone()).await.unwrap();
    let cleanup_report = composite.cleanup_run(request).await.unwrap();

    assert_eq!(first.begin_count(), 1);
    assert_eq!(second.begin_count(), 1);
    assert_eq!(first.cleanup_count(), 1);
    assert_eq!(second.cleanup_count(), 1);
    assert_eq!(begin_report.diagnostics.len(), 0);
    assert_eq!(cleanup_report.diagnostics.len(), 0);
}

#[tokio::test]
async fn composite_backend_instance_runtime_hooks_collects_snapshots_per_instance() {
    let composite = CompositeBackendInstanceRuntimeHooks::new(vec![
        Arc::new(RecordingHooks::new("fake:first")) as Arc<dyn BackendInstanceRuntimeHooks>,
        Arc::new(RecordingHooks::new("fake:second")) as Arc<dyn BackendInstanceRuntimeHooks>,
    ]);

    let snapshots = composite.snapshots().await;
    let instances: Vec<String> = snapshots
        .iter()
        .map(|snapshot| snapshot.backend_instance.to_string())
        .collect();

    assert_eq!(instances, vec!["fake:first", "fake:second"]);
    assert_eq!(
        snapshots[0].observations.get("label"),
        Some(&"fake:first".to_owned())
    );
}

#[tokio::test]
async fn composite_snapshots_are_recursive_for_nested_composites() {
    let inner = CompositeBackendInstanceRuntimeHooks::new(vec![
        Arc::new(RecordingHooks::new("fake:a")) as Arc<dyn BackendInstanceRuntimeHooks>,
        Arc::new(RecordingHooks::new("fake:b")) as Arc<dyn BackendInstanceRuntimeHooks>,
    ]);
    let outer = CompositeBackendInstanceRuntimeHooks::new(vec![
        Arc::new(inner) as Arc<dyn BackendInstanceRuntimeHooks>
    ]);

    let snapshots = outer.snapshots().await;
    let instances: Vec<String> = snapshots
        .iter()
        .map(|snapshot| snapshot.backend_instance.to_string())
        .collect();

    assert_eq!(instances, vec!["fake:a", "fake:b"]);
}

#[tokio::test]
async fn empty_composite_returns_no_snapshots() {
    let composite = CompositeBackendInstanceRuntimeHooks::new(vec![]);
    let snapshots = composite.snapshots().await;
    assert!(snapshots.is_empty());
}
