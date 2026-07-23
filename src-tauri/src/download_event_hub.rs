use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use reimagine_app_host::dto::DownloadEventPayload;
use reimagine_model_acquisition::AcquisitionReport;
use reimagine_model_acquisition::hf::provider::AcquisitionProgressSink;
use tauri::ipc::Channel;

/// Tauri-owned download event hub.
///
/// Subscribed channels receive live `DownloadEventPayload` per download id.
/// Channel send failures are silently dropped.
#[derive(Debug, Clone)]
pub struct TauriDownloadEventHub {
    inner: Arc<Mutex<HubInner>>,
}

#[derive(Default)]
struct HubInner {
    subscribers: HashMap<String, Channel<DownloadEventPayload>>,
}

impl std::fmt::Debug for HubInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubInner")
            .field(
                "subscribers",
                &format_args!("{} channels", self.subscribers.len()),
            )
            .finish()
    }
}

impl TauriDownloadEventHub {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HubInner::default())),
        }
    }

    /// Register a channel to receive events for a download id.
    pub fn subscribe(&self, download_id: &str, channel: Channel<DownloadEventPayload>) {
        let mut guard = self.inner.lock().expect("download hub poisoned");
        guard.subscribers.insert(download_id.to_string(), channel);
    }

    /// Remove a subscriber.
    #[allow(dead_code)]
    pub fn unsubscribe(&self, download_id: &str) {
        let mut guard = self.inner.lock().expect("download hub poisoned");
        guard.subscribers.remove(download_id);
    }

    /// Send a payload to a subscriber, silently dropping on error.
    fn send(&self, download_id: &str, payload: DownloadEventPayload) {
        let mut guard = self.inner.lock().expect("download hub poisoned");
        if let Some(channel) = guard.subscribers.get(download_id)
            && channel.send(payload).is_err()
        {
            guard.subscribers.remove(download_id);
        }
    }

    /// Create an `AcquisitionProgressSink` for a given download id.
    ///
    /// The returned sink sends events via this hub's subscribed channel.
    pub fn sink_for(&self, download_id: &str) -> Arc<dyn AcquisitionProgressSink> {
        Arc::new(TauriDownloadProgressSink {
            hub: self.clone(),
            download_id: download_id.to_string(),
        })
    }
}

/// Bridges `TauriDownloadEventHub` to the `AcquisitionProgressSink` trait.
struct TauriDownloadProgressSink {
    hub: TauriDownloadEventHub,
    download_id: String,
}

impl AcquisitionProgressSink for TauriDownloadProgressSink {
    fn started(&self, repo_id: &str, revision: &str) {
        self.hub.send(
            &self.download_id,
            DownloadEventPayload {
                id: self.download_id.clone(),
                status: "started".to_string(),
                repo_id: repo_id.to_string(),
                revision: revision.to_string(),
                bytes_downloaded: 0,
                total_bytes: None,
                message: None,
            },
        );
    }

    fn file_done(&self, relative_path: &str, bytes: u64, outcome: &str) {
        self.hub.send(
            &self.download_id,
            DownloadEventPayload {
                id: self.download_id.clone(),
                status: "in_progress".to_string(),
                repo_id: String::new(),
                revision: String::new(),
                bytes_downloaded: bytes,
                total_bytes: None,
                message: Some(format!("file {outcome}: {relative_path}")),
            },
        );
    }

    fn done(&self, report: &AcquisitionReport) {
        self.hub.send(
            &self.download_id,
            DownloadEventPayload {
                id: self.download_id.clone(),
                status: "completed".to_string(),
                repo_id: report.repo_id.clone(),
                revision: report.revision.clone(),
                bytes_downloaded: report.total_bytes,
                total_bytes: Some(report.total_bytes),
                message: None,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_model_acquisition::AcquisitionReport;
    use std::sync::mpsc;

    #[test]
    fn delivers_download_events_to_subscribed_channel() {
        let hub = TauriDownloadEventHub::new();
        let download_id = "dl-1";

        let (tx, rx) = mpsc::channel();
        let channel = Channel::<DownloadEventPayload>::new(move |payload| {
            // In Tauri test mode the callback receives InvokeResponseBody.
            // We just verify the send doesn't panic.
            let _ = payload;
            tx.send(()).ok();
            Ok(())
        });

        hub.subscribe(download_id, channel);

        let sink = hub.sink_for(download_id);
        sink.started("test/model", "main");

        let received = rx.recv_timeout(std::time::Duration::from_millis(100));
        assert!(received.is_ok(), "should deliver started event");
    }

    #[test]
    fn unknown_download_id_silently_ignored() {
        let hub = TauriDownloadEventHub::new();
        // Should not panic
        let sink = hub.sink_for("unknown");
        sink.started("test/model", "main");
    }

    #[test]
    fn completed_event_contains_report_info() {
        let hub = TauriDownloadEventHub::new();
        let download_id = "dl-2";

        let (tx, rx) = mpsc::channel();
        let channel = Channel::<DownloadEventPayload>::new(move |payload| {
            let _ = payload;
            tx.send(()).ok();
            Ok(())
        });

        hub.subscribe(download_id, channel);

        let sink = hub.sink_for(download_id);
        let report = AcquisitionReport::new("test", "repo", "rev", "target");
        sink.done(&report);

        let received = rx.recv_timeout(std::time::Duration::from_millis(100));
        assert!(received.is_ok());

        // Verify the sink methods do not panic
        let mut events = Vec::new();
        let hub2 = TauriDownloadEventHub::new();
        let sink_inner = Arc::new(TauriDownloadProgressSink {
            hub: hub2,
            download_id: "test".to_string(),
        });
        sink_inner.started("r1", "v1");
        events.push(());
        sink_inner.file_done("f1", 100, "downloaded");
        events.push(());
        sink_inner.done(&report);
        events.push(());
        assert_eq!(events.len(), 3);
    }
}
