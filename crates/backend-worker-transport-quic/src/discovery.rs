use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

use crate::Error;

/// The mDNS service type for Reimagine GPU workers.
const SERVICE_TYPE: &str = "_reimagine-worker._tcp.local.";

/// TXT records published/consumed by the discovery protocol:
/// `endpoint=quic://<ip>:<port>`, `backend=<kind>`,
/// `devices=<comma-separated>`, `capabilities=<comma-separated>`,
/// `fingerprint=<sha256-of-cert-der, hex>` (T19 trust model).

/// A discovered worker on the LAN.
#[derive(Debug, Clone)]
pub struct DiscoveredWorker {
    pub id: String,
    pub addr: SocketAddr,
    pub properties: HashMap<String, String>,
}

impl DiscoveredWorker {
    /// Parse the QUIC endpoint from the properties.
    pub fn quic_endpoint(&self) -> Option<SocketAddr> {
        self.properties
            .get("endpoint")
            .and_then(|ep| ep.strip_prefix("quic://"))
            .and_then(|addr| addr.parse().ok())
    }

    /// Get the backend kind.
    pub fn backend_kind(&self) -> &str {
        self.properties
            .get("backend")
            .map(|s| s.as_str())
            .unwrap_or("unknown")
    }

    /// Get the device labels.
    pub fn devices(&self) -> Vec<&str> {
        self.properties
            .get("devices")
            .map(|d| d.split(',').collect())
            .unwrap_or_default()
    }

    /// Get the supported capabilities.
    pub fn capabilities(&self) -> Vec<&str> {
        self.properties
            .get("capabilities")
            .map(|c| c.split(',').collect())
            .unwrap_or_default()
    }

    /// The certificate fingerprint advertised by the worker, if any.
    ///
    /// Published as the `fingerprint` TXT record (SHA-256 over the
    /// worker's self-signed certificate DER, hex-encoded). A host can
    /// pre-validate this against its pinned keys before connecting.
    #[must_use]
    pub fn fingerprint(&self) -> Option<&str> {
        self.properties.get("fingerprint").map(String::as_str)
    }
}

/// Registers a worker service via mDNS.
pub struct MdnsWorkerRegister {
    daemon: ServiceDaemon,
    service_name: String,
}

impl MdnsWorkerRegister {
    /// Register a new worker service.
    pub fn register(
        id: &str,
        addr: SocketAddr,
        properties: HashMap<String, String>,
    ) -> Result<Self, Error> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| Error::ConnectionFailed(format!("mdns daemon: {e}")))?;

        let service_info = ServiceInfo::new(
            SERVICE_TYPE,
            &format!("{id}.local"),
            &format!("{id}.local"),
            addr.ip().to_string(),
            addr.port(),
            properties,
        )
        .map_err(|e| Error::ConnectionFailed(format!("mdns service info: {e}")))?;

        daemon
            .register(service_info)
            .map_err(|e| Error::ConnectionFailed(format!("mdns register: {e}")))?;

        let service_name = format!("{id}.{}", SERVICE_TYPE);

        Ok(Self {
            daemon,
            service_name,
        })
    }

    /// Unregister the service.
    pub fn unregister(&self) -> Result<(), Error> {
        self.daemon
            .unregister(&self.service_name)
            .map_err(|e| Error::ConnectionFailed(format!("mdns unregister: {e}")))?;
        Ok(())
    }
}

impl Drop for MdnsWorkerRegister {
    fn drop(&mut self) {
        let _ = self.unregister();
    }
}

/// Discovers workers via mDNS browsing.
pub struct MdnsWorkerDiscovery {
    daemon: ServiceDaemon,
    workers: Arc<Mutex<HashMap<String, DiscoveredWorker>>>,
    _handle: std::thread::JoinHandle<()>,
}

impl MdnsWorkerDiscovery {
    /// Start browsing for workers.
    pub fn start() -> Result<Self, Error> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| Error::ConnectionFailed(format!("mdns daemon: {e}")))?;

        let rx = daemon
            .browse(SERVICE_TYPE)
            .map_err(|e| Error::ConnectionFailed(format!("mdns browse: {e}")))?;

        let workers = Arc::new(Mutex::new(HashMap::new()));
        let workers_clone = workers.clone();

        let handle = std::thread::spawn(move || {
            while let Ok(event) = rx.recv() {
                match event {
                    ServiceEvent::ServiceResolved(info) => {
                        let addr: Option<SocketAddr> =
                            info.get_addresses().iter().next().and_then(|addr| {
                                format!("{}:{}", addr, info.get_port()).parse().ok()
                            });
                        if let Some(addr) = addr {
                            let properties: HashMap<String, String> = info
                                .get_properties()
                                .iter()
                                .filter_map(|prop| {
                                    let val = prop.val()?;
                                    let key = prop.key().to_string();
                                    let val_str = String::from_utf8(val.to_vec()).ok()?;
                                    Some((key, val_str))
                                })
                                .collect();
                            let worker = DiscoveredWorker {
                                id: info.get_fullname().to_string(),
                                addr,
                                properties,
                            };
                            workers_clone
                                .lock()
                                .unwrap()
                                .insert(worker.id.clone(), worker);
                        }
                    }
                    ServiceEvent::ServiceRemoved { .. } => {
                        // Handle removal if needed
                    }
                    _ => {}
                }
            }
        });

        // Wait a bit for initial discoveries
        std::thread::sleep(Duration::from_millis(500));

        Ok(Self {
            daemon,
            workers,
            _handle: handle,
        })
    }

    /// Get currently discovered workers.
    pub fn discovered(&self) -> Vec<DiscoveredWorker> {
        self.workers.lock().unwrap().values().cloned().collect()
    }

    /// Stop browsing.
    pub fn stop(&self) -> Result<(), Error> {
        self.daemon
            .stop_browse(SERVICE_TYPE)
            .map_err(|e| Error::ConnectionFailed(format!("mdns stop browse: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovered_worker_parsing() {
        let mut props = HashMap::new();
        props.insert(
            "endpoint".to_string(),
            "quic://192.168.1.100:9100".to_string(),
        );
        props.insert("backend".to_string(), "burn".to_string());
        props.insert("devices".to_string(), "cuda:0,cuda:1".to_string());
        props.insert(
            "capabilities".to_string(),
            "load_bundle,text_encode".to_string(),
        );

        let worker = DiscoveredWorker {
            id: "test-worker._reimagine-worker._tcp.local.".to_string(),
            addr: "192.168.1.100:9100".parse().unwrap(),
            properties: props,
        };

        assert_eq!(
            worker.quic_endpoint(),
            Some("192.168.1.100:9100".parse().unwrap())
        );
        assert_eq!(worker.backend_kind(), "burn");
        assert_eq!(worker.devices(), vec!["cuda:0", "cuda:1"]);
        assert_eq!(worker.capabilities(), vec!["load_bundle", "text_encode"]);
    }
}
