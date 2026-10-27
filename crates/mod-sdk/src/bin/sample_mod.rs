//! Reference mod binary used exclusively by integration and conformance tests.
//! This binary exists only for the mods tests and is never shipped.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mod_sdk::testing::SampleMod;
use mod_sdk::{Mod, ModContext, ModManifest, ProcessMod, ProcessModSpec};
use reimagine_core::diagnostic::Diagnostic;
use reimagine_core::event::{Event, EventBus};
use reimagine_core::task::CancelToken;
use serde_json::Value;

struct ExitOnInvokeMod(SampleMod);

#[async_trait]
impl Mod for ExitOnInvokeMod {
    fn manifest(&self) -> &ModManifest {
        self.0.manifest()
    }

    async fn start(&self, ctx: ModContext) -> Result<(), Diagnostic> {
        self.0.start(ctx).await
    }

    async fn invoke(
        &self,
        _operation: &str,
        _payload: Value,
        _cancel: CancelToken,
    ) -> Result<Value, Diagnostic> {
        eprintln!("sample-mod: exiting on invoke");
        std::process::exit(3);
    }

    async fn stop(&self) -> Result<(), Diagnostic> {
        self.0.stop().await
    }
}

/// Answers the handshake, then hangs forever in `start`.
/// Optionally writes a marker when `start` begins, so a test can observe the hang.
struct HangStartMod {
    inner: SampleMod,
    marker: Option<std::path::PathBuf>,
}

impl HangStartMod {
    fn new(marker: Option<std::path::PathBuf>) -> Self {
        Self {
            inner: SampleMod::new(),
            marker,
        }
    }
}

#[async_trait]
impl Mod for HangStartMod {
    fn manifest(&self) -> &ModManifest {
        self.inner.manifest()
    }

    async fn start(&self, _ctx: ModContext) -> Result<(), Diagnostic> {
        if let Some(marker) = &self.marker {
            let _ = std::fs::write(marker, "started");
        }
        std::future::pending().await
    }

    async fn invoke(
        &self,
        operation: &str,
        payload: Value,
        cancel: CancelToken,
    ) -> Result<Value, Diagnostic> {
        self.inner.invoke(operation, payload, cancel).await
    }

    async fn stop(&self) -> Result<(), Diagnostic> {
        self.inner.stop().await
    }
}

/// Panics inside `invoke`; the SDK must turn that into a diagnostic.
struct PanicOnInvokeMod(SampleMod);

#[async_trait]
impl Mod for PanicOnInvokeMod {
    fn manifest(&self) -> &ModManifest {
        self.0.manifest()
    }

    async fn start(&self, ctx: ModContext) -> Result<(), Diagnostic> {
        self.0.start(ctx).await
    }

    async fn invoke(
        &self,
        _operation: &str,
        _payload: Value,
        _cancel: CancelToken,
    ) -> Result<Value, Diagnostic> {
        panic!("sample-mod: panicking on invoke")
    }

    async fn stop(&self) -> Result<(), Diagnostic> {
        self.0.stop().await
    }
}

fn write_pid(path: &str) -> std::io::Result<()> {
    std::fs::write(path, std::process::id().to_string())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str());

    match mode {
        None => {
            mod_sdk::sdk::serve(SampleMod::new()).await?;
        }
        Some("failing") => {
            mod_sdk::sdk::serve(SampleMod::failing_start()).await?;
        }
        Some("exit-on-invoke") => {
            mod_sdk::sdk::serve(ExitOnInvokeMod(SampleMod::new())).await?;
        }
        Some("hang-start") => {
            mod_sdk::sdk::serve(HangStartMod::new(None)).await?;
        }
        Some("panic-on-invoke") => {
            mod_sdk::sdk::serve(PanicOnInvokeMod(SampleMod::new())).await?;
        }
        Some("wrong-protocol") => {
            use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
            let stdin = tokio::io::stdin();
            let mut reader = tokio::io::BufReader::new(stdin).lines();
            if let Ok(Some(line)) = reader.next_line().await {
                let parsed: Value = serde_json::from_str(&line)?;
                let req_id = parsed.get("id").and_then(|v| v.as_i64()).unwrap_or(1);
                let manifest = SampleMod::new().manifest().clone();
                let reply = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "9",
                        "manifest": manifest,
                    }
                });
                let mut stdout = tokio::io::stdout();
                stdout.write_all(reply.to_string().as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
            while let Ok(Some(_)) = reader.next_line().await {}
        }
        Some("pid-file") => {
            let path = args.get(2).expect("pid file path required");
            write_pid(path)?;
            mod_sdk::sdk::serve(SampleMod::new()).await?;
        }
        Some("pid-file-failing") => {
            let path = args.get(2).expect("pid file path required");
            write_pid(path)?;
            mod_sdk::sdk::serve(SampleMod::failing_start()).await?;
        }
        Some("pid-file-hang-start") => {
            let path = args.get(2).expect("pid file path required");
            write_pid(path)?;
            mod_sdk::sdk::serve(HangStartMod::new(None)).await?;
        }
        Some("pid-file-hang-start-marker") => {
            let path = args.get(2).expect("pid file path required");
            let marker = args.get(3).expect("start marker path required");
            write_pid(path)?;
            mod_sdk::sdk::serve(HangStartMod::new(Some(marker.into()))).await?;
        }
        Some("host") => {
            let pid_file_path = args.get(2).expect("pid file path required");
            let exe = std::env::current_exe()?;
            let spec = ProcessModSpec {
                program: exe,
                args: vec!["pid-file".into(), pid_file_path.into()],
                startup_timeout: Duration::from_secs(5),
                shutdown_timeout: Duration::from_secs(3),
            };
            let _mod = ProcessMod::spawn(spec)
                .await
                .map_err(|e| e.message().to_owned())?;
            use std::io::Write;
            println!("ready");
            std::io::stdout().flush()?;
            tokio::time::sleep(Duration::from_secs(3600)).await;
        }
        Some("host-waiting") => {
            let pid_file_path = args.get(2).expect("pid file path required");
            let exe = std::env::current_exe()?;
            let spec = ProcessModSpec {
                program: exe,
                args: vec!["pid-file".into(), pid_file_path.into()],
                startup_timeout: Duration::from_secs(5),
                shutdown_timeout: Duration::from_secs(3),
            };
            let module = Arc::new(
                ProcessMod::spawn(spec)
                    .await
                    .map_err(|e| e.message().to_owned())?,
            );
            let bus = EventBus::new();
            let mut events = bus.subscribe();
            let ctx = ModContext::new(Arc::new(module.manifest().clone()), bus);
            module
                .start(ctx)
                .await
                .map_err(|e| e.message().to_owned())?;
            let invoking = Arc::clone(&module);
            tokio::spawn(async move {
                let _ = invoking
                    .invoke("sample.wait", serde_json::json!({}), CancelToken::new())
                    .await;
            });
            loop {
                match tokio::time::timeout(Duration::from_secs(10), events.recv()).await {
                    Ok(Ok(Event::Mod { kind, .. })) if kind == "sample.progress.started" => break,
                    Ok(Ok(_)) => {}
                    Ok(Err(_)) => {
                        return Err("the sample mod's event bus closed".to_owned().into());
                    }
                    Err(_) => {
                        return Err("the sample mod did not start its waiting call"
                            .to_owned()
                            .into());
                    }
                }
            }
            use std::io::Write;
            println!("ready");
            std::io::stdout().flush()?;
            tokio::time::sleep(Duration::from_secs(3600)).await;
        }
        Some("host-starting") => {
            let pid_file_path = args.get(2).expect("pid file path required");
            let marker_path = args.get(3).expect("start marker path required");
            let exe = std::env::current_exe()?;
            let spec = ProcessModSpec {
                program: exe,
                args: vec![
                    "pid-file-hang-start-marker".into(),
                    pid_file_path.as_str().into(),
                    marker_path.as_str().into(),
                ],
                startup_timeout: Duration::from_secs(3600),
                shutdown_timeout: Duration::from_secs(3),
            };
            let module = Arc::new(
                ProcessMod::spawn(spec)
                    .await
                    .map_err(|e| e.message().to_owned())?,
            );
            let bus = EventBus::new();
            let ctx = ModContext::new(Arc::new(module.manifest().clone()), bus);
            let starting = Arc::clone(&module);
            tokio::spawn(async move {
                let _ = starting.start(ctx).await;
            });
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !std::path::Path::new(marker_path.as_str()).exists() {
                if std::time::Instant::now() > deadline {
                    return Err("the sample mod did not enter start".to_owned().into());
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            use std::io::Write;
            println!("ready");
            std::io::stdout().flush()?;
            tokio::time::sleep(Duration::from_secs(3600)).await;
        }
        Some(other) => {
            eprintln!("unknown mode `{other}`");
            std::process::exit(1);
        }
    }

    Ok(())
}
