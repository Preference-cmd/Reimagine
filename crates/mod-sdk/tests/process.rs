//! Process-only behavior tests for ProcessMod.

use std::sync::Arc;
use std::time::Duration;

use mod_sdk::{Mod, ModContext, ModRegistry, ProcessMod, ProcessModSpec};
use reimagine_core::event::EventBus;
use reimagine_core::health::Health;
use reimagine_core::task::CancelToken;
use serde_json::json;

#[tokio::test]
async fn wrong_protocol_returns_protocol_mismatch() {
    let bin = env!("CARGO_BIN_EXE_sample-mod");
    let mut spec = ProcessModSpec::new(bin);
    spec.args.push("wrong-protocol".into());

    let err = ProcessMod::spawn(spec).await.expect_err("spawn must fail");
    assert_eq!(err.code().as_str(), "MOD/INTERNAL_PROTOCOL_MISMATCH");
}

#[tokio::test]
async fn nonexistent_program_returns_diagnostic() {
    let spec = ProcessModSpec::new("nonexistent-mod-binary-12345");
    let err = ProcessMod::spawn(spec).await.expect_err("spawn must fail");
    assert_eq!(err.code().as_str(), "MOD/INTERNAL_SPAWN_FAILED");
}

#[tokio::test]
async fn exit_on_invoke_reports_exited_with_stderr() {
    let bin = env!("CARGO_BIN_EXE_sample-mod");
    let mut spec = ProcessModSpec::new(bin);
    spec.args.push("exit-on-invoke".into());

    let proc = ProcessMod::spawn(spec).await.expect("spawn must succeed");
    let bus = EventBus::new();
    let ctx = ModContext::new(Arc::new(proc.manifest().clone()), bus);
    proc.start(ctx).await.expect("start must succeed");

    let err = proc
        .invoke("sample.echo", json!({ "text": "hi" }), CancelToken::new())
        .await
        .expect_err("invoke must fail on exit");
    assert_eq!(err.code().as_str(), "MOD/INTERNAL_EXITED");
    assert!(
        err.message().contains("sample-mod: exiting on invoke"),
        "error message `{}` missing stderr line",
        err.message()
    );

    let err2 = proc
        .invoke("sample.echo", json!({ "text": "hi" }), CancelToken::new())
        .await
        .expect_err("second invoke must also fail");
    assert_eq!(err2.code().as_str(), "MOD/INTERNAL_EXITED");
    assert!(
        err2.message().contains("sample-mod: exiting on invoke"),
        "error message `{}` missing stderr line",
        err2.message()
    );
}

#[tokio::test]
async fn a_panicking_invoke_returns_a_diagnostic() {
    let mut spec = ProcessModSpec::new(env!("CARGO_BIN_EXE_sample-mod"));
    spec.args.push("panic-on-invoke".into());

    let proc = ProcessMod::spawn(spec).await.expect("spawn must succeed");
    let bus = EventBus::new();
    let ctx = ModContext::new(Arc::new(proc.manifest().clone()), bus);
    proc.start(ctx).await.expect("start must succeed");

    let err = tokio::time::timeout(
        Duration::from_secs(10),
        proc.invoke("sample.echo", json!({ "text": "hi" }), CancelToken::new()),
    )
    .await
    .expect("a panicking invoke must not hang")
    .expect_err("a panic is a failed call");
    assert_eq!(err.code().as_str(), "MOD/INTERNAL_PANIC");
}

#[cfg(unix)]
fn temp_pid_file(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "sample-mod-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after the epoch")
            .as_nanos()
    ))
}

#[cfg(unix)]
fn read_child_pid(path: &std::path::Path) -> i32 {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(text) = std::fs::read_to_string(path)
            && let Ok(pid) = text.trim().parse::<i32>()
        {
            return pid;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no child pid in {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(unix)]
fn process_is_alive(pid: i32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(unix)]
fn wait_for_process_exit(pid: i32, timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if !process_is_alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

#[cfg(unix)]
#[tokio::test]
async fn host_killed_cleans_up_child() {
    let pid_file = temp_pid_file("idle");
    let _ = std::fs::remove_file(&pid_file);

    let mut host_cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_sample-mod"));
    host_cmd
        .args(["host", pid_file.to_str().unwrap()])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut host_child = host_cmd.spawn().expect("spawn host");
    let stdout = host_child.stdout.take().expect("stdout piped");
    let mut reader = tokio::io::BufReader::new(stdout).lines();
    use tokio::io::AsyncBufReadExt;
    let ready_line = tokio::time::timeout(Duration::from_secs(5), reader.next_line())
        .await
        .expect("timeout waiting for ready line")
        .expect("read line error")
        .expect("unexpected EOF before ready");
    assert_eq!(ready_line.trim(), "ready");

    let child_pid = read_child_pid(&pid_file);
    assert!(
        process_is_alive(child_pid),
        "child must be running before host is killed"
    );

    // SIGKILL the host
    host_child.kill().await.expect("kill host");

    assert!(
        wait_for_process_exit(child_pid, Duration::from_secs(5)),
        "child {child_pid} did not exit after host was killed"
    );
    let _ = std::fs::remove_file(&pid_file);
}

#[cfg(unix)]
#[tokio::test]
async fn host_killed_during_an_invoke_cleans_up_child() {
    let pid_file = temp_pid_file("inflight");
    let _ = std::fs::remove_file(&pid_file);

    let mut host_cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_sample-mod"));
    host_cmd
        .args(["host-waiting", pid_file.to_str().unwrap()])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut host_child = host_cmd.spawn().expect("spawn host");
    let stdout = host_child.stdout.take().expect("stdout piped");
    let mut reader = tokio::io::BufReader::new(stdout).lines();
    use tokio::io::AsyncBufReadExt;
    let ready_line = tokio::time::timeout(Duration::from_secs(10), reader.next_line())
        .await
        .expect("timeout waiting for ready line")
        .expect("read line error")
        .expect("unexpected EOF before ready");
    assert_eq!(ready_line.trim(), "ready");

    let child_pid = read_child_pid(&pid_file);
    assert!(
        process_is_alive(child_pid),
        "child must be in flight before the host is killed"
    );

    host_child.kill().await.expect("kill host");

    assert!(
        wait_for_process_exit(child_pid, Duration::from_secs(5)),
        "child {child_pid} survived a host killed during an invoke"
    );
    let _ = std::fs::remove_file(&pid_file);
}

#[cfg(unix)]
#[tokio::test]
async fn host_killed_during_a_hung_start_cleans_up_child() {
    let pid_file = temp_pid_file("hung-start");
    let marker = temp_pid_file("hung-start-marker");
    let _ = std::fs::remove_file(&pid_file);
    let _ = std::fs::remove_file(&marker);

    let mut host_cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_sample-mod"));
    host_cmd
        .args([
            "host-starting",
            pid_file.to_str().unwrap(),
            marker.to_str().unwrap(),
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut host_child = host_cmd.spawn().expect("spawn host");
    let stdout = host_child.stdout.take().expect("stdout piped");
    let mut reader = tokio::io::BufReader::new(stdout).lines();
    use tokio::io::AsyncBufReadExt;
    let ready_line = tokio::time::timeout(Duration::from_secs(10), reader.next_line())
        .await
        .expect("timeout waiting for ready line")
        .expect("read line error")
        .expect("unexpected EOF before ready");
    assert_eq!(ready_line.trim(), "ready");

    let child_pid = read_child_pid(&pid_file);
    assert!(
        process_is_alive(child_pid),
        "child must be hung in start before the host is killed"
    );

    host_child.kill().await.expect("kill host");

    assert!(
        wait_for_process_exit(child_pid, Duration::from_secs(5)),
        "child {child_pid} survived a host killed during a hung start"
    );
    let _ = std::fs::remove_file(&pid_file);
    let _ = std::fs::remove_file(&marker);
}

#[cfg(unix)]
#[tokio::test]
async fn stopping_a_mod_lets_its_process_exit() {
    let pid_file = temp_pid_file("stop");
    let _ = std::fs::remove_file(&pid_file);

    let mut spec = ProcessModSpec::new(env!("CARGO_BIN_EXE_sample-mod"));
    spec.args.push("pid-file".into());
    spec.args.push(pid_file.as_os_str().to_owned());
    let proc = Arc::new(ProcessMod::spawn(spec).await.expect("spawn must succeed"));

    let registry = ModRegistry::start(vec![proc as Arc<dyn Mod>], &EventBus::new()).await;
    assert!(registry.list()[0].health.is_available());
    let child_pid = read_child_pid(&pid_file);

    registry.stop().await;

    assert!(
        wait_for_process_exit(child_pid, Duration::from_secs(5)),
        "child {child_pid} survived a graceful stop"
    );
    let _ = std::fs::remove_file(&pid_file);
}

#[cfg(unix)]
#[tokio::test]
async fn a_failed_start_kills_the_child() {
    let pid_file = temp_pid_file("failed-start");
    let _ = std::fs::remove_file(&pid_file);

    let mut spec = ProcessModSpec::new(env!("CARGO_BIN_EXE_sample-mod"));
    spec.args.push("pid-file-failing".into());
    spec.args.push(pid_file.as_os_str().to_owned());
    let proc = Arc::new(ProcessMod::spawn(spec).await.expect("spawn must succeed"));

    let registry = ModRegistry::start(vec![proc as Arc<dyn Mod>], &EventBus::new()).await;
    match &registry.list()[0].health {
        Health::Unavailable(diagnostic) => {
            assert_eq!(diagnostic.code().as_str(), "MOD/INTERNAL_START_FAILED");
        }
        Health::Available => panic!("a failing mod was listed available"),
    }

    let child_pid = read_child_pid(&pid_file);
    assert!(
        wait_for_process_exit(child_pid, Duration::from_secs(5)),
        "child {child_pid} survived a failed start"
    );
    let _ = std::fs::remove_file(&pid_file);
}

#[cfg(unix)]
#[tokio::test]
async fn a_hung_start_times_out_and_kills_the_child() {
    let pid_file = temp_pid_file("hang-start");
    let _ = std::fs::remove_file(&pid_file);

    let mut spec = ProcessModSpec::new(env!("CARGO_BIN_EXE_sample-mod"));
    spec.args.push("pid-file-hang-start".into());
    spec.args.push(pid_file.as_os_str().to_owned());
    spec.startup_timeout = Duration::from_secs(2);
    let proc = ProcessMod::spawn(spec).await.expect("spawn must succeed");

    let bus = EventBus::new();
    let ctx = ModContext::new(Arc::new(proc.manifest().clone()), bus);
    let err = tokio::time::timeout(Duration::from_secs(10), proc.start(ctx))
        .await
        .expect("start must not hang")
        .expect_err("start must time out");
    assert_eq!(err.code().as_str(), "MOD/INTERNAL_TIMEOUT");

    let child_pid = read_child_pid(&pid_file);
    assert!(
        wait_for_process_exit(child_pid, Duration::from_secs(5)),
        "child {child_pid} survived a start timeout"
    );
    let _ = std::fs::remove_file(&pid_file);
}
