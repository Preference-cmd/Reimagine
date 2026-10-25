//! mDNS registration and discovery tests.
//!
//! `registered_worker_is_discovered` exercises the real mDNS path
//! (register, browse, resolve) and only runs where the host OS permits
//! multicast; on hosts that reject multicast sends it skips with an
//! explicit note instead of failing (registration itself is still
//! covered everywhere by `register_then_unregister_roundtrip`).

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};

use reimagine_backend_worker_protocol::discovery::{
    DiscoveredWorker, MdnsWorkerDiscovery, MdnsWorkerRegister,
};

/// `true` when a same-host multicast round-trip works.
///
/// mDNS rides on link-local multicast; a host that rejects multicast
/// sends (OS privacy restriction, no multicast route) cannot run the
/// end-to-end discovery test. The probe mirrors what the discovery
/// daemons do: join a group, send to it, and expect the packet back via
/// multicast loopback. A missing loopback delivery is treated as
/// unavailable (conservative: the e2e test would time out anyway).
fn multicast_available() -> bool {
    use std::net::UdpSocket;

    let group_ip = Ipv4Addr::new(224, 0, 0, 9);
    let receiver = match UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))) {
        Ok(socket) => socket,
        Err(_) => return false,
    };
    if receiver
        .join_multicast_v4(&group_ip, &Ipv4Addr::UNSPECIFIED)
        .is_err()
    {
        return false;
    }
    let sender = match UdpSocket::bind(SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0))) {
        Ok(socket) => socket,
        Err(_) => return false,
    };
    let _ = sender.set_multicast_loop_v4(true);
    let _ = sender.set_multicast_ttl_v4(4);
    let port = receiver.local_addr().unwrap().port();
    if sender
        .send_to(b"reimagine-mdns-probe", SocketAddr::from((group_ip, port)))
        .is_err()
    {
        return false;
    }
    let mut buffer = [0_u8; 64];
    receiver
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    matches!(
        receiver.recv_from(&mut buffer),
        Ok((n, _)) if &buffer[..n] == b"reimagine-mdns-probe"
    )
}

/// Register a worker and wait for the discovery browser to resolve it.
///
/// mDNS propagation is asynchronous and interface-dependent, so the
/// lookup polls well past the discovery daemon's initial settling
/// window before giving up.
#[tokio::test]
async fn registered_worker_is_discovered() {
    if !multicast_available() {
        eprintln!(
            "skipping registered_worker_is_discovered: the host OS rejects multicast sends, \
             so same-host mDNS discovery cannot run here (covered on multicast-capable hosts)"
        );
        return;
    }

    let id = format!("reimagine-mdns-test-{}", std::process::id());
    let addr: SocketAddr = "127.0.0.1:59100".parse().unwrap();
    let mut properties = HashMap::new();
    properties.insert("endpoint".to_owned(), format!("https://{addr}"));
    properties.insert("backend".to_owned(), "burn".to_owned());
    properties.insert("devices".to_owned(), "cuda:0".to_owned());
    properties.insert("capabilities".to_owned(), "echo".to_owned());
    properties.insert("fingerprint".to_owned(), "aa".repeat(32));

    let _registration =
        MdnsWorkerRegister::register(&id, addr, properties).expect("mdns registration");

    let discovery = MdnsWorkerDiscovery::start().expect("mdns discovery");
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut found: Option<DiscoveredWorker> = None;
    while Instant::now() < deadline {
        found = discovery
            .discovered()
            .into_iter()
            .find(|worker| worker.id.starts_with(&id));
        if found.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    let worker = found.expect("registered worker should be discovered within 20s");
    assert_eq!(
        worker.https_endpoint(),
        Some(addr),
        "the advertised https endpoint must round-trip"
    );
    assert_eq!(worker.backend_kind(), "burn");
    assert_eq!(worker.devices(), vec!["cuda:0"]);
    assert_eq!(worker.capabilities(), vec!["echo"]);
    assert_eq!(
        worker.fingerprint(),
        Some("aa".repeat(32).as_str()),
        "the advertised fingerprint must round-trip"
    );

    let _ = discovery.stop();
}

/// Registration and deregistration are daemon commands, not network
/// I/O: they must succeed even where multicast is blocked.
#[test]
fn register_then_unregister_roundtrip() {
    let id = format!("reimagine-mdns-reg-{}", std::process::id());
    let addr: SocketAddr = "127.0.0.1:59101".parse().unwrap();
    let properties = HashMap::from([("endpoint".to_owned(), format!("https://{addr}"))]);

    let registration = MdnsWorkerRegister::register(&id, addr, properties)
        .expect("mdns registration must succeed");
    registration.unregister().expect("mdns unregister");
}

#[test]
fn endpoint_txt_must_be_https() {
    let worker = DiscoveredWorker {
        id: "w._reimagine-worker._tcp.local.".to_owned(),
        addr: "192.168.1.10:9100".parse().unwrap(),
        properties: HashMap::from([("endpoint".to_owned(), "quic://192.168.1.10:9100".to_owned())]),
    };
    assert_eq!(worker.https_endpoint(), None);
}
