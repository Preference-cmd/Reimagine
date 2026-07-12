use reimagine_backend_worker_protocol::{
    HandshakeError, ProtocolRange, ProtocolVersion, negotiate_protocol,
};

#[test]
fn negotiation_selects_highest_mutually_supported_version() {
    assert_eq!(
        negotiate_protocol(ProtocolRange::new(1, 4), ProtocolRange::new(2, 3)).unwrap(),
        ProtocolVersion(3)
    );
}

#[test]
fn negotiation_reports_both_incompatible_ranges() {
    let host = ProtocolRange::new(1, 2);
    let worker = ProtocolRange::new(3, 4);
    assert_eq!(
        negotiate_protocol(host, worker),
        Err(HandshakeError::NoCompatibleVersion { host, worker })
    );
}

#[test]
fn negotiation_rejects_an_inverted_range() {
    let invalid = ProtocolRange::new(3, 2);
    assert_eq!(
        negotiate_protocol(invalid, ProtocolRange::new(1, 3)),
        Err(HandshakeError::InvalidRange(invalid))
    );
}
