//! JSON-RPC protocol for the agent daemon.
//!
//! The daemon speaks JSON-RPC 2.0 over stdio with the Tauri host.
//! `protocol` defines the message envelope types and the request /
//! notification payloads for V1. Transport and dispatch land in later
//! tickets; this crate only owns the wire contract.

#![deny(unsafe_code)]

pub mod protocol;
