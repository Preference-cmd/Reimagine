//! JSON-RPC protocol for the agent daemon.
//!
//! The daemon speaks JSON-RPC 2.0 over stdio with the Tauri host.
//! `protocol` defines the message envelope types and the request /
//! notification payloads for V1; `transport` owns the newline-delimited
//! stdio byte loop; `init` owns the daemon workspace host. Dispatch lands
//! in a later ticket.

#![deny(unsafe_code)]

pub mod init;
pub mod protocol;
pub mod transport;
