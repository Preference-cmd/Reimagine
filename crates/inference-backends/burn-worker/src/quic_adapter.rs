//! Blocking adapters for quinn async streams.
//!
//! Wraps `quinn::SendStream` and `quinn::RecvStream` behind
//! `std::io::Read` / `std::io::Write` so the existing synchronous
//! `serve_loop` can serve QUIC connections without modification.
//!
//! Each adapter holds an `Arc<tokio::runtime::Runtime>` and uses
//! `block_on` to bridge async stream operations into the blocking
//! thread that the serve loop runs on.

use std::io;
use std::sync::Arc;

use tokio::io::AsyncWriteExt;

/// Blocking reader that wraps a `quinn::RecvStream`.
pub struct QuicReadAdapter {
    recv: quinn::RecvStream,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl QuicReadAdapter {
    pub fn new(recv: quinn::RecvStream, runtime: Arc<tokio::runtime::Runtime>) -> Self {
        Self { recv, runtime }
    }
}

impl io::Read for QuicReadAdapter {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.runtime.block_on(async {
            match self.recv.read(buf).await {
                Ok(Some(n)) => Ok(n),
                // Stream finished cleanly — signal EOF (0 bytes read).
                Ok(None) => Ok(0),
                Err(e) => Err(io::Error::other(e)),
            }
        })
    }
}

/// Blocking writer that wraps a `quinn::SendStream`.
pub struct QuicWriteAdapter {
    send: quinn::SendStream,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl QuicWriteAdapter {
    pub fn new(send: quinn::SendStream, runtime: Arc<tokio::runtime::Runtime>) -> Self {
        Self { send, runtime }
    }
}

impl io::Write for QuicWriteAdapter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.runtime
            .block_on(async { self.send.write(buf).await.map_err(io::Error::other) })
    }

    fn flush(&mut self) -> io::Result<()> {
        self.runtime
            .block_on(async { self.send.flush().await.map_err(io::Error::other) })
    }
}
