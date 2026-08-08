//! Newline-delimited JSON-RPC transport over stdio.
//!
//! The daemon reads requests as one JSON object per line on stdin and
//! writes responses and notifications as JSON lines on stdout. All
//! writes go through a single `Mutex<W>` so concurrent turn
//! notifications serialize into complete, non-interleaved lines. The
//! transport is generic over `BufRead` / `Write` so tests can drive it
//! with in-memory buffers instead of the real stdin/stdout.

use std::io::{self, BufRead, Write};
use std::sync::{Mutex, PoisonError};

use serde::Serialize;
use serde_json::Value;

use crate::protocol::{JsonRpcError, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};

/// Response id used for parse errors.
///
/// JSON-RPC 2.0 mandates `id: null` when the request id cannot be
/// recovered from malformed input, but the `u64` response envelope
/// cannot carry null. The transport therefore uses this sentinel id for
/// every parse-error response.
pub const PARSE_ERROR_ID: u64 = 0;

/// Newline-delimited JSON-RPC transport.
///
/// `reader` supplies request lines; `writer` takes responses and
/// notifications behind a mutex so write access can be shared across
/// turn threads without interleaving.
#[derive(Debug)]
pub struct StdioTransport<R, W> {
    reader: R,
    writer: Mutex<W>,
}

impl<R: BufRead, W: Write + Send> StdioTransport<R, W> {
    /// Wrap an arbitrary reader/writer pair (in-memory buffers in tests).
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer: Mutex::new(writer),
        }
    }

    /// Unwrap the transport back into its reader and writer.
    pub fn into_parts(self) -> (R, W) {
        (
            self.reader,
            self.writer
                .into_inner()
                .unwrap_or_else(PoisonError::into_inner),
        )
    }

    /// Read the next request line, or `None` when stdin is exhausted.
    ///
    /// Blank lines are skipped. Lines that fail to deserialize as a
    /// `JsonRpcRequest` produce a parse-error response (see
    /// [`PARSE_ERROR_ID`]) and reading continues with the next line.
    /// I/O errors on the reader are treated as end-of-input.
    pub fn read_request(&mut self) -> Option<JsonRpcRequest<Value>> {
        let mut line = String::new();
        loop {
            line.clear();
            let read = self.reader.read_line(&mut line).ok()?;
            if read == 0 {
                return None;
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<JsonRpcRequest<Value>>(line) {
                Ok(request) => return Some(request),
                Err(_) => {
                    let error = JsonRpcResponse::<Value>::error(
                        PARSE_ERROR_ID,
                        JsonRpcError::parse_error(),
                    );
                    self.write_response(&error).ok()?;
                }
            }
        }
    }

    /// Serialize a response as one JSON line on the writer.
    pub fn write_response<T: Serialize>(&self, response: &JsonRpcResponse<T>) -> io::Result<()> {
        self.write_line(response)
    }

    /// Serialize a notification as one JSON line on the writer.
    pub fn write_notification<T: Serialize>(
        &self,
        notification: &JsonRpcNotification<T>,
    ) -> io::Result<()> {
        self.write_line(notification)
    }

    fn write_line<T: Serialize>(&self, message: &T) -> io::Result<()> {
        let mut writer = self.writer.lock().unwrap_or_else(PoisonError::into_inner);
        let line =
            serde_json::to_string(message).expect("protocol messages serialize without failure");
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.flush()
    }
}

impl StdioTransport<io::BufReader<io::Stdin>, io::Stdout> {
    /// Transport bound to the process stdin/stdout.
    pub fn from_stdio() -> Self {
        Self::new(io::BufReader::new(io::stdin()), io::stdout())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{METHOD_AGENT_CONTENT_DELTA, METHOD_PROVIDERS_LIST, METHOD_SESSION_LIST};
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::sync::Arc;

    use serde_json::json;

    #[test]
    fn request_response_roundtrip() {
        let input = br#"{"jsonrpc":"2.0","id":7,"method":"session.list","params":{}}"#.to_vec();
        let mut transport = StdioTransport::new(Cursor::new(input), Vec::new());

        let request = transport.read_request().expect("request parses");
        assert_eq!(request.id, 7);
        assert_eq!(request.method, METHOD_SESSION_LIST);
        assert_eq!(request.params, json!({}));

        let response = JsonRpcResponse::<Value>::success(7, json!({"sessions": []}));
        transport.write_response(&response).unwrap();

        let (_, output) = transport.into_parts();
        let output = String::from_utf8(output).unwrap();
        assert!(output.ends_with('\n'));
        let parsed: JsonRpcResponse<Value> = serde_json::from_str(output.trim()).unwrap();
        assert_eq!(parsed, response);
    }

    #[test]
    fn malformed_json_emits_parse_error_and_continues() {
        let input = b"this is not json\r\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"providers.list\",\"params\":{}}\n"
            .to_vec();
        let mut transport = StdioTransport::new(Cursor::new(input), Vec::new());

        let request = transport.read_request().expect("second line parses");
        assert_eq!(request.id, 2);
        assert_eq!(request.method, METHOD_PROVIDERS_LIST);

        let (_, output) = transport.into_parts();
        let line = String::from_utf8(output).unwrap();
        assert!(line.ends_with('\n'));
        let parsed: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(parsed["id"].as_u64(), Some(PARSE_ERROR_ID));
        assert_eq!(parsed["error"]["code"].as_i64(), Some(-32700));
        assert_eq!(parsed["error"]["message"], "parse error");
        assert!(parsed.get("result").is_none());
    }

    #[test]
    fn blank_lines_are_skipped() {
        let input =
            b"\r\n  \n{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"session.list\",\"params\":{}}\n"
                .to_vec();
        let mut transport = StdioTransport::new(Cursor::new(input), Vec::new());

        let request = transport.read_request().expect("request parses");
        assert_eq!(request.id, 3);

        let (_, output) = transport.into_parts();
        assert!(output.is_empty());
    }

    #[test]
    fn empty_stdin_returns_none() {
        let mut transport = StdioTransport::new(Cursor::new(Vec::new()), Vec::new());
        assert!(transport.read_request().is_none());
    }

    #[test]
    fn truncated_trailing_line_emits_parse_error_then_eof() {
        let input = br#"{"jsonrpc":"2.0","id":9"#.to_vec();
        let mut transport = StdioTransport::new(Cursor::new(input), Vec::new());

        assert!(transport.read_request().is_none());

        let (_, output) = transport.into_parts();
        let parsed: Value = serde_json::from_str(&String::from_utf8(output).unwrap()).unwrap();
        assert_eq!(parsed["id"].as_u64(), Some(PARSE_ERROR_ID));
        assert_eq!(parsed["error"]["code"].as_i64(), Some(-32700));
    }

    struct ChunkedReader {
        data: Vec<u8>,
        pos: usize,
    }

    impl io::Read for ChunkedReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.pos >= self.data.len() {
                return Ok(0);
            }
            let n = buf.len().min(3).min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[test]
    fn partial_reads_are_buffered_until_newline() {
        let input = r#"{"jsonrpc":"2.0","id":1,"method":"session.list","params":{}}
{"jsonrpc":"2.0","id":2,"method":"providers.list","params":{}}
"#;
        let reader = io::BufReader::new(ChunkedReader {
            data: input.as_bytes().to_vec(),
            pos: 0,
        });
        let mut transport = StdioTransport::new(reader, Vec::new());

        let first = transport.read_request().expect("first request parses");
        assert_eq!(first.id, 1);
        assert_eq!(first.method, METHOD_SESSION_LIST);

        let second = transport.read_request().expect("second request parses");
        assert_eq!(second.id, 2);
        assert_eq!(second.method, METHOD_PROVIDERS_LIST);

        assert!(transport.read_request().is_none());
    }

    #[test]
    fn concurrent_writes_do_not_interleave() {
        let transport = Arc::new(StdioTransport::new(Cursor::new(Vec::new()), Vec::new()));
        let threads = 8;
        let per_thread = 25;
        let mut handles = Vec::new();
        for thread in 0..threads {
            let transport = Arc::clone(&transport);
            handles.push(std::thread::spawn(move || {
                for i in 0..per_thread {
                    let notification = JsonRpcNotification::new(
                        METHOD_AGENT_CONTENT_DELTA,
                        json!({
                            "session_id": format!("s{thread}"),
                            "turn_id": format!("t{thread}-{i}"),
                            "text": "x".repeat(1 + (thread + i) % 64),
                        }),
                    );
                    transport.write_notification(&notification).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        let transport = Arc::try_unwrap(transport).unwrap();
        let (_, output) = transport.into_parts();
        let mut lines: Vec<&str> = std::str::from_utf8(&output).unwrap().split('\n').collect();
        assert_eq!(lines.pop(), Some(""));
        assert_eq!(lines.len(), threads * per_thread);
        let mut sessions = HashMap::new();
        for line in lines {
            let parsed: JsonRpcNotification<Value> = serde_json::from_str(line).unwrap();
            assert_eq!(parsed.jsonrpc, "2.0");
            let session = parsed.params["session_id"].as_str().unwrap();
            *sessions.entry(session.to_string()).or_insert(0) += 1;
        }
        assert_eq!(sessions.len(), threads);
        for count in sessions.values() {
            assert_eq!(*count, per_thread);
        }
    }
}
