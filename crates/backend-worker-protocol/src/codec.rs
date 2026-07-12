use std::fmt;
use std::io::{self, Read, Write};

use crate::WireMessage;

#[derive(Debug)]
pub enum CodecError {
    Io(io::Error),
    FrameTooLarge { declared: u32, maximum: u32 },
    PayloadLengthOverflow { actual: usize },
    MalformedJson(serde_json::Error),
    UnknownMessageKind(String),
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "worker protocol I/O failed: {error}"),
            Self::FrameTooLarge { declared, maximum } => write!(
                formatter,
                "worker protocol frame declares {declared} bytes, exceeding maximum {maximum}"
            ),
            Self::PayloadLengthOverflow { actual } => write!(
                formatter,
                "worker protocol payload has {actual} bytes, exceeding the u32 frame prefix"
            ),
            Self::MalformedJson(error) => {
                write!(formatter, "malformed worker protocol JSON: {error}")
            }
            Self::UnknownMessageKind(kind) => {
                write!(formatter, "unknown worker protocol message kind `{kind}`")
            }
        }
    }
}

impl std::error::Error for CodecError {}

impl From<io::Error> for CodecError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub struct FrameCodec {
    maximum_frame_bytes: u32,
}

impl FrameCodec {
    #[must_use]
    pub const fn new(maximum_frame_bytes: u32) -> Self {
        Self {
            maximum_frame_bytes,
        }
    }

    pub fn write(&self, writer: &mut impl Write, message: &WireMessage) -> Result<(), CodecError> {
        let payload = serde_json::to_vec(message).map_err(CodecError::MalformedJson)?;
        let declared = checked_payload_length(payload.len())?;
        if declared > self.maximum_frame_bytes {
            return Err(CodecError::FrameTooLarge {
                declared,
                maximum: self.maximum_frame_bytes,
            });
        }
        writer.write_all(&declared.to_be_bytes())?;
        writer.write_all(&payload)?;
        writer.flush()?;
        Ok(())
    }

    pub fn read(&self, reader: &mut impl Read) -> Result<WireMessage, CodecError> {
        let mut prefix = [0_u8; 4];
        reader.read_exact(&mut prefix)?;
        let declared = u32::from_be_bytes(prefix);
        if declared > self.maximum_frame_bytes {
            return Err(CodecError::FrameTooLarge {
                declared,
                maximum: self.maximum_frame_bytes,
            });
        }
        let mut payload = vec![0_u8; declared as usize];
        reader.read_exact(&mut payload)?;
        let value: serde_json::Value =
            serde_json::from_slice(&payload).map_err(CodecError::MalformedJson)?;
        if let Some(kind) = value.get("kind").and_then(serde_json::Value::as_str)
            && !WireMessage::is_known_kind(kind)
        {
            return Err(CodecError::UnknownMessageKind(kind.to_owned()));
        }
        serde_json::from_value(value).map_err(CodecError::MalformedJson)
    }
}

fn checked_payload_length(actual: usize) -> Result<u32, CodecError> {
    u32::try_from(actual).map_err(|_| CodecError::PayloadLengthOverflow { actual })
}

#[cfg(test)]
mod tests {
    use super::{CodecError, checked_payload_length};

    #[test]
    fn payload_length_must_fit_the_u32_prefix() {
        let actual = (u32::MAX as usize).checked_add(1).unwrap();
        assert!(matches!(
            checked_payload_length(actual),
            Err(CodecError::PayloadLengthOverflow { actual: value }) if value == actual
        ));
    }
}
