/// Caller-supplied timestamp (string newtype). Core does not read the clock.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Timestamp(String);

impl Timestamp {
    pub fn new(ts: impl Into<String>) -> Self {
        Self(ts.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Timestamp {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for Timestamp {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_display_and_roundtrip() {
        let ts = Timestamp::new("2026-06-08T12:00:00Z");
        assert_eq!(ts.as_str(), "2026-06-08T12:00:00Z");
        assert_eq!(ts.to_string(), "2026-06-08T12:00:00Z");
        assert_eq!(Timestamp::from("2026-01-01"), Timestamp::new("2026-01-01"));
        assert_eq!(
            Timestamp::from("2026-01-01".to_string()),
            Timestamp::new("2026-01-01")
        );
    }
}
