//! Name of the workspace a host and its services are bound to. Operators
//! (IPC, HTTP, future adapters) pass it in; app-host only compares and
//! displays it.

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct WorkspaceScope(String);

impl WorkspaceScope {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for WorkspaceScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for WorkspaceScope {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for WorkspaceScope {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_scope_api_roundtrip() {
        let scope = WorkspaceScope::new("ws-test");
        assert_eq!(scope.as_str(), "ws-test");
        assert_eq!(format!("{scope}"), "ws-test");
        let from_str: WorkspaceScope = "ws-from-str".into();
        assert_eq!(from_str.as_str(), "ws-from-str");
        let from_string: WorkspaceScope = String::from("ws-from-string").into();
        assert_eq!(from_string.as_str(), "ws-from-string");

        let serialized = serde_json::to_string(&scope).expect("serialize");
        assert_eq!(serialized, "\"ws-test\"");
        let deserialized: WorkspaceScope = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(deserialized, scope);
    }

    #[test]
    fn workspace_scope_compare() {
        let a = WorkspaceScope::new("a");
        let b = WorkspaceScope::new("b");
        assert_ne!(a, b);
        assert!(a < b);
    }
}
