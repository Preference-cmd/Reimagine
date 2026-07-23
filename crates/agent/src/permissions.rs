//! Permission and risk classification for Agent tools.

use std::collections::BTreeSet;

/// Logical permission that a tool requires. Tool policy compares the
/// tool's required permission against the permissions carried in the
/// `ToolContext` for the current session.
///
/// Names are stable string identifiers that are used both for tool
/// registration and for policy comparison. Concrete app-host tools
/// introduce their own permission names (e.g. `"workflow.read"`,
/// `"workflow.write"`, `"model.read"`).
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ToolPermission(String);

impl ToolPermission {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ToolPermission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ToolPermission {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ToolPermission {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

/// Tool risk classification. Tool policy may use this to constrain which
/// modes may invoke a tool, or to require additional policy mediation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ToolRiskLevel {
    /// Pure read-only tool. No side effects.
    Read,
    /// Editor-only mutation. Reversible through workflow history.
    Editor,
    /// External-effect operation requiring human/host approval.
    External,
}

impl ToolRiskLevel {
    /// `true` if the tool performs side effects that must be mediated
    /// beyond editor-only history.
    pub fn is_external(self) -> bool {
        matches!(self, Self::External)
    }
}

impl std::fmt::Display for ToolRiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => f.write_str("read"),
            Self::Editor => f.write_str("editor"),
            Self::External => f.write_str("external"),
        }
    }
}

/// A set of permissions carried by a `ToolContext`. Stored as a sorted set
/// to give deterministic equality and to avoid exposing an external hash
/// collection.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PermissionSet {
    permissions: BTreeSet<ToolPermission>,
}

impl PermissionSet {
    /// Create an empty permission set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when the set contains `permission`.
    pub fn contains(&self, permission: &ToolPermission) -> bool {
        self.permissions.contains(permission)
    }

    /// Returns `true` if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.permissions.is_empty()
    }

    /// Number of permissions in the set.
    pub fn len(&self) -> usize {
        self.permissions.len()
    }

    /// Iterate over permissions in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = &ToolPermission> {
        self.permissions.iter()
    }

    /// Add a permission. Returns `true` if the set did not already contain
    /// the permission.
    pub fn insert(&mut self, permission: ToolPermission) -> bool {
        self.permissions.insert(permission)
    }
}

impl FromIterator<ToolPermission> for PermissionSet {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ToolPermission>,
    {
        Self {
            permissions: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_set_contains_and_insert() {
        let mut set = PermissionSet::new();
        assert!(set.is_empty());
        assert!(set.insert(ToolPermission::new("workflow.read")));
        assert!(!set.insert(ToolPermission::new("workflow.read")));
        assert!(set.contains(&ToolPermission::new("workflow.read")));
        assert!(!set.contains(&ToolPermission::new("workflow.write")));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn permission_set_iterates_in_sorted_order() {
        let set = PermissionSet::from_iter([
            ToolPermission::new("workflow.write"),
            ToolPermission::new("model.read"),
            ToolPermission::new("workflow.read"),
        ]);
        let names: Vec<&str> = set.iter().map(|p| p.as_str()).collect();
        assert_eq!(names, vec!["model.read", "workflow.read", "workflow.write"]);
    }

    #[test]
    fn risk_level_predicate() {
        assert!(!ToolRiskLevel::Read.is_external());
        assert!(!ToolRiskLevel::Editor.is_external());
        assert!(ToolRiskLevel::External.is_external());
    }
}
