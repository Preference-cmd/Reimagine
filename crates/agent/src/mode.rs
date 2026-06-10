//! Agent mode: editor-only auto-apply vs. proposal-only build.

/// Agent execution mode.
///
/// `Agent` may auto-apply low-risk, reversible, editor-only edits after a
/// successful preview. `Build` produces a complete proposal that a human
/// must approve before any commands are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AgentMode {
    /// Auto-apply mode: tool policy may allow low-risk, reversible,
    /// editor-only commands.
    Agent,
    /// Build mode: tools produce proposals; application requires human or
    /// host approval.
    Build,
}

impl AgentMode {
    /// `true` when the mode is `Agent`.
    pub fn is_agent(self) -> bool {
        matches!(self, Self::Agent)
    }

    /// `true` when the mode is `Build`.
    pub fn is_build(self) -> bool {
        matches!(self, Self::Build)
    }
}

impl std::fmt::Display for AgentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Agent => f.write_str("agent"),
            Self::Build => f.write_str("build"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_predicates() {
        assert!(AgentMode::Agent.is_agent());
        assert!(!AgentMode::Agent.is_build());
        assert!(!AgentMode::Build.is_agent());
        assert!(AgentMode::Build.is_build());
    }

    #[test]
    fn mode_display() {
        assert_eq!(format!("{}", AgentMode::Agent), "agent");
        assert_eq!(format!("{}", AgentMode::Build), "build");
    }
}
