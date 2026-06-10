//! String-typed identifier newtypes for the agent runtime.

macro_rules! agent_id_type {
    ($name:ident) => {
        #[derive(
            Debug,
            Clone,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            serde::Serialize,
            serde::Deserialize,
        )]
        pub struct $name(String);

        impl $name {
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }
    };
}

agent_id_type!(AgentSessionId);
agent_id_type!(WorkspaceScope);
agent_id_type!(ToolName);
agent_id_type!(ProviderName);
agent_id_type!(ModelName);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_newtypes_roundtrip() {
        let sid = AgentSessionId::new("sess-1");
        assert_eq!(sid.as_str(), "sess-1");
        assert_eq!(format!("{sid}"), "sess-1");
        let _: AgentSessionId = "sess-1".into();
        let _: AgentSessionId = String::from("sess-1").into();

        let scope = WorkspaceScope::new("ws-alpha");
        assert_eq!(scope.as_str(), "ws-alpha");

        let tool = ToolName::new("workflow.preview_commands");
        assert_eq!(tool.as_str(), "workflow.preview_commands");

        let provider = ProviderName::new("openai");
        assert_eq!(provider.as_str(), "openai");

        let model = ModelName::new("gpt-4o-mini");
        assert_eq!(model.as_str(), "gpt-4o-mini");
    }

    #[test]
    fn id_newtypes_compare() {
        let a = AgentSessionId::new("a");
        let b = AgentSessionId::new("b");
        assert_ne!(a, b);
        assert!(a < b);
    }
}
