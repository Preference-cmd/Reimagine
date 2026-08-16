macro_rules! id_type {
    ($name:ident) => {
        id_type!(@inner $name, false);
    };
    ($name:ident, allow_slash) => {
        id_type!(@inner $name, true);
    };
    (@inner $name:ident, $allow_slash:expr) => {
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
                let s = id.into();
                assert!(!s.is_empty(), "ID must not be empty");
                assert!(s.is_ascii(), "ID must be ASCII");
                let banned: &[char] = if $allow_slash {
                    &['\\', '\0']
                } else {
                    &['/', '\\', '\0']
                };
                assert!(
                    !s.contains(banned),
                    "ID contains invalid characters"
                );
                Self(s)
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

id_type!(WorkflowId);
id_type!(ProjectId);
id_type!(NodeId);
id_type!(EdgeId);
id_type!(NodeTypeId);
id_type!(SlotId);
id_type!(WorkflowInputId);
id_type!(WorkflowOutputId);
id_type!(RunId);
// ArtifactId may carry path-like segments (the artifact reference is
// validated for path safety separately by the artifact access layer).
id_type!(ArtifactId, allow_slash);
// DiagnosticId mirrors `DiagnosticCode`, whose namespace separator is "/"
// (e.g. "CONFIG/MISSING_FILE", "AGENT/TOOL_PERMISSION_DENIED").
id_type!(DiagnosticId, allow_slash);
id_type!(HistoryEntryId);
id_type!(CommandBatchId);
id_type!(ProposalId);
id_type!(ModelId);
id_type!(BoardId);
id_type!(BoardItemId);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct WorkflowVersion(u64);

impl WorkflowVersion {
    pub fn new(version: u64) -> Self {
        Self(version)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for WorkflowVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for WorkflowVersion {
    fn from(version: u64) -> Self {
        Self(version)
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct BoardVersion(u64);

impl BoardVersion {
    pub fn new(version: u64) -> Self {
        Self(version)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for BoardVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for BoardVersion {
    fn from(version: u64) -> Self {
        Self(version)
    }
}
