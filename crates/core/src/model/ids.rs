macro_rules! id_type {
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

id_type!(WorkflowId);
id_type!(NodeId);
id_type!(EdgeId);
id_type!(NodeTypeId);
id_type!(SlotId);
id_type!(WorkflowInputId);
id_type!(WorkflowOutputId);
id_type!(RunId);
id_type!(ArtifactId);
id_type!(DiagnosticId);
id_type!(HistoryEntryId);
id_type!(CommandBatchId);
id_type!(ProposalId);
id_type!(ModelId);

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
