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
        )]
        pub struct $name(String);

        // Ids from JSON (caller input, documents on disk) obey the same rules
        // as `try_new`: several ids become path segments, so an unchecked
        // `/` would reach outside its directory.
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let raw = String::deserialize(deserializer)?;
                Self::try_new(raw).map_err(serde::de::Error::custom)
            }
        }

        impl $name {
            pub fn try_new(id: impl Into<String>) -> Result<Self, &'static str> {
                let s = id.into();
                if s.is_empty() {
                    return Err("ID must not be empty");
                }
                if !s.is_ascii() {
                    return Err("ID must be ASCII");
                }
                let banned: &[char] = if $allow_slash {
                    &['\\', '\0']
                } else {
                    &['/', '\\', '\0']
                };
                if s.contains(banned) {
                    return Err("ID contains invalid characters");
                }
                Ok(Self(s))
            }

            pub fn new(id: impl Into<String>) -> Self {
                match Self::try_new(id) {
                    Ok(val) => val,
                    Err(msg) => panic!("{msg}"),
                }
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
id_type!(CollectionId);
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
id_type!(AssetId);
id_type!(StepId);
id_type!(DynamicWorkflowId);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct DocumentVersion(u64);

impl DocumentVersion {
    pub fn new(version: u64) -> Self {
        Self(version)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for DocumentVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for DocumentVersion {
    fn from(version: u64) -> Self {
        Self(version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_applies_the_same_rules_as_try_new() {
        let id: CollectionId = serde_json::from_str("\"col-1\"").expect("plain id");
        assert_eq!(id.as_str(), "col-1");

        for raw in ["\"../other/col-1\"", "\"\"", "\"a\\\\b\"", "\"列\""] {
            assert!(
                serde_json::from_str::<CollectionId>(raw).is_err(),
                "{raw} must be rejected"
            );
        }

        let artifact: ArtifactId = serde_json::from_str("\"runs/a/b.png\"").expect("slash allowed");
        assert_eq!(artifact.as_str(), "runs/a/b.png");
    }
}
