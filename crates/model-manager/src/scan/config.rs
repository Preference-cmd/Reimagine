use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanConfig {
    recursive: bool,
    ignore_hidden: bool,
}

impl ScanConfig {
    pub fn new(recursive: bool, ignore_hidden: bool) -> Self {
        Self {
            recursive,
            ignore_hidden,
        }
    }

    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn with_ignore_hidden(mut self, ignore_hidden: bool) -> Self {
        self.ignore_hidden = ignore_hidden;
        self
    }

    pub fn recursive(&self) -> bool {
        self.recursive
    }

    pub fn ignore_hidden(&self) -> bool {
        self.ignore_hidden
    }
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            recursive: true,
            ignore_hidden: true,
        }
    }
}
