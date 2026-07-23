use std::collections::BTreeMap;

use reimagine_core::model::{NodeCatalog, NodeDef, NodeTypeId};

use crate::builtins::all_builtin_defs;

#[derive(Debug, Clone)]
pub struct BuiltinNodeCatalog {
    defs: BTreeMap<NodeTypeId, NodeDef>,
}

impl BuiltinNodeCatalog {
    pub fn v1() -> Self {
        Self::new(all_builtin_defs())
    }

    pub fn new(defs: impl IntoIterator<Item = NodeDef>) -> Self {
        let defs = defs
            .into_iter()
            .map(|def| (def.type_id().clone(), def))
            .collect();
        Self { defs }
    }

    pub fn iter(&self) -> impl Iterator<Item = &NodeDef> {
        self.defs.values()
    }

    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }
}

impl Default for BuiltinNodeCatalog {
    fn default() -> Self {
        Self::v1()
    }
}

impl NodeCatalog for BuiltinNodeCatalog {
    fn get(&self, type_id: &NodeTypeId) -> Option<&NodeDef> {
        self.defs.get(type_id)
    }
}
