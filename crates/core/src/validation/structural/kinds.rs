use crate::model::{ParamValue, SlotKind};

pub(super) fn param_value_matches_slot(value: &ParamValue, slot_kind: SlotKind) -> bool {
    matches!(
        (value, slot_kind),
        (ParamValue::String(_), SlotKind::String)
            | (ParamValue::Text(_), SlotKind::Text)
            | (ParamValue::Integer(_), SlotKind::Integer)
            | (ParamValue::Float(_), SlotKind::Float)
            | (ParamValue::Bool(_), SlotKind::Bool)
            | (ParamValue::Seed(_), SlotKind::Seed)
            | (ParamValue::Select(_), SlotKind::Select)
            | (ParamValue::Path(_), SlotKind::Path)
            | (ParamValue::ModelRef(_), SlotKind::ModelRef)
            | (ParamValue::Null, SlotKind::Null)
    )
}
