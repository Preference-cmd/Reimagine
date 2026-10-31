//! The six operation implementations, plus the option readers they share.

use mod_sdk::diagnostic::{Diagnostic, caller_error};
use mod_sdk::serde_json::Value;

use crate::ID;

pub mod inspect;
pub mod join;
pub mod normalize;
pub mod replace;
pub mod split;
pub mod template;

/// Read an optional boolean option, applying the schema default.
fn boolean(input: &Value, key: &str, default: bool) -> Result<bool, Diagnostic> {
    match input.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(caller_error(
            ID,
            "INVALID_OPTION",
            format!("`{key}` must be a boolean"),
        )),
    }
}

/// Read an optional string option that must be one of `allowed`.
fn choice<'a>(
    input: &'a Value,
    key: &str,
    allowed: &[&str],
    default: &'a str,
) -> Result<&'a str, Diagnostic> {
    match input.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::String(value)) if allowed.contains(&value.as_str()) => Ok(value),
        Some(_) => Err(caller_error(
            ID,
            "INVALID_OPTION",
            format!("`{key}` must be one of {allowed:?}"),
        )),
    }
}
