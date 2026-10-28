//! JSON binding paths for dynamic workflow inputs, params and step outputs.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The single object key that marks a binding.
pub const BINDING_KEY: &str = "$from";

/// A single-key `{ "$from": "<root>.<name>[.<path>...]" }` reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    #[serde(rename = "$from")]
    from: String,
}

impl Binding {
    /// Build a binding from its `$from` path.
    pub fn new(from: impl Into<String>) -> Result<Self, BindingError> {
        let from = from.into();
        if from.is_empty() {
            return Err(BindingError::EmptyPath);
        }
        let mut segments = from.split('.');
        let root = segments.next().unwrap_or_default();
        if !matches!(root, "inputs" | "params" | "steps") {
            return Err(BindingError::UnknownRoot(root.to_owned()));
        }
        match segments.next() {
            Some(name) if !name.is_empty() => {}
            _ => return Err(BindingError::MissingName),
        }
        if segments.any(str::is_empty) {
            return Err(BindingError::EmptySegment);
        }
        Ok(Self { from })
    }

    /// Parse a JSON value as a binding object.
    pub fn parse(value: &Value) -> Result<Self, BindingError> {
        let Value::Object(map) = value else {
            return Err(BindingError::NotAnObject);
        };
        let Some(from) = map.get(BINDING_KEY) else {
            return Err(BindingError::MissingFrom);
        };
        if map.len() != 1 {
            return Err(BindingError::ExtraKeys);
        }
        let Value::String(from) = from else {
            return Err(BindingError::NonStringPath);
        };
        Self::new(from.clone())
    }

    /// The original `$from` path.
    pub fn from(&self) -> &str {
        &self.from
    }

    /// The first path segment: `inputs`, `params` or `steps`.
    pub fn root(&self) -> &str {
        self.from.split('.').next().unwrap_or_default()
    }

    /// The second path segment: an input, param or step name.
    pub fn name(&self) -> &str {
        self.from.split('.').nth(1).unwrap_or_default()
    }

    fn remainder(&self) -> impl Iterator<Item = &str> {
        self.from.split('.').skip(2)
    }

    /// Resolve this binding against the current inputs, params and step outputs.
    pub fn resolve(&self, scope: &BindingScope<'_>) -> Result<Value, BindingError> {
        match self.root() {
            "inputs" => {
                let value = scope
                    .inputs
                    .get(self.name())
                    .ok_or_else(|| BindingError::Unresolved(self.from.clone()))?;
                walk(value, self.remainder(), &self.from)
            }
            "params" => {
                let segments = std::iter::once(self.name()).chain(self.remainder());
                walk(scope.params, segments, &self.from)
            }
            "steps" => {
                let value = scope
                    .steps
                    .get(self.name())
                    .ok_or_else(|| BindingError::Unresolved(self.from.clone()))?;
                walk(value, self.remainder(), &self.from)
            }
            _ => Err(BindingError::UnknownRoot(self.root().to_owned())),
        }
    }
}

impl fmt::Display for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "$from:{}", self.from)
    }
}

/// Values a [`Binding`] resolves against.
pub struct BindingScope<'a> {
    pub inputs: &'a BTreeMap<String, Value>,
    pub params: &'a Value,
    pub steps: &'a BTreeMap<String, Value>,
}

/// Why a binding could not be parsed or resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingError {
    NotAnObject,
    MissingFrom,
    ExtraKeys,
    NonStringPath,
    EmptyPath,
    UnknownRoot(String),
    MissingName,
    EmptySegment,
    Unresolved(String),
}

impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAnObject => write!(f, "binding must be a JSON object"),
            Self::MissingFrom => write!(f, "binding object has no `$from` key"),
            Self::ExtraKeys => write!(f, "binding object must contain only `$from`"),
            Self::NonStringPath => write!(f, "`$from` must be a string"),
            Self::EmptyPath => write!(f, "`$from` path is empty"),
            Self::UnknownRoot(root) => write!(
                f,
                "`$from` root `{root}` must be one of `inputs`, `params` or `steps`"
            ),
            Self::MissingName => write!(f, "`$from` path needs a name after the root"),
            Self::EmptySegment => write!(f, "`$from` path contains an empty segment"),
            Self::Unresolved(path) => write!(f, "binding `{path}` has no value"),
        }
    }
}

impl std::error::Error for BindingError {}

/// Collect every binding nested in `value`, in document order.
pub fn collect_bindings(value: &Value) -> Result<Vec<Binding>, BindingError> {
    let mut bindings = Vec::new();
    collect_into(value, &mut bindings)?;
    Ok(bindings)
}

fn collect_into(value: &Value, bindings: &mut Vec<Binding>) -> Result<(), BindingError> {
    match value {
        Value::Object(map) if map.contains_key(BINDING_KEY) => {
            bindings.push(Binding::parse(value)?);
        }
        Value::Object(map) => {
            for value in map.values() {
                collect_into(value, bindings)?;
            }
        }
        Value::Array(items) => {
            for value in items {
                collect_into(value, bindings)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Resolve every binding nested in `value`; other objects and arrays are
/// rebuilt recursively.
pub fn resolve_value(value: &Value, scope: &BindingScope<'_>) -> Result<Value, BindingError> {
    match value {
        Value::Object(map) if map.contains_key(BINDING_KEY) => {
            Binding::parse(value)?.resolve(scope)
        }
        Value::Object(map) => {
            let mut resolved = Map::with_capacity(map.len());
            for (key, value) in map {
                resolved.insert(key.clone(), resolve_value(value, scope)?);
            }
            Ok(Value::Object(resolved))
        }
        Value::Array(items) => {
            let mut resolved = Vec::with_capacity(items.len());
            for value in items {
                resolved.push(resolve_value(value, scope)?);
            }
            Ok(Value::Array(resolved))
        }
        other => Ok(other.clone()),
    }
}

fn walk<'v, 'p>(
    mut value: &'v Value,
    segments: impl Iterator<Item = &'p str>,
    path: &str,
) -> Result<Value, BindingError> {
    for segment in segments {
        value = match value {
            Value::Object(map) => map
                .get(segment)
                .ok_or_else(|| BindingError::Unresolved(path.to_owned()))?,
            Value::Array(items) => {
                let index: usize = segment
                    .parse()
                    .map_err(|_| BindingError::Unresolved(path.to_owned()))?;
                items
                    .get(index)
                    .ok_or_else(|| BindingError::Unresolved(path.to_owned()))?
            }
            _ => return Err(BindingError::Unresolved(path.to_owned())),
        };
    }
    Ok(value.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scope<'a>(
        inputs: &'a BTreeMap<String, Value>,
        params: &'a Value,
        steps: &'a BTreeMap<String, Value>,
    ) -> BindingScope<'a> {
        BindingScope {
            inputs,
            params,
            steps,
        }
    }

    #[test]
    fn parse_requires_a_single_from_key() {
        let binding = Binding::parse(&json!({ "$from": "inputs.doc" })).expect("valid");
        assert_eq!(binding.root(), "inputs");
        assert_eq!(binding.name(), "doc");

        assert_eq!(
            Binding::parse(&json!({ "$from": "inputs.doc", "extra": 1 })),
            Err(BindingError::ExtraKeys)
        );
        assert_eq!(
            Binding::parse(&json!({ "$from": 3 })),
            Err(BindingError::NonStringPath)
        );
        assert_eq!(Binding::parse(&json!({})), Err(BindingError::MissingFrom));
        assert_eq!(
            Binding::parse(&json!("inputs.doc")),
            Err(BindingError::NotAnObject)
        );
    }

    #[test]
    fn parse_rejects_bad_paths() {
        assert_eq!(Binding::new(""), Err(BindingError::EmptyPath));
        assert_eq!(
            Binding::new("output.doc"),
            Err(BindingError::UnknownRoot("output".to_owned()))
        );
        assert_eq!(Binding::new("inputs"), Err(BindingError::MissingName));
        assert_eq!(Binding::new("inputs."), Err(BindingError::MissingName));
        assert_eq!(
            Binding::new("steps.norm..doc"),
            Err(BindingError::EmptySegment)
        );
    }

    #[test]
    fn resolve_reads_inputs_params_steps_and_nested_paths() {
        let inputs = BTreeMap::from([("doc".to_owned(), json!("asset-7"))]);
        let params = json!({ "glossary": ["a", "b"], "meta": { "author": "Ada" } });
        let steps = BTreeMap::from([("norm".to_owned(), json!({ "doc": "asset-8", "n": 2 }))]);
        let scope = scope(&inputs, &params, &steps);

        assert_eq!(
            Binding::new("inputs.doc").unwrap().resolve(&scope).unwrap(),
            json!("asset-7")
        );
        assert_eq!(
            Binding::new("params.glossary.1")
                .unwrap()
                .resolve(&scope)
                .unwrap(),
            json!("b")
        );
        assert_eq!(
            Binding::new("params.meta.author")
                .unwrap()
                .resolve(&scope)
                .unwrap(),
            json!("Ada")
        );
        assert_eq!(
            Binding::new("steps.norm.doc")
                .unwrap()
                .resolve(&scope)
                .unwrap(),
            json!("asset-8")
        );
        assert_eq!(
            Binding::new("inputs.missing").unwrap().resolve(&scope),
            Err(BindingError::Unresolved("inputs.missing".to_owned()))
        );
        assert_eq!(
            Binding::new("steps.norm.missing").unwrap().resolve(&scope),
            Err(BindingError::Unresolved("steps.norm.missing".to_owned()))
        );
    }

    #[test]
    fn resolve_value_rebuilds_objects_and_arrays() {
        let inputs = BTreeMap::new();
        let params = json!({ "author": "Ada" });
        let steps = BTreeMap::from([(
            "outline".to_owned(),
            json!({ "title": { "text": "Intro" } }),
        )]);
        let scope = scope(&inputs, &params, &steps);

        let input = json!({
            "values": {
                "title": { "$from": "steps.outline.title.text" },
                "author": { "$from": "params.author" }
            },
            "literal": [1, true, null],
            "nested": [{ "x": { "$from": "params.author" } }]
        });
        let resolved = resolve_value(&input, &scope).expect("resolves");
        assert_eq!(
            resolved,
            json!({
                "values": { "title": "Intro", "author": "Ada" },
                "literal": [1, true, null],
                "nested": [{ "x": "Ada" }]
            })
        );
    }

    #[test]
    fn collect_bindings_walks_nested_values() {
        let input = json!({
            "doc": { "$from": "inputs.doc" },
            "list": [{ "$from": "params.a" }, 3],
            "plain": { "key": { "$from": "steps.one.out" } }
        });
        let paths: Vec<String> = collect_bindings(&input)
            .unwrap()
            .into_iter()
            .map(|b| b.from().to_owned())
            .collect();
        assert_eq!(
            paths,
            vec![
                "inputs.doc".to_owned(),
                "params.a".to_owned(),
                "steps.one.out".to_owned()
            ]
        );
    }
}
