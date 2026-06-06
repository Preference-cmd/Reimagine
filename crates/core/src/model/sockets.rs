#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SocketKind {
    Model,
    Conditioning,
    Latent,
    Image,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketDef {
    pub id: String,
    pub kind: SocketKind,
    pub label: String,
}

impl SocketDef {
    pub fn new(id: impl Into<String>, kind: SocketKind, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            label: label.into(),
        }
    }
}
