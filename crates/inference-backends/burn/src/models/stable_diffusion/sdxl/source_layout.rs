use std::path::{Path, PathBuf};

pub(crate) const DIFFUSERS_STYLE_SPLIT_SAFETENSORS: &str = "diffusers_style_split_safetensors";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BurnSdxlSourceSet {
    root: PathBuf,
}

impl BurnSdxlSourceSet {
    pub(crate) fn diffusers_style_split_safetensors(root: PathBuf) -> Self {
        Self { root }
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn diffusion_path(&self) -> PathBuf {
        self.root.join("unet/model.safetensors")
    }

    pub(crate) fn vae_path(&self) -> PathBuf {
        self.root.join("vae/model.safetensors")
    }

    pub(crate) fn text_encoder_path(&self) -> PathBuf {
        self.root.join("text_encoder/model.safetensors")
    }

    pub(crate) fn text_encoder_2_path(&self) -> PathBuf {
        self.root.join("text_encoder_2/model.safetensors")
    }
}

#[cfg(test)]
mod tests {
    use super::BurnSdxlSourceSet;

    #[test]
    fn resolves_expected_diffusers_style_split_paths() {
        let root = std::path::PathBuf::from("/models/sdxl-split");

        let set = BurnSdxlSourceSet::diffusers_style_split_safetensors(root.clone());

        assert_eq!(set.root(), root.as_path());
        assert_eq!(set.diffusion_path(), root.join("unet/model.safetensors"));
        assert_eq!(set.vae_path(), root.join("vae/model.safetensors"));
        assert_eq!(
            set.text_encoder_path(),
            root.join("text_encoder/model.safetensors")
        );
        assert_eq!(
            set.text_encoder_2_path(),
            root.join("text_encoder_2/model.safetensors")
        );
    }
}
