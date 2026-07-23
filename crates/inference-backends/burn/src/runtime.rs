use burn_store::{ApplyResult, ModuleSnapshot, ModuleStore};
use burn_tensor::backend::Backend;

#[derive(Debug, Clone)]
pub(crate) struct BurnRuntime<B: Backend> {
    device: B::Device,
}

impl<B: Backend> BurnRuntime<B> {
    pub(crate) fn new(device: B::Device) -> Self {
        Self { device }
    }

    #[allow(dead_code)]
    pub(crate) fn device(&self) -> &B::Device {
        &self.device
    }

    #[allow(dead_code)]
    pub(crate) fn load_module_store<M, S>(
        &self,
        module: &mut M,
        store: &mut S,
    ) -> Result<ApplyResult, S::Error>
    where
        M: ModuleSnapshot<B>,
        S: ModuleStore,
    {
        module.load_from(store)
    }
}

#[cfg(test)]
mod tests {
    use burn_core as burn;
    use burn_core::module::Module;
    use burn_nn::{Linear, LinearConfig};
    use burn_store::{ModuleSnapshot, SafetensorsStore};
    use burn_tensor::Tensor;

    use crate::active_backend::{ActiveBurnBackend, active_device};
    use crate::config::BurnBackendConfig;

    use super::BurnRuntime;

    #[derive(Module, Debug)]
    struct TinyModule<B: burn_tensor::backend::Backend> {
        linear: Linear<B>,
    }

    impl<B: burn_tensor::backend::Backend> TinyModule<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                linear: LinearConfig::new(2, 3).init(device),
            }
        }

        fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            self.linear.forward(input)
        }
    }

    #[test]
    fn typed_runtime_loads_module_through_burn_store() {
        type B = ActiveBurnBackend;

        let config = BurnBackendConfig::new("/models", "/output");
        let runtime = BurnRuntime::<B>::new(active_device(config.device()));
        let source = TinyModule::<B>::init(runtime.device());
        let mut target = TinyModule::<B>::init(runtime.device());
        let input = Tensor::<B, 2>::ones([1, 2], runtime.device());
        let source_output = source.forward(input.clone());
        let mut save_store = SafetensorsStore::from_bytes(None);
        source
            .save_into(&mut save_store)
            .expect("source module should save into burn-store");
        let bytes = save_store
            .get_bytes()
            .expect("saved safetensors bytes should be readable");
        let mut load_store = SafetensorsStore::from_bytes(Some(bytes));

        let result = runtime
            .load_module_store(&mut target, &mut load_store)
            .expect("target module should load from burn-store");

        assert!(result.errors.is_empty(), "unexpected load errors: {result}");
        assert_eq!(source_output.to_data(), target.forward(input).to_data());
    }
}
