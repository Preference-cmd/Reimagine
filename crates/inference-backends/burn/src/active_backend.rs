#[cfg(any(
    all(feature = "wgpu", feature = "flex"),
    all(feature = "wgpu", feature = "cuda"),
    all(feature = "flex", feature = "cuda"),
))]
compile_error!(
    "Burn backend features `wgpu`, `flex`, and `cuda` are mutually exclusive; \
     use exactly one of `wgpu`, `flex`, or `cuda`."
);

#[cfg(not(any(feature = "wgpu", feature = "flex", feature = "cuda")))]
compile_error!(
    "Burn backend requires an active production runtime feature: default `wgpu`, `--features cuda`, or `--features flex`."
);

#[cfg(feature = "cuda")]
pub(crate) type ActiveBurnBackend = burn_cuda::Cuda<f32, i32>;

#[cfg(feature = "cuda")]
pub(crate) fn active_device(device: &crate::device::BurnDevice) -> burn_cuda::CudaDevice {
    match device {
        crate::device::BurnDevice::Cuda(device) => device.clone(),
    }
}

#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
pub(crate) type ActiveBurnBackend = burn_wgpu::Wgpu;

#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
pub(crate) fn active_device(device: &crate::device::BurnDevice) -> burn_wgpu::WgpuDevice {
    match device {
        crate::device::BurnDevice::Wgpu(device) => device.clone(),
    }
}

#[cfg(all(not(any(feature = "cuda", feature = "wgpu")), feature = "flex"))]
pub(crate) type ActiveBurnBackend = burn_flex::Flex;

#[cfg(all(not(any(feature = "cuda", feature = "wgpu")), feature = "flex"))]
pub(crate) fn active_device(_device: &crate::device::BurnDevice) -> burn_flex::FlexDevice {
    burn_flex::FlexDevice
}
