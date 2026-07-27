#[cfg(any(
    all(feature = "wgpu", feature = "flex"),
    all(feature = "wgpu", feature = "cuda"),
    all(feature = "wgpu", feature = "rocm"),
    all(feature = "flex", feature = "cuda"),
    all(feature = "flex", feature = "rocm"),
    all(feature = "cuda", feature = "rocm"),
))]
compile_error!(
    "Burn backend features `wgpu`, `flex`, `cuda`, and `rocm` are mutually exclusive; \
     use exactly one of `wgpu`, `flex`, `cuda`, or `rocm`."
);

#[cfg(not(any(feature = "wgpu", feature = "flex", feature = "cuda", feature = "rocm")))]
compile_error!(
    "Burn backend requires an active production runtime feature: default `wgpu`, `--features cuda`, `--features rocm`, or `--features flex`."
);

#[cfg(feature = "cuda")]
pub(crate) type ActiveBurnBackend = burn_cuda::Cuda<f32, i32>;

#[cfg(feature = "cuda")]
pub(crate) fn active_device(device: &crate::device::BurnDevice) -> burn_cuda::CudaDevice {
    match device {
        crate::device::BurnDevice::Cuda(device) => device.clone(),
    }
}

#[cfg(feature = "rocm")]
pub(crate) type ActiveBurnBackend = burn_rocm::Rocm;

#[cfg(feature = "rocm")]
pub(crate) fn active_device(device: &crate::device::BurnDevice) -> burn_rocm::RocmDevice {
    match device {
        crate::device::BurnDevice::Rocm(device) => device.clone(),
    }
}

#[cfg(all(not(any(feature = "cuda", feature = "rocm")), feature = "wgpu"))]
pub(crate) type ActiveBurnBackend = burn_wgpu::Wgpu;

#[cfg(all(not(any(feature = "cuda", feature = "rocm")), feature = "wgpu"))]
pub(crate) fn active_device(device: &crate::device::BurnDevice) -> burn_wgpu::WgpuDevice {
    match device {
        crate::device::BurnDevice::Wgpu(device) => device.clone(),
    }
}

#[cfg(all(
    not(any(feature = "cuda", feature = "rocm", feature = "wgpu")),
    feature = "flex"
))]
pub(crate) type ActiveBurnBackend = burn_flex::Flex;

#[cfg(all(
    not(any(feature = "cuda", feature = "rocm", feature = "wgpu")),
    feature = "flex"
))]
pub(crate) fn active_device(_device: &crate::device::BurnDevice) -> burn_flex::FlexDevice {
    burn_flex::FlexDevice
}
