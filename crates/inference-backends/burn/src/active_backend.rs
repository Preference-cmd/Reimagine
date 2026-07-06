#[cfg(all(feature = "wgpu", feature = "flex"))]
compile_error!(
    "Burn backend features `wgpu` and `flex` are mutually exclusive; use default `wgpu` or `--no-default-features --features flex`."
);

#[cfg(not(any(feature = "wgpu", feature = "flex")))]
compile_error!(
    "Burn backend requires an active production runtime feature: default `wgpu` or `--features flex`."
);

#[cfg(feature = "wgpu")]
pub(crate) type ActiveBurnBackend = burn_wgpu::Wgpu;

#[cfg(all(not(feature = "wgpu"), feature = "flex"))]
pub(crate) type ActiveBurnBackend = burn_flex::Flex;

#[cfg(feature = "wgpu")]
pub(crate) fn active_device(device: &crate::device::BurnDevice) -> burn_wgpu::WgpuDevice {
    match device {
        crate::device::BurnDevice::Wgpu(device) => device.clone(),
    }
}

#[cfg(all(not(feature = "wgpu"), feature = "flex"))]
pub(crate) fn active_device(_device: &crate::device::BurnDevice) -> burn_flex::FlexDevice {
    burn_flex::FlexDevice
}
