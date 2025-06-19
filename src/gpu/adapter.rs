

use wgpu::util::DeviceExt;
use crate::error::GpuError;

#[derive(Clone)]
pub struct GpuAdapter {
    adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub max_workgroup_size: u32
}

impl GpuAdapter {
    pub fn new() -> Result<Self, GpuError> {

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|e| GpuError::NoAdapter(e.to_string()) )?;
        println!("Running on Adapter: {:#?}", adapter.get_info());

        // Ensure adapter supports a compute shader
        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        if !downlevel_capabilities.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
            return Err(GpuError::NoShader);
        }

        // Create device and queue
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
            .map_err(|e| GpuError::NoDevice(e.to_string()))?;

        // Get limits
        let limits = device.limits();
        let max_workgroup_size: u32 = limits.max_compute_invocations_per_workgroup;

        Ok( Self { adapter, device, queue, max_workgroup_size })
    }
}


