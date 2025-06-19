
use std::collections::HashMap;
use std::sync::Mutex;
use crate::error::GpuError;
use super::{GpuAdapter, GpuLayout, GpuResourceLayout, GpuMatrix, GpuModule, GpuAlgorithm, GpuParams};

#[derive(Default)]
pub struct GpuStation {
    cargo: Mutex<GpuStationCargo>
}

#[derive(Default)]
struct GpuStationCargo {
    adapter: Option<GpuAdapter>,
    layouts: HashMap<GpuLayout, GpuResourceLayout>,
    workgroups: HashMap<GpuWorkgroupParams, GpuWorkgroup>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct GpuWorkgroupParams {
    algorithm: GpuAlgorithm,
    workgroup_size: usize // must be 64, 128, 256, 512, or 1024
}

#[derive(Clone)]
pub struct GpuWorkgroup {
    pub shader: wgpu::ShaderModule,
    pub pipeline: wgpu::ComputePipeline
}

impl GpuStation {
    pub fn new() -> Self {
        Self {
            cargo: Mutex::new(GpuStationCargo::default())
        }
    }

    pub fn get_adapter(&self) -> Result<GpuAdapter, GpuError> {
        let mut cargo = self.cargo.lock()
            .map_err(|e| GpuError::MutexLock(e.to_string()))?;

        if let Some(ref adapter) = cargo.adapter {
            return Ok(adapter.clone());
        }
        let adapter = GpuAdapter::new()?;
        cargo.adapter = Some(adapter.clone());
        Ok(adapter)
    }

    pub fn get_layout(&self, layout: GpuLayout, adapter: &GpuAdapter) -> Result<GpuResourceLayout, GpuError> {
        let mut cargo = self.cargo.lock()
            .map_err(|e| GpuError::MutexLock(e.to_string()))?;

        if let Some(group) = cargo.layouts.get(&layout) {
            return Ok(group.clone());
        }

        let group = layout.create_resource_layout(&adapter);
        cargo.layouts.insert(layout, group.clone());
        Ok(group)
    }

    pub fn get_workgroup<M>(&self, module: &M, matrix: &GpuMatrix, params: &GpuParams, adapter: &GpuAdapter, resources: &GpuResourceLayout) -> Result<GpuWorkgroup, GpuError> 
        where M: GpuModule,
    {

        let mut cargo = self.cargo.lock()
            .map_err(|e| GpuError::MutexLock(e.to_string()))?;

        let wg_params = GpuWorkgroupParams::new(params.algorithm, matrix.get_workgroup_size())?;

        if let Some(group) = cargo.workgroups.get(&wg_params) {
            return Ok(group.clone());
        }

        let shader_source = module.get_wgsl_code(&matrix, &params);
        let shader = adapter.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline = adapter.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&resources.pipeline_layout),
            module: &shader,
            entry_point: Some(&params.entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let group = GpuWorkgroup {
            shader,
            pipeline
        };
        cargo.workgroups.insert(wg_params, group.clone());

        Ok(group)
    }
}

impl GpuWorkgroupParams {
    pub fn new(algorithm: GpuAlgorithm, workgroup_size: usize) -> Result<Self, GpuError> { 
        if ![64, 128, 256, 512, 1024].contains(&workgroup_size) {
            return Err(GpuError::InvalidWorkgroupSize);
        }

        Ok( Self { algorithm, workgroup_size })
    }
}



