
use wgpu::util::DeviceExt;
use crate::numbers::basenum::Number;
use super::{GpuAdapter, GpuParams, GpuMatrix};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum GpuBuffer {
    Samples,
    Targets,
    Weights,
    TempStorage,
    Params,
    Download
}

impl GpuBuffer {
    pub fn included_in_bind_group(&self) -> bool {
        match self {
            GpuBuffer::Download => false,
            _ => true
        }
    }

    pub fn is_read_only(&self) -> bool {
        match self {
            GpuBuffer::Weights => false,
            GpuBuffer::TempStorage => false,
            _ => true
        }
    }
}


pub fn create_samples(adapter: &GpuAdapter, matrix: &GpuMatrix) -> wgpu::Buffer {
    adapter.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Samples"),
        contents: bytemuck::cast_slice(&matrix.data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

pub fn create_targets<T>(adapter: &GpuAdapter, targets: &Vec<T>) -> wgpu::Buffer 
    where T: Number + Ord
{

    let u_targets = targets.iter().filter_map(|&val| val.to_u32()).collect::<Vec<u32>>();
    adapter.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Targets"),
        contents: bytemuck::cast_slice(&u_targets),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

pub fn create_weights(adapter: &GpuAdapter, num_features: usize) -> wgpu::Buffer {
    let zeros = vec![0.0f32; num_features];
    adapter.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Weights"),
        contents: bytemuck::cast_slice(&zeros),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}


pub fn create_temp_storage(adapter: &GpuAdapter, buffer_size: u64) -> wgpu::Buffer {
    adapter.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Temp Storage"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub fn create_params(adapter: &GpuAdapter, data: &Vec<u32>) -> wgpu::Buffer {

    adapter.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params"),
        contents: bytemuck::cast_slice(&data.as_slice()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

pub fn create_download(adapter: &GpuAdapter, buffer_size: u64) -> wgpu::Buffer {
    adapter.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Download"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}




