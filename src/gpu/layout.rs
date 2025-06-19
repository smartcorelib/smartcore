
use super::{GpuAdapter, GpuBuffer};

#[derive(Default, Copy, Clone, Eq, PartialEq, Hash)]
pub enum GpuLayout {
    #[default]
    Supervised,    // Samples + Targets + Weights + TempStorage + Params
    //Clustering,    // Samples + Centroids + Assignments + Params  
    //Decomposition, // Samples + Vectors + Values + TempStorage + Params
}

#[derive(Clone)]
pub struct GpuResourceLayout {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
}

impl GpuLayout {
    pub fn create_resource_layout(&self, adapter: &GpuAdapter) -> GpuResourceLayout {

        // Bind group layout
        let bind_group_layout = self.create_bind_group_layout(&adapter);

        // Pipeline layout
        let pipeline_layout = adapter.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        GpuResourceLayout {
            bind_group_layout,
            pipeline_layout
        }
    }

    pub fn get_buffer_templates(&self) -> Vec<GpuBuffer> {
        match self {
            GpuLayout::Supervised => vec![GpuBuffer::Samples, GpuBuffer::Targets, GpuBuffer::Weights, GpuBuffer::TempStorage, GpuBuffer::Params],
        }
    }

    pub fn get_buffer_index(&self, buffer: GpuBuffer) -> Option<usize> {
        let templates = self.get_buffer_templates();
        templates.iter().position(|&buf| buf == buffer)
    }

    fn create_bind_group_layout(&self, adapter: &GpuAdapter) -> wgpu::BindGroupLayout  {

        let templates = self.get_buffer_templates();
        let mut layout_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();

        let mut binding_num = 0;
        for template in templates.iter() {

            if !template.included_in_bind_group() {
                continue;
            }

            layout_entries.push( wgpu::BindGroupLayoutEntry {
                binding: binding_num,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: template.is_read_only() },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });

            binding_num += 1;
        }

        let layout = adapter.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &layout_entries.as_slice()
        });

        layout
    }
}


