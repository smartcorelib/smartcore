
use wgpu::util::DeviceExt;
use crate::error::GpuError;
use crate::numbers::basenum::Number;
use crate::linalg::basic::arrays::ArrayView1;
use super::{buffer, STATION, GpuAdapter, GpuMatrix, GpuModule, GpuParams, GpuResourceLayout, GpuWorkgroup, GpuBuffer};

pub struct GpuWorker<M: GpuModule, T: Number + Ord> {
    module: M,
    adapter: GpuAdapter,
    params: GpuParams,
    resources: GpuResourceLayout,
    workgroup: GpuWorkgroup,
    matrix: GpuMatrix,
    targets: Vec<T>,
    buffers: Vec<(GpuBuffer, wgpu::Buffer)>,
}

impl<M: GpuModule, T: Number + Ord> GpuWorker<M, T> {
    pub fn new(module: M, matrix: GpuMatrix, targets: Vec<T>) -> Result<Self, GpuError> {

        // Get adapter
        let adapter = STATION.get_adapter()?;

        // Get params
        let num_classes = targets.unique().len();
        let params = module.get_params(&matrix, num_classes)?;

        // Get resource layout
        let resources = STATION.get_layout(params.layout, &adapter)?;

        // Get workgroup
        let workgroup = STATION.get_workgroup(&module, &matrix, &params, &adapter, &resources)?;

        Ok( Self {
            module,
            adapter,
            params,
            resources,
            workgroup,
            matrix,
            targets,
            buffers: vec![]
        })
    }

    pub fn run(&mut self) {

        // Create buffers
        self.create_buffers();

        // Create bind group
        let bind_group = self.create_bind_group();

        // Get workgroup counts
        let workgroup_count = if self.matrix.cols > 1024 {
            self.matrix.rows.div_ceil(self.matrix.get_workgroup_size()) * self.matrix.cols.div_ceil(1024)
        } else {
            self.matrix.rows.div_ceil(self.matrix.get_workgroup_size())
        };

        // Get command encoder
        let mut encoder = self.adapter.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Run commands
        for step in 0..self.params.total_steps {
            self.set_step(step as u32);

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });
            pass.set_pipeline(&self.workgroup.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            drop(pass);
        }

        // Submit work to GPU
        self.adapter.queue.submit(Some(encoder.finish()));
    }

    fn create_buffers(&mut self) {

        // Get templates
        let templates = self.params.layout.get_buffer_templates();
        let mut input_size: u64 = 0;

        // Create buffers
        for template in templates.iter() {
            let buffer = match template {
                GpuBuffer::Samples => buffer::create_samples(&self.adapter, &self.matrix),
                GpuBuffer::Targets => buffer::create_targets(&self.adapter, &self.targets),
                GpuBuffer::Weights => buffer::create_weights(&self.adapter, self.matrix.cols),
                GpuBuffer::TempStorage => buffer::create_temp_storage(&self.adapter, input_size),
                GpuBuffer::Params => buffer::create_params(&self.adapter, &self.module.get_params_buffer_data(&self.params)),
                GpuBuffer::Download => buffer::create_download(&self.adapter, input_size),
                _ => unreachable!()
            };

            if *template == GpuBuffer::Samples {
                input_size = buffer.size();
            }
            self.buffers.push((*template, buffer));
        }

    }

    fn create_bind_group(&self) -> wgpu::BindGroup {

        let mut binding_num = 0;
        let mut bind_group_entries: Vec<wgpu::BindGroupEntry> = Vec::new();

        for index in 0..self.buffers.len() {

            if !self.buffers[index].0.included_in_bind_group() {
                continue;
            }

            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: binding_num,
                resource: self.buffers[index].1.as_entire_binding(),
            });
            binding_num += 1;
        }

        // Create bind group
        let bind_group = self.adapter.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.resources.bind_group_layout,
            entries: &bind_group_entries.as_slice()
        });

        bind_group
    }

    pub fn set_step(&mut self, step: u32) -> Result<(), GpuError> {
        self.params.step = step;
        self.update_params_buffer()
    }

    pub fn set_learning_rate(&mut self, learning_rate: f32) -> Result<(), GpuError> {
        self.params.learning_rate = learning_rate;
        self.update_params_buffer()
    }

    fn update_params_buffer(&self) -> Result<(), GpuError> {
        match self.buffers.iter().position(|buf| buf.0 == GpuBuffer::Params) {
            Some(index) => {
                self.adapter.queue.write_buffer(&self.buffers[index].1, 0, bytemuck::cast_slice(&self.module.get_params_buffer_data(&self.params).as_slice()));
                Ok(())
            },
            None => Err(GpuError::ParamsBufferNotFound)
        }
    }
}


