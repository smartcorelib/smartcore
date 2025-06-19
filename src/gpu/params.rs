
use super::{GpuLayout, GpuAlgorithm, GpuMatrix};

pub struct GpuParams {
    pub algorithm: GpuAlgorithm,
    pub layout: GpuLayout,
    pub step: u32,
    pub total_steps: usize,
    pub entry_point: String,
    pub num_features: u32,
    pub num_samples: u32,
    pub num_classes: u32,
    pub learning_rate: f32
}

impl GpuParams {
    pub fn new(algorithm: GpuAlgorithm, layout: GpuLayout, matrix: &GpuMatrix, num_classes: u32, total_steps: usize, entry_point: &str, learning_rate: f32) -> Self {
        Self {
            algorithm, layout, total_steps, num_classes,
            step: 0,
            num_features: matrix.cols as u32,
            num_samples: matrix.rows as u32,
            entry_point: entry_point.to_string(),
            learning_rate
        }
    }

}




