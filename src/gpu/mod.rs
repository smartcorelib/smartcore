
use crate::error::GpuError;

use lazy_static::lazy_static;

lazy_static! {
    pub static ref STATION: GpuStation = GpuStation::new();
}

pub use self::adapter::GpuAdapter;
pub use self::buffer::GpuBuffer;
pub use self::layout::{GpuLayout, GpuResourceLayout};
pub use self::matrix::GpuMatrix;
pub use self::params::GpuParams;
pub use self::station::{GpuStation, GpuWorkgroup};
pub use self::worker::GpuWorker;

mod adapter;
pub mod buffer;
mod layout;
mod matrix;
pub mod models;
mod params;
mod station;
mod worker;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuAlgorithm {
    LogisticRegressionGradientDescentBinaryClassification,
    LogisticRegressionGradientDescentMultiClassification,
}
pub trait GpuModule {
    fn get_params(&self, matrix: &GpuMatrix, num_classes: usize) -> Result<GpuParams, GpuError>;
    fn get_wgsl_code(&self, matrix: &GpuMatrix, params: &GpuParams) -> String;
    fn get_params_buffer_data(&self, params: &GpuParams) -> Vec<u32>;
}


