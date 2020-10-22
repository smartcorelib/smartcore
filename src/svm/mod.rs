//! # Support Vector Machines
//!

pub mod svc;
pub mod svr;

use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::RealNumber;

/// Kernel
pub trait Kernel<T: RealNumber, V: BaseVector<T>> {
    /// Apply kernel function to x_i and x_j
    fn apply(&self, x_i: &V, x_j: &V) -> T;
}

/// Linear Kernel
#[derive(Serialize, Deserialize, Debug)]
pub struct LinearKernel {}

impl<T: RealNumber, V: BaseVector<T>> Kernel<T, V> for LinearKernel {
    fn apply(&self, x_i: &V, x_j: &V) -> T {
        x_i.dot(x_j)
    }
}
