//! # Common Interfaces and methods
//!
//! This module consolidates interfaces and uniform basic API that is used elsewhere in the code.

use crate::error::Failed;

/// Implements method predict that offers a way to estimate target value from new data
pub trait Predictor<X, Y> {
    fn predict(&self, x: &X) -> Result<Y, Failed>;
}
