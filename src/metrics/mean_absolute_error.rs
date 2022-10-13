//! # Mean Absolute Error
//!
//! MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
//!
//! \\[mse(y, \hat{y}) = \frac{1}{n_{samples}} \sum_{i=1}^{n_{samples}} \lvert y_i - \hat{y_i} \rvert \\]
//!
//! where \\(\hat{y}\\) are predictions and \\(y\\) are true target values.
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::mean_absolute_error::MeanAbsoluteError;
//! use smartcore::metrics::Metrics;
//! let y_pred: Vec<f64> = vec![3., -0.5, 2., 7.];
//! let y_true: Vec<f64> = vec![2.5, 0.0, 2., 8.];
//!
//! let mse: f64 = MeanAbsoluteError::new().get_score(&y_pred, &y_true);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;

use crate::metrics::Metrics;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// Mean Absolute Error
pub struct MeanAbsoluteError<T> {
    _phantom: PhantomData<T>  
}

impl<T: Number + FloatNumber> Metrics<T> for MeanAbsoluteError<T> {
    /// create a typed object to call MeanAbsoluteError functions
    fn new() -> Self {
        Self {
            _phantom: PhantomData
        }
    }
    fn new_with(_parameter: T) -> Self {
        Self {
            _phantom: PhantomData
        }
    }
    /// Computes mean absolute error
    /// * `y_true` - Ground truth (correct) target values.
    /// * `y_pred` - Estimated target values.
    fn get_score(&self,
        y_true: &dyn ArrayView1<T>, 
        y_pred: &dyn ArrayView1<T>
    ) -> T {
        if y_true.shape() != y_pred.shape() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.shape(),
                y_pred.shape()
            );
        }

        let n = y_true.shape();
        let mut ras: T = T::zero();
        for i in 0..n {
            let res: T = *y_true.get(i) - *y_pred.get(i);
            ras += res.abs();
        }

        ras / T::from_usize(n).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_absolute_error() {
        let y_true: Vec<f64> = vec![3., -0.5, 2., 7.];
        let y_pred: Vec<f64> = vec![2.5, 0.0, 2., 8.];

        let score1: f64 = MeanAbsoluteError::new().get_score(&y_pred, &y_true);
        let score2: f64 = MeanAbsoluteError::new().get_score(&y_true, &y_true);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 0.0).abs() < 1e-8);
    }
}
