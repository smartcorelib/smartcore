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
//! let y_pred: Vec<f64> = vec![3., -0.5, 2., 7.];
//! let y_true: Vec<f64> = vec![2.5, 0.0, 2., 8.];
//!
//! let mse: f64 = MeanAbsoluteError {}.get_score(&y_pred, &y_true);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::RealNumber;

#[derive(Serialize, Deserialize, Debug)]
/// Mean Absolute Error
pub struct MeanAbsoluteError {}

impl MeanAbsoluteError {
    /// Computes mean absolute error
    /// * `y_true` - Ground truth (correct) target values.
    /// * `y_pred` - Estimated target values.
    pub fn get_score<T: RealNumber, V: BaseVector<T>>(&self, y_true: &V, y_pred: &V) -> T {
        if y_true.len() != y_pred.len() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.len(),
                y_pred.len()
            );
        }

        let n = y_true.len();
        let mut ras = T::zero();
        for i in 0..n {
            ras += (y_true.get(i) - y_pred.get(i)).abs();
        }

        ras / T::from_usize(n).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_absolute_error() {
        let y_true: Vec<f64> = vec![3., -0.5, 2., 7.];
        let y_pred: Vec<f64> = vec![2.5, 0.0, 2., 8.];

        let score1: f64 = MeanAbsoluteError {}.get_score(&y_pred, &y_true);
        let score2: f64 = MeanAbsoluteError {}.get_score(&y_true, &y_true);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 0.0).abs() < 1e-8);
    }
}
