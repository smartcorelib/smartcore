//! # Mean Squared Error
//!
//! MSE measures the average magnitude of the errors in a set of predictions, without considering their direction.
//!
//! \\[mse(y, \hat{y}) = \frac{1}{n_{samples}} \sum_{i=1}^{n_{samples}} (y_i - \hat{y_i})^2 \\]
//!
//! where \\(\hat{y}\\) are predictions and \\(y\\) are true target values.
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::mean_squared_error::MeanSquareError;
//! let y_pred: Vec<f64> = vec![3., -0.5, 2., 7.];
//! let y_true: Vec<f64> = vec![2.5, 0.0, 2., 8.];
//!
//! let mse: f64 = MeanSquareError {}.get_score(&y_pred, &y_true);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::base::Array1;
use crate::num::Number;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// Mean Squared Error
pub struct MeanSquareError {}

impl MeanSquareError {
    /// Computes mean squared error
    /// * `y_true` - Ground truth (correct) target values.
    /// * `y_pred` - Estimated target values.
    pub fn get_score<T: Number, V: Array1<T>>(&self, y_true: &V, y_pred: &V) -> f64 {
        if y_true.shape() != y_pred.shape() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.shape(),
                y_pred.shape()
            );
        }

        let n = y_true.shape();
        let mut rss = T::zero();
        for i in 0..n {
            let res = *y_true.get(i) - *y_pred.get(i);
            rss += res * res;
        }

        rss.to_f64().unwrap() / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean_squared_error() {
        let y_true: Vec<f64> = vec![3., -0.5, 2., 7.];
        let y_pred: Vec<f64> = vec![2.5, 0.0, 2., 8.];

        let score1: f64 = MeanSquareError {}.get_score(&y_pred, &y_true);
        let score2: f64 = MeanSquareError {}.get_score(&y_true, &y_true);

        assert!((score1 - 0.375).abs() < 1e-8);
        assert!((score2 - 0.0).abs() < 1e-8);
    }
}
