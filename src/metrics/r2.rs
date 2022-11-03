//! Coefficient of Determination (R2)
//!
//! Coefficient of determination, denoted R2 is the proportion of the variance in the dependent variable that can be explained be explanatory (independent) variable(s).
//!
//! \\[R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y_i})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \\]
//!
//! where \\(\hat{y}\\) are predictions, \\(y\\) are true target values, \\(\bar{y}\\) is the mean of the observed data
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::mean_absolute_error::MeanAbsoluteError;
//! use smartcore::metrics::Metrics;
//! let y_pred: Vec<f64> = vec![3., -0.5, 2., 7.];
//! let y_true: Vec<f64> = vec![2.5, 0.0, 2., 8.];
//!
//! let mse: f64 = MeanAbsoluteError::new().get_score( &y_true, &y_pred);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::basenum::Number;

use crate::metrics::Metrics;

/// Coefficient of Determination (R2)
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct R2<T> {
    _phantom: PhantomData<T>,
}

impl<T: Number> Metrics<T> for R2<T> {
    /// create a typed object to call R2 functions
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
    fn new_with(_parameter: f64) -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
    /// Computes R2 score
    /// * `y_true` - Ground truth (correct) target values.
    /// * `y_pred` - Estimated target values.
    fn get_score(&self, y_true: &dyn ArrayView1<T>, y_pred: &dyn ArrayView1<T>) -> f64 {
        if y_true.shape() != y_pred.shape() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.shape(),
                y_pred.shape()
            );
        }

        let n = y_true.shape();

        let mean: f64 = y_true.mean_by();
        let mut ss_tot = T::zero();
        let mut ss_res = T::zero();

        for i in 0..n {
            let y_i = *y_true.get(i);
            let f_i = *y_pred.get(i);
            ss_tot += (y_i - T::from(mean).unwrap()) * (y_i - T::from(mean).unwrap());
            ss_res += (y_i - f_i) * (y_i - f_i);
        }

        (T::one() - ss_res / ss_tot).to_f64().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn r2() {
        let y_true: Vec<f64> = vec![3., -0.5, 2., 7.];
        let y_pred: Vec<f64> = vec![2.5, 0.0, 2., 8.];

        let score1: f64 = R2::new().get_score(&y_true, &y_pred);
        let score2: f64 = R2::new().get_score(&y_true, &y_true);

        assert!((score1 - 0.948608137).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
