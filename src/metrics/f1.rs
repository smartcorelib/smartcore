//! # F-measure
//!
//! Harmonic mean of the precision and recall.
//!
//! \\[f1 = (1 + \beta^2)\frac{precision \times recall}{\beta^2 \times precision + recall}\\]
//!
//! where \\(\beta \\) is a positive real factor, where \\(\beta \\) is chosen such that recall is considered \\(\beta \\) times as important as precision.
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::f1::F1;
//! use smartcore::metrics::Metrics;
//! let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
//! let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];
//!
//! let beta = 1.0; // beta default is equal 1.0 anyway
//! let score: f64 = F1::new_with(beta).get_score(&y_pred, &y_true);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::ArrayView1;
use crate::metrics::precision::Precision;
use crate::metrics::recall::Recall;
use crate::numbers::realnum::RealNumber;
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;

use crate::metrics::Metrics;

/// F-measure
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct F1<T> {
    /// a positive real factor
    pub beta: T,
}

impl<T: Number + RealNumber + FloatNumber> Metrics<T> for F1<T> {
    fn new() -> Self {
        let beta: T = T::from(1f64).unwrap();
        Self { beta }
    }
    /// create a typed object to call Recall functions
    fn new_with(beta: T) -> Self {
        Self {
            beta
        }
    }
    /// Computes F1 score
    /// * `y_true` - cround truth (correct) labels.
    /// * `y_pred` - predicted labels, as returned by a classifier.
    fn get_score(&self,
        y_true: &dyn ArrayView1<T>, 
        y_pred: &dyn ArrayView1<T>
    ) -> f64 {
        if y_true.shape() != y_pred.shape() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.shape(),
                y_pred.shape()
            );
        }
        let beta2 = self.beta * self.beta;

        let p = Precision::new().get_score(y_true, y_pred);
        let r = Recall::new().get_score(y_true, y_pred);

        (T::one() + beta2).to_f64().unwrap() * (p * r) / ((beta2.to_f64().unwrap() * p) + r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn f1() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let score1: f64 = F1 { beta: 1.0 }.get_score(&y_pred, &y_true);
        let score2: f64 = F1 { beta: 1.0 }.get_score(&y_true, &y_true);

        println!("{:?}", score1);
        println!("{:?}", score2);

        assert!((score1 - 0.57142857).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
