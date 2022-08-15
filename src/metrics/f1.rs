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
//! let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
//! let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];
//!
//! let score: f64 = F1 {beta: 1.0}.get_score(&y_pred, &y_true);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::RealNumber;
use crate::metrics::precision::Precision;
use crate::metrics::recall::Recall;

/// F-measure
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct F1<T: RealNumber> {
    /// a positive real factor
    pub beta: T,
}

impl<T: RealNumber> F1<T> {
    /// Computes F1 score
    /// * `y_true` - cround truth (correct) labels.
    /// * `y_pred` - predicted labels, as returned by a classifier.
    pub fn get_score<V: BaseVector<T>>(&self, y_true: &V, y_pred: &V) -> T {
        if y_true.len() != y_pred.len() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.len(),
                y_pred.len()
            );
        }
        let beta2 = self.beta * self.beta;

        let p = Precision {}.get_score(y_true, y_pred);
        let r = Recall {}.get_score(y_true, y_pred);

        (T::one() + beta2) * (p * r) / (beta2 * p + r)
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

        assert!((score1 - 0.57142857).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
