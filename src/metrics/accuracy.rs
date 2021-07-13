//! # Accuracy score
//!
//! Calculates accuracy of predictions \\(\hat{y}\\) when compared to true labels \\(y\\)
//!
//! \\[ accuracy(y, \hat{y}) = \frac{1}{n_{samples}} \sum_{i=1}^{n_{samples}} 1(y_i = \hat{y_i}) \\]
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::accuracy::Accuracy;
//! let y_pred: Vec<f64> = vec![0., 2., 1., 3.];
//! let y_true: Vec<f64> = vec![0., 1., 2., 3.];
//!
//! let score: f64 = Accuracy {}.get_score(&y_pred, &y_true);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::base::Array1;
use crate::num::Number;

/// Accuracy metric.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct Accuracy {}

impl Accuracy {
    /// Function that calculated accuracy score.
    /// * `y_true` - cround truth (correct) labels
    /// * `y_pred` - predicted labels, as returned by a classifier.
    pub fn get_score<T: Number, V: Array1<T>>(&self, y_true: &V, y_pred: &V) -> f64 {
        if y_true.shape() != y_pred.shape() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.shape(),
                y_pred.shape()
            );
        }

        let n = y_true.shape();

        let mut positive = 0;
        for i in 0..n {
            if y_true.get(i) == y_pred.get(i) {
                positive += 1;
            }
        }

        positive as f64 / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn accuracy() {
        let y_pred: Vec<f64> = vec![0., 2., 1., 3.];
        let y_true: Vec<f64> = vec![0., 1., 2., 3.];

        let score1: f64 = Accuracy {}.get_score(&y_pred, &y_true);
        let score2: f64 = Accuracy {}.get_score(&y_true, &y_true);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
