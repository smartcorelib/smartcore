//! # Precision score
//!
//! How many predicted items are relevant?
//!
//! \\[precision = \frac{tp}{tp + fp}\\]
//!
//! where tp (true positive) - correct result, fp (false positive) - unexpected result
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::precision::Precision;
//! let y_pred: Vec<f64> = vec![0., 1., 1., 0.];
//! let y_true: Vec<f64> = vec![0., 0., 1., 1.];
//!
//! let score: f64 = Precision {}.get_score(&y_pred, &y_true);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::Array1;
use crate::numbers::realnum::RealNumber;

/// Precision metric.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct Precision {}

impl Precision {
    /// Calculated precision score
    /// * `y_true` - cround truth (correct) labels.
    /// * `y_pred` - predicted labels, as returned by a classifier.
    pub fn get_score<T: RealNumber, V: Array1<T>>(&self, y_true: &V, y_pred: &V) -> T {
        if y_true.shape() != y_pred.shape() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.shape(),
                y_pred.shape()
            );
        }

        let mut tp = 0;
        let mut p = 0;
        let n = y_true.shape();
        for i in 0..n {
            if *y_true.get(i) != T::zero() && *y_true.get(i) != T::one() {
                panic!(
                    "Precision can only be applied to binary classification: {}",
                    y_true.get(i)
                );
            }

            if *y_pred.get(i) != T::zero() && *y_pred.get(i) != T::one() {
                panic!(
                    "Precision can only be applied to binary classification: {}",
                    y_pred.get(i)
                );
            }

            if *y_pred.get(i) == T::one() {
                p += 1;

                if *y_true.get(i) == T::one() {
                    tp += 1;
                }
            }
        }

        T::from_i64(tp).unwrap() / T::from_i64(p).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn precision() {
        let y_true: Vec<f64> = vec![0., 1., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1.];

        let score1: f64 = Precision {}.get_score(&y_pred, &y_true);
        let score2: f64 = Precision {}.get_score(&y_pred, &y_pred);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);

        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let score3: f64 = Precision {}.get_score(&y_pred, &y_true);
        assert!((score3 - 0.5).abs() < 1e-8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn precision_multiclass() {
        let y_true: Vec<f64> = vec![0., 0., 0., 1., 1., 1., 2., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 2., 0., 1., 2., 0., 1., 2.];

        let score1: f64 = Precision {}.get_score(&y_pred, &y_true);
        let score2: f64 = Precision {}.get_score(&y_pred, &y_pred);

        assert!((score1 - 0.333333333).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
