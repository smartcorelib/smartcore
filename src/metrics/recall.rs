//! # Recall score
//!
//! How many relevant items are selected?
//!
//! \\[recall = \frac{tp}{tp + fn}\\]
//!
//! where tp (true positive) - correct result, fn (false negative) - missing result
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::recall::Recall;
//! let y_pred: Vec<f64> = vec![0., 1., 1., 0.];
//! let y_true: Vec<f64> = vec![0., 0., 1., 1.];
//!
//! let score: f64 = Recall {}.get_score(&y_pred, &y_true);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

use std::collections::HashSet;
use std::convert::TryInto;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::Array1;
use crate::numbers::realnum::RealNumber;

/// Recall metric.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct Recall {}

impl Recall {
    /// Calculated recall score
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

        let mut classes = HashSet::new();
        for i in 0..y_true.shape() {
            classes.insert(y_true.get(i).to_f64_bits());
        }
        let classes: i64 = classes.len().try_into().unwrap();

        let mut tp = 0;
        let mut fne = 0;
        for i in 0..y_true.shape() {
            if y_pred.get(i) == y_true.get(i) {
                if classes == 2 {
                    if *y_true.get(i) == T::one() {
                        tp += 1;
                    }
                } else {
                    tp += 1;
                }
            } else if classes == 2 {
                if *y_true.get(i) != T::one() {
                    fne += 1;
                }
            } else {
                fne += 1;
            }
        }
        T::from_i64(tp).unwrap() / (T::from_i64(tp).unwrap() + T::from_i64(fne).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn recall() {
        let y_true: Vec<f64> = vec![0., 1., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1.];

        let score1: f64 = Recall {}.get_score(&y_pred, &y_true);
        let score2: f64 = Recall {}.get_score(&y_pred, &y_pred);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);

        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let score3: f64 = Recall {}.get_score(&y_pred, &y_true);
        assert!((score3 - 0.6666666666666666).abs() < 1e-8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn recall_multiclass() {
        let y_true: Vec<f64> = vec![0., 0., 0., 1., 1., 1., 2., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 2., 0., 1., 2., 0., 1., 2.];

        let score1: f64 = Recall {}.get_score(&y_pred, &y_true);
        let score2: f64 = Recall {}.get_score(&y_pred, &y_pred);

        assert!((score1 - 0.333333333).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
