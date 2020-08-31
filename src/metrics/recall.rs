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
//! <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::RealNumber;

/// Recall metric.
#[derive(Serialize, Deserialize, Debug)]
pub struct Recall {}

impl Recall {
    /// Calculated recall score
    /// * `y_true` - cround truth (correct) labels.
    /// * `y_pred` - predicted labels, as returned by a classifier.
    pub fn get_score<T: RealNumber, V: BaseVector<T>>(&self, y_true: &V, y_pred: &V) -> T {
        if y_true.len() != y_pred.len() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.len(),
                y_pred.len()
            );
        }

        let mut tp = 0;
        let mut p = 0;
        let n = y_true.len();
        for i in 0..n {
            if y_true.get(i) != T::zero() && y_true.get(i) != T::one() {
                panic!(
                    "Recall can only be applied to binary classification: {}",
                    y_true.get(i)
                );
            }

            if y_pred.get(i) != T::zero() && y_pred.get(i) != T::one() {
                panic!(
                    "Recall can only be applied to binary classification: {}",
                    y_pred.get(i)
                );
            }

            if y_true.get(i) == T::one() {
                p += 1;

                if y_pred.get(i) == T::one() {
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

    #[test]
    fn recall() {
        let y_true: Vec<f64> = vec![0., 1., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1.];

        let score1: f64 = Recall {}.get_score(&y_pred, &y_true);
        let score2: f64 = Recall {}.get_score(&y_pred, &y_pred);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
