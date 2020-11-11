//! # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
//! Computes the area under the receiver operating characteristic (ROC) curve that is equal to the probability that a classifier will rank a
//! randomly chosen positive instance higher than a randomly chosen negative one.
//!
//! SmartCore calculates ROC AUC from Wilcoxon or Mann-Whitney U test.
//!
//! Example:
//! ```
//! use smartcore::metrics::auc::AUC;
//!
//! let y_true: Vec<f64> = vec![0., 0., 1., 1.];
//! let y_pred: Vec<f64> = vec![0.1, 0.4, 0.35, 0.8];
//!
//! let score1: f64 = AUC {}.get_score(&y_true, &y_pred);
//! ```
//!
//! ## References:
//! * ["Areas beneath the relative operating characteristics (ROC) and relative operating levels (ROL) curves: Statistical significance and interpretation", Mason S. J., Graham N. E.](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.458.8392)
//! * [Wikipedia article on ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)
//! * ["The ROC-AUC and the Mann-Whitney U-test", Haupt, J.](https://johaupt.github.io/roc-auc/model%20evaluation/Area_under_ROC_curve.html)
#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

use crate::algorithm::sort::quick_sort::QuickArgSort;
use crate::linalg::BaseVector;
use crate::math::num::RealNumber;

/// Area Under the Receiver Operating Characteristic Curve (ROC AUC)
#[derive(Serialize, Deserialize, Debug)]
pub struct AUC {}

impl AUC {
    /// AUC score.
    /// * `y_true` - cround truth (correct) labels.
    /// * `y_pred_probabilities` - probability estimates, as returned by a classifier.
    pub fn get_score<T: RealNumber, V: BaseVector<T>>(&self, y_true: &V, y_pred_prob: &V) -> T {
        let mut pos = T::zero();
        let mut neg = T::zero();

        let n = y_true.len();

        for i in 0..n {
            if y_true.get(i) == T::zero() {
                neg += T::one();
            } else if y_true.get(i) == T::one() {
                pos += T::one();
            } else {
                panic!(
                    "AUC is only for binary classification. Invalid label: {}",
                    y_true.get(i)
                );
            }
        }

        let mut y_pred = y_pred_prob.to_vec();

        let label_idx = y_pred.quick_argsort_mut();

        let mut rank = vec![T::zero(); n];
        let mut i = 0;
        while i < n {
            if i == n - 1 || y_pred[i] != y_pred[i + 1] {
                rank[i] = T::from_usize(i + 1).unwrap();
            } else {
                let mut j = i + 1;
                while j < n && y_pred[j] == y_pred[i] {
                    j += 1;
                }
                let r = T::from_usize(i + 1 + j).unwrap() / T::two();
                for k in i..j {
                    rank[k] = r;
                }
                i = j - 1;
            }
            i += 1;
        }

        let mut auc = T::zero();
        for i in 0..n {
            if y_true.get(label_idx[i]) == T::one() {
                auc += rank[i];
            }
        }

        (auc - (pos * (pos + T::one()) / T::two())) / (pos * neg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc() {
        let y_true: Vec<f64> = vec![0., 0., 1., 1.];
        let y_pred: Vec<f64> = vec![0.1, 0.4, 0.35, 0.8];

        let score1: f64 = AUC {}.get_score(&y_true, &y_pred);
        let score2: f64 = AUC {}.get_score(&y_true, &y_true);

        assert!((score1 - 0.75).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
