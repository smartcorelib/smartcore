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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::Array1;
use crate::numbers::basenum::Number;

/// Area Under the Receiver Operating Characteristic Curve (ROC AUC)
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct AUC {}

impl AUC {
    /// AUC score.
    /// * `y_true` - cround truth (correct) labels.
    /// * `y_pred_probabilities` - probability estimates, as returned by a classifier.
    pub fn get_score<T: Number + PartialOrd, V: Array1<T>>(
        &self,
        y_true: &V,
        y_pred_prob: &V,
    ) -> f64 {
        let mut pos = T::zero();
        let mut neg = T::zero();

        let n = y_true.shape();

        for i in 0..n {
            if y_true.get(i) == &T::zero() {
                neg += T::one();
            } else if y_true.get(i) == &T::one() {
                pos += T::one();
            } else {
                panic!(
                    "AUC is only for binary classification. Invalid label: {}",
                    y_true.get(i)
                );
            }
        }

        let mut y_pred = y_pred_prob.clone();

        let label_idx = y_pred.argsort_mut();

        let two = T::from(2).unwrap();

        let mut rank = vec![0f64; n];
        let mut i = 0;
        while i < n {
            if i == n - 1 || y_pred.get(i) != y_pred.get(i + 1) {
                rank[i] = (i + 1) as f64;
            } else {
                let mut j = i + 1;
                while j < n && y_pred.get(j) == y_pred.get(i) {
                    j += 1;
                }
                let r = (i + 1 + j) as f64 / 2f64;
                for rank_k in rank.iter_mut().take(j).skip(i) {
                    *rank_k = r;
                }
                i = j - 1;
            }
            i += 1;
        }

        let mut auc = 0f64;
        for i in 0..n {
            if y_true.get(label_idx[i]) == &T::one() {
                auc += rank[i];
            }
        }
        let pos = pos.to_f64().unwrap();
        let neg = neg.to_f64().unwrap();

        (auc - (pos * (pos + 1f64) / 2.0)) / (pos * neg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
