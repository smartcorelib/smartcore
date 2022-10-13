//! # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
//! Computes the area under the receiver operating characteristic (ROC) curve that is equal to the probability that a classifier will rank a
//! randomly chosen positive instance higher than a randomly chosen negative one.
//!
//! SmartCore calculates ROC AUC from Wilcoxon or Mann-Whitney U test.
//!
//! Example:
//! ```
//! use smartcore::metrics::auc::AUC;
//! use smartcore::metrics::Metrics;
//!
//! let y_true: Vec<f64> = vec![0., 0., 1., 1.];
//! let y_pred: Vec<f64> = vec![0.1, 0.4, 0.35, 0.8];
//!
//! let score1: f64 = AUC::new().get_score(&y_true, &y_pred);
//! ```
//!
//! ## References:
//! * ["Areas beneath the relative operating characteristics (ROC) and relative operating levels (ROL) curves: Statistical significance and interpretation", Mason S. J., Graham N. E.](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.458.8392)
//! * [Wikipedia article on ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)
//! * ["The ROC-AUC and the Mann-Whitney U-test", Haupt, J.](https://johaupt.github.io/roc-auc/model%20evaluation/Area_under_ROC_curve.html)
#![allow(non_snake_case)]

use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::{Array1, ArrayView1, MutArrayView1};
use crate::numbers::basenum::Number;

use crate::metrics::Metrics;

/// Area Under the Receiver Operating Characteristic Curve (ROC AUC)
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct AUC<T> {
    _phantom: PhantomData<T>
}

impl<T: Number + Ord> Metrics<T> for AUC<T> {
    /// create a typed object to call AUC functions
    fn new() -> Self {
        Self {
            _phantom: PhantomData
        }
    }
    fn new_with(_parameter: T) -> Self {
        Self {
            _phantom: PhantomData
        }
    }
    /// AUC score.
    /// * `y_true` - ground truth (correct) labels.
    /// * `y_pred_prob` - probability estimates, as returned by a classifier.
    fn get_score(
        &self,
        y_true: &dyn ArrayView1<T>,
        y_pred_prob: &dyn ArrayView1<T>,
    ) -> T {
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

        let y_pred = y_pred_prob.clone();

        let label_idx = y_pred.argsort();

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

        T::from(auc - (pos * (pos + 1f64) / 2.0)).unwrap() / T::from(pos * neg).unwrap()
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

        let score1: f64 = AUC::new().get_score(&y_true, &y_pred);
        let score2: f64 = AUC::new().get_score(&y_true, &y_true);

        assert!((score1 - 0.75).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
