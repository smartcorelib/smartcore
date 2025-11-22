//! # Precision score
//!
//! How many predicted items are relevant?
//!
//! \\[precision = \frac{tp}{tp + fp}\\]
//!
//! where tp (true positive) - correct result, fp (false positive) - unexpected result.
//! For binary classification, this is precision for the positive class (assumed to be 1.0).
//! For multiclass, this is macro-averaged precision (average of per-class precisions).
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::precision::Precision;
//! use smartcore::metrics::Metrics;
//! let y_pred: Vec<f64> = vec![0., 1., 1., 0.];
//! let y_true: Vec<f64> = vec![0., 0., 1., 1.];
//!
//! let score: f64 = Precision::new().get_score(&y_true, &y_pred);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::realnum::RealNumber;

use crate::metrics::Metrics;

/// Precision metric.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct Precision<T> {
    _phantom: PhantomData<T>,
}

impl<T: RealNumber> Metrics<T> for Precision<T> {
    /// create a typed object to call Precision functions
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
    /// Calculated precision score
    /// * `y_true` - ground truth (correct) labels.
    /// * `y_pred` - predicted labels, as returned by a classifier.
    fn get_score(&self, y_true: &dyn ArrayView1<T>, y_pred: &dyn ArrayView1<T>) -> f64 {
        if y_true.shape() != y_pred.shape() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.shape(),
                y_pred.shape()
            );
        }

        let n = y_true.shape();

        let mut classes_set: HashSet<u64> = HashSet::new();
        for i in 0..n {
            classes_set.insert(y_true.get(i).to_f64_bits());
        }
        let classes: usize = classes_set.len();

        if classes == 2 {
            // Binary case: precision for positive class (assumed T::one())
            let positive = T::one();
            let mut tp: usize = 0;
            let mut fp_count: usize = 0;
            for i in 0..n {
                let t = *y_true.get(i);
                let p = *y_pred.get(i);
                if p == t {
                    if t == positive {
                        tp += 1;
                    }
                } else {
                    if t != positive {
                        fp_count += 1;
                    }
                }
            }
            if tp + fp_count == 0 {
                0.0
            } else {
                tp as f64 / (tp + fp_count) as f64
            }
        } else {
            // Multiclass case: macro-averaged precision
            let mut predicted: HashMap<u64, usize> = HashMap::new();
            let mut tp_map: HashMap<u64, usize> = HashMap::new();
            for i in 0..n {
                let p_bits = y_pred.get(i).to_f64_bits();
                *predicted.entry(p_bits).or_insert(0) += 1;
                if *y_true.get(i) == *y_pred.get(i) {
                    *tp_map.entry(p_bits).or_insert(0) += 1;
                }
            }
            let mut precision_sum = 0.0;
            for &bits in &classes_set {
                let pred_count = *predicted.get(&bits).unwrap_or(&0);
                let tp = *tp_map.get(&bits).unwrap_or(&0);
                let prec = if pred_count > 0 {
                    tp as f64 / pred_count as f64
                } else {
                    0.0
                };
                precision_sum += prec;
            }
            if classes == 0 {
                0.0
            } else {
                precision_sum / classes as f64
            }
        }
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
    fn precision() {
        let y_true: Vec<f64> = vec![0., 1., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1.];

        let score1: f64 = Precision::new().get_score(&y_true, &y_pred);
        let score2: f64 = Precision::new().get_score(&y_pred, &y_pred);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);

        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];

        let score3: f64 = Precision::new().get_score(&y_true, &y_pred);
        assert!((score3 - 0.5).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn precision_multiclass() {
        let y_true: Vec<f64> = vec![0., 0., 0., 1., 1., 1., 2., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 2., 0., 1., 2., 0., 1., 2.];

        let score1: f64 = Precision::new().get_score(&y_true, &y_pred);
        let score2: f64 = Precision::new().get_score(&y_pred, &y_pred);

        assert!((score1 - 0.333333333).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn precision_multiclass_imbalanced() {
        let y_true: Vec<f64> = vec![0., 0., 1., 2., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 1., 2., 0., 2.];

        let score: f64 = Precision::new().get_score(&y_true, &y_pred);
        let expected = (0.5 + 0.5 + 1.0) / 3.0;
        assert!((score - expected).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn precision_multiclass_unpredicted_class() {
        let y_true: Vec<f64> = vec![0., 0., 1., 2., 2., 2., 3.];
        let y_pred: Vec<f64> = vec![0., 1., 1., 2., 0., 2., 0.];

        let score: f64 = Precision::new().get_score(&y_true, &y_pred);
        // Class 0: pred=3, tp=1 -> 1/3 ≈0.333
        // Class 1: pred=2, tp=1 -> 0.5
        // Class 2: pred=2, tp=2 -> 1.0
        // Class 3: pred=0, tp=0 -> 0.0
        let expected = (1.0 / 3.0 + 0.5 + 1.0 + 0.0) / 4.0;
        assert!((score - expected).abs() < 1e-8);
    }
}
