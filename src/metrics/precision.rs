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

use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::ArrayView1;
use crate::metrics::confusion::ConfusionCounts;
use crate::numbers::realnum::RealNumber;

use crate::metrics::Metrics;

/// Precision metric.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct Precision<T> {
    _phantom: PhantomData<T>,
}

impl<T: RealNumber> Precision<T> {
    /// Per-class precision scores derived from shared confusion counts.
    ///
    /// Returns a map from label bit pattern to that class's precision
    /// (`tp / predicted`, or `0.0` when the class is never predicted).
    pub(crate) fn per_class_scores_from_counts(
        &self,
        counts: &ConfusionCounts,
    ) -> HashMap<u64, f64> {
        let mut scores: HashMap<u64, f64> = HashMap::new();
        for &bits in counts.classes_set() {
            let pred_count = counts.predicted(bits);
            let tp = counts.tp(bits);
            let prec = if pred_count > 0 {
                tp as f64 / pred_count as f64
            } else {
                0.0
            };
            scores.insert(bits, prec);
        }
        scores
    }
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
        // Empty input has no classes; return 0.0 (the multiclass path below
        // relies on classes >= 1 to divide by `classes`).
        if n == 0 {
            return 0.0;
        }

        let counts = ConfusionCounts::from(y_true, y_pred);
        let classes = counts.classes_set().len();
        let scores = self.per_class_scores_from_counts(&counts);

        if classes == 2 {
            // Binary case: precision for the positive class, assumed to be
            // T::one() (i.e. 1.0 when labels are 0.0/1.0). If the positive
            // label is not present in y_true the score is 0.0.
            let positive_bits = T::one().to_f64_bits();
            *scores.get(&positive_bits).unwrap_or(&0.0)
        } else {
            // Multiclass case: macro-averaged precision. classes >= 1 is
            // guaranteed here because of the `n == 0` guard above.
            scores.values().sum::<f64>() / classes as f64
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
