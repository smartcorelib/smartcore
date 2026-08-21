//! # Recall score
//!
//! How many relevant items are selected?
//!
//! \\[recall = \frac{tp}{tp + fn}\\]
//!
//! where tp (true positive) - correct result, fn (false negative) - missing result.
//! For binary classification, this is recall for the positive class (assumed to be 1.0).
//! For multiclass, this is macro-averaged recall (average of per-class recalls).
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::recall::Recall;
//! use smartcore::metrics::Metrics;
//! let y_pred: Vec<f64> = vec![0., 1., 1., 0.];
//! let y_true: Vec<f64> = vec![0., 0., 1., 1.];
//!
//! let score: f64 = Recall::new().get_score( &y_true, &y_pred);
//! ```
//!
//! Integer labels work too, so these metrics pair with classifiers like
//! `RandomForestClassifier`, whose `fit` takes ordered integer labels:
//!
//! ```
//! use smartcore::metrics::recall::Recall;
//! use smartcore::metrics::Metrics;
//! let y_pred: Vec<u16> = vec![0, 1, 1, 0];
//! let y_true: Vec<u16> = vec![0, 0, 1, 1];
//!
//! let score: f64 = Recall::new().get_score(&y_true, &y_pred);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::ArrayView1;
use crate::metrics::confusion::{ConfusionCounts, label_bits};
use crate::numbers::basenum::Number;

use crate::metrics::Metrics;

/// Recall metric.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct Recall<T> {
    _phantom: PhantomData<T>,
}

impl<T: Number> Recall<T> {
    /// Per-class recall scores derived from shared confusion counts.
    ///
    /// Returns a map from label bit pattern to that class's recall
    /// (`tp / support`, or `0.0` when the class has no support).
    ///
    /// Iterates only over `counts.classes_set()` (labels seen in `y_true`).
    /// A label that appears in `y_pred` but never in `y_true` has no support
    /// and no true positives, so it is silently ignored — it does not
    /// inflate or deflate any class's recall. This matches sklearn's
    /// behaviour, where the label set is derived from `y_true`.
    pub(crate) fn per_class_scores_from_counts(
        &self,
        counts: &ConfusionCounts,
    ) -> HashMap<u64, f64> {
        let mut scores: HashMap<u64, f64> = HashMap::new();
        for &bits in counts.classes_set() {
            let support_count = counts.support(bits);
            let tp = counts.tp(bits);
            let rec = if support_count > 0 {
                tp as f64 / support_count as f64
            } else {
                0.0
            };
            scores.insert(bits, rec);
        }
        scores
    }
}

impl<T: Number> Metrics<T> for Recall<T> {
    /// create a typed object to call Recall functions
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
    /// Calculated recall score
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

        let counts = ConfusionCounts::new(y_true, y_pred);
        let classes = counts.classes_set().len();
        let scores = self.per_class_scores_from_counts(&counts);

        if classes == 2 {
            // Binary case: recall for the positive class, assumed to be
            // T::one() (i.e. 1.0 when labels are 0.0/1.0). If the positive
            // label is not present in y_true the score is 0.0.
            let positive_bits = label_bits(T::one());
            *scores.get(&positive_bits).unwrap_or(&0.0)
        } else {
            // Multiclass case: macro-averaged recall. classes >= 1 is
            // guaranteed here because of the `n == 0` guard above. The sum
            // over `HashMap::values()` is order-independent (floating-point
            // addition of non-negative finite values is commutative and
            // associative for the magnitudes involved here).
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
    fn recall() {
        let y_true: Vec<f64> = vec![0., 1., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1.];

        let score1: f64 = Recall::new().get_score(&y_true, &y_pred);
        let score2: f64 = Recall::new().get_score(&y_pred, &y_pred);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);

        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];

        let score3: f64 = Recall::new().get_score(&y_true, &y_pred);
        assert!((score3 - (2.0 / 3.0)).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn recall_multiclass() {
        let y_true: Vec<f64> = vec![0., 0., 0., 1., 1., 1., 2., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 2., 0., 1., 2., 0., 1., 2.];

        let score1: f64 = Recall::new().get_score(&y_true, &y_pred);
        let score2: f64 = Recall::new().get_score(&y_pred, &y_pred);

        assert!((score1 - 0.333333333).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn recall_multiclass_imbalanced() {
        let y_true: Vec<f64> = vec![0., 0., 1., 2., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 1., 2., 0., 2.];

        let score: f64 = Recall::new().get_score(&y_true, &y_pred);
        let expected = (0.5 + 1.0 + (2.0 / 3.0)) / 3.0;
        assert!((score - expected).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn recall_integer_labels() {
        // Binary case with ordered integer labels (#322).
        let y_true: Vec<i32> = vec![0, 1, 1, 0];
        let y_pred: Vec<i32> = vec![0, 0, 1, 1];
        let score: f64 = Recall::new().get_score(&y_true, &y_pred);
        assert!((score - 0.5).abs() < 1e-8);

        // Multiclass macro-average with u16 labels.
        let y_true: Vec<u16> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let y_pred: Vec<u16> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let score: f64 = Recall::new().get_score(&y_true, &y_pred);
        assert!((score - 0.333333333).abs() < 1e-8);

        // Perfect predictions score 1.0.
        let score: f64 = Recall::new().get_score(&y_true, &y_true);
        assert!((score - 1.0).abs() < 1e-8);
    }
}
