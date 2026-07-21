//! # F-measure
//!
//! Harmonic mean of the precision and recall.
//!
//! \\[f1 = (1 + \beta^2)\frac{precision \times recall}{\beta^2 \times precision + recall}\\]
//!
//! where \\(\beta \\) is a positive real factor, where \\(\beta \\) is chosen such that recall is considered \\(\beta \\) times as important as precision.
//!
//! For binary classification, this is the F-measure of the positive class (assumed to be 1.0).
//! For multiclass, this is the macro-averaged F-measure (mean of the per-class F-measures).
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::f1::F1;
//! use smartcore::metrics::Metrics;
//! let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
//! let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];
//!
//! let beta = 1.0; // beta default is equal 1.0 anyway
//! let score: f64 = F1::new_with(beta).get_score( &y_true, &y_pred);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::ArrayView1;
use crate::metrics::precision::Precision;
use crate::metrics::recall::Recall;
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

use crate::metrics::Metrics;

/// F-measure
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct F1<T> {
    /// a positive real factor
    pub beta: f64,
    _phantom: PhantomData<T>,
}

impl<T: Number + RealNumber + FloatNumber> Metrics<T> for F1<T> {
    fn new() -> Self {
        let beta: f64 = 1f64;
        Self {
            beta,
            _phantom: PhantomData,
        }
    }
    /// create a typed object to call Recall functions
    fn new_with(beta: f64) -> Self {
        Self {
            beta,
            _phantom: PhantomData,
        }
    }
    /// Computes F1 score
    /// * `y_true` - cround truth (correct) labels.
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
        // Empty input has no classes; return 0.0 (matches sklearn's empty-score
        // behaviour and lets the multiclass path below assume classes >= 1).
        if n == 0 {
            return 0.0;
        }
        let beta2 = self.beta * self.beta;

        // Single pass over y_true/y_pred: collect the class set (needed to pick
        // the binary vs. multiclass path) and the per-class tp / support /
        // predicted counts used by the multiclass path. Labels are keyed by
        // their f64 bit pattern; note that -0.0 and +0.0 have distinct bit
        // patterns and would be counted as separate classes — an existing
        // convention shared with Precision and Recall.
        let mut classes_set: HashSet<u64> = HashSet::new();
        let mut predicted: HashMap<u64, usize> = HashMap::new();
        let mut support: HashMap<u64, usize> = HashMap::new();
        let mut tp_map: HashMap<u64, usize> = HashMap::new();
        for i in 0..n {
            let t_bits = y_true.get(i).to_f64_bits();
            classes_set.insert(t_bits);
            *support.entry(t_bits).or_insert(0) += 1;
            *predicted.entry(y_pred.get(i).to_f64_bits()).or_insert(0) += 1;
            if *y_true.get(i) == *y_pred.get(i) {
                *tp_map.entry(t_bits).or_insert(0) += 1;
            }
        }
        let classes = classes_set.len();

        if classes == 2 {
            // Binary case: F-measure of the positive class. The positive class
            // is assumed to be the label with the higher bit representation
            // (i.e. 1.0 when labels are 0.0/1.0) — the convention baked into
            // Precision and Recall, which already return the positive-class
            // scores.
            let p = Precision::new().get_score(y_true, y_pred);
            let r = Recall::new().get_score(y_true, y_pred);
            (1f64 + beta2) * (p * r) / ((beta2 * p) + r)
        } else {
            // Multiclass case (including classes == 1, where the macro F-beta
            // is just the single class's F-beta): macro F-measure is the mean
            // of the per-class F-measures, not the F-measure of the
            // macro-averaged precision and recall.
            let mut fbeta_sum = 0.0;
            for &bits in &classes_set {
                let tp = *tp_map.get(&bits).unwrap_or(&0);
                let pred_count = *predicted.get(&bits).unwrap_or(&0);
                let support_count = *support.get(&bits).unwrap_or(&0);
                let p_c = if pred_count > 0 {
                    tp as f64 / pred_count as f64
                } else {
                    0.0
                };
                let r_c = if support_count > 0 {
                    tp as f64 / support_count as f64
                } else {
                    0.0
                };
                let denom = beta2 * p_c + r_c;
                if denom > 0.0 {
                    fbeta_sum += (1f64 + beta2) * p_c * r_c / denom;
                }
            }
            // classes >= 1 is guaranteed here: n > 0 (early return above) means
            // classes_set is non-empty.
            fbeta_sum / classes as f64
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
    fn f1() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let beta = 1.0;
        let score1: f64 = F1::new_with(beta).get_score(&y_true, &y_pred);
        let score2: f64 = F1::new_with(beta).get_score(&y_true, &y_true);

        println!("{score1:?}");
        println!("{score2:?}");

        assert!((score1 - 0.57142857).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn f1_multiclass_macro() {
        // Macro F1 is the mean of the per-class F1 scores, not the F1 of the
        // macro-averaged precision and recall. Here precision (0.888889) and
        // recall (0.833333) both match sklearn, so this isolates the aggregation:
        // per-class F1 = [2/3, 0.8, 1.0], mean = 0.822222 (sklearn macro f1).
        let y_true: Vec<f64> = vec![0., 0., 1., 1., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 1., 1., 2., 2.];

        let score: f64 = F1::new_with(1.0).get_score(&y_true, &y_pred);
        let expected = (2.0 / 3.0 + 0.8 + 1.0) / 3.0;
        assert!((score - expected).abs() < 1e-8);
        assert!((score - 0.822222).abs() < 1e-6);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn f1_multiclass_macro_beta() {
        // Same case with beta != 1: still the mean of the per-class F-beta scores.
        let y_true: Vec<f64> = vec![0., 0., 1., 1., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 1., 1., 2., 2.];

        // per-class (precision, recall): (1, 0.5), (2/3, 1), (1, 1)
        let fbeta = |b: f64, p: f64, r: f64| (1.0 + b * b) * p * r / (b * b * p + r);
        for beta in [0.5, 2.0] {
            let expected =
                (fbeta(beta, 1.0, 0.5) + fbeta(beta, 2.0 / 3.0, 1.0) + fbeta(beta, 1.0, 1.0)) / 3.0;
            let score: f64 = F1::new_with(beta).get_score(&y_true, &y_pred);
            assert!((score - expected).abs() < 1e-8);
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn f1_single_class() {
        // When y_true has a single class, the multiclass path computes the
        // F-beta for that one class (macro of one value is the value itself).
        let y_true: Vec<f64> = vec![0., 0., 0.];

        // Perfect predictions -> F1 = 1.0.
        let perfect: f64 = F1::new_with(1.0).get_score(&y_true, &y_true);
        assert!((perfect - 1.0).abs() < 1e-8);

        // tp=2, predicted(class 0)=2, support(class 0)=3
        //   -> p=1.0, r=2/3 -> F1 = 2 * 1.0 * (2/3) / (1.0 + 2/3) = 0.8
        let y_pred: Vec<f64> = vec![0., 1., 0.];
        let score: f64 = F1::new_with(1.0).get_score(&y_true, &y_pred);
        assert!((score - 0.8).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn f1_multiclass_imbalanced() {
        let y_true: Vec<f64> = vec![0., 0., 1., 2., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 1., 2., 0., 2.];

        // per-class F1: class0 0.5, class1 2/3, class2 0.8 (matches sklearn macro f1)
        let score: f64 = F1::new_with(1.0).get_score(&y_true, &y_pred);
        let expected = (0.5 + 2.0 / 3.0 + 0.8) / 3.0;
        assert!((score - expected).abs() < 1e-8);

        let perfect: f64 = F1::new_with(1.0).get_score(&y_true, &y_true);
        assert!((perfect - 1.0).abs() < 1e-8);
    }
}
