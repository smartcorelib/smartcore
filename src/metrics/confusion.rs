//! Shared per-class confusion-count helpers for classification metrics.
//!
//! [`ConfusionCounts`] computes, in a single pass over `(y_true, y_pred)`,
//! the per-class true-positive, predicted, and support counts used by
//! [`Precision`](crate::metrics::precision::Precision),
//! [`Recall`](crate::metrics::recall::Recall), and
//! [`F1`](crate::metrics::f1::F1).
//!
//! Labels are keyed by their `f64` bit pattern; note that `-0.0` and `+0.0`
//! have distinct bit patterns and would be counted as separate classes. This
//! convention is shared across the classification metrics.

use std::collections::{HashMap, HashSet};

use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::realnum::RealNumber;

/// Per-class confusion counts for a classification result.
///
/// Built in a single pass over `(y_true, y_pred)`. Exposes per-class
/// true-positive, predicted, and support counts so that `Precision`,
/// `Recall`, and `F1` can derive their per-class scores from a single
/// source of truth instead of each re-implementing the bookkeeping.
pub(crate) struct ConfusionCounts {
    classes_set: HashSet<u64>,
    predicted: HashMap<u64, usize>,
    support: HashMap<u64, usize>,
    tp_map: HashMap<u64, usize>,
}

impl ConfusionCounts {
    /// Compute per-class confusion counts in a single pass over
    /// `(y_true, y_pred)`.
    ///
    /// Named `new` rather than `from` to avoid shadowing the conventional
    /// `std::convert::From` trait method (the two-argument signature does
    /// not collide with `From::from`'s single-argument form, but the
    /// shadowing is still confusing for readers).
    pub(crate) fn new<T: RealNumber>(
        y_true: &dyn ArrayView1<T>,
        y_pred: &dyn ArrayView1<T>,
    ) -> Self {
        let n = y_true.shape();
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
        Self {
            classes_set,
            predicted,
            support,
            tp_map,
        }
    }

    /// The set of label bit patterns observed in `y_true`.
    pub(crate) fn classes_set(&self) -> &HashSet<u64> {
        &self.classes_set
    }

    /// Number of predictions equal to the label with the given bit pattern.
    pub(crate) fn predicted(&self, bits: u64) -> usize {
        *self.predicted.get(&bits).unwrap_or(&0)
    }

    /// Number of `y_true` entries equal to the label with the given bit
    /// pattern (the class support).
    pub(crate) fn support(&self, bits: u64) -> usize {
        *self.support.get(&bits).unwrap_or(&0)
    }

    /// Number of true positives for the label with the given bit pattern.
    pub(crate) fn tp(&self, bits: u64) -> usize {
        *self.tp_map.get(&bits).unwrap_or(&0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits_of(v: f64) -> u64 {
        v.to_f64_bits()
    }

    #[test]
    fn confusion_counts_binary_basic() {
        // y_true = [0, 1, 1, 0], y_pred = [0, 0, 1, 1]
        let y_true: Vec<f64> = vec![0., 1., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1.];
        let counts = ConfusionCounts::new(&y_true, &y_pred);

        assert_eq!(counts.classes_set().len(), 2);
        assert!(counts.classes_set().contains(&bits_of(0.0)));
        assert!(counts.classes_set().contains(&bits_of(1.0)));

        // Class 0: support=2, predicted=2 (y_pred has two 0s), tp=1
        assert_eq!(counts.support(bits_of(0.0)), 2);
        assert_eq!(counts.predicted(bits_of(0.0)), 2);
        assert_eq!(counts.tp(bits_of(0.0)), 1);

        // Class 1: support=2, predicted=2 (y_pred has two 1s), tp=1
        assert_eq!(counts.support(bits_of(1.0)), 2);
        assert_eq!(counts.predicted(bits_of(1.0)), 2);
        assert_eq!(counts.tp(bits_of(1.0)), 1);
    }

    #[test]
    fn confusion_counts_multiclass() {
        // y_true = [0, 0, 1, 2, 2, 2], y_pred = [0, 1, 1, 2, 0, 2]
        let y_true: Vec<f64> = vec![0., 0., 1., 2., 2., 2.];
        let y_pred: Vec<f64> = vec![0., 1., 1., 2., 0., 2.];
        let counts = ConfusionCounts::new(&y_true, &y_pred);

        assert_eq!(counts.classes_set().len(), 3);

        // Class 0: support=2, predicted=2, tp=1
        assert_eq!(counts.support(bits_of(0.0)), 2);
        assert_eq!(counts.predicted(bits_of(0.0)), 2);
        assert_eq!(counts.tp(bits_of(0.0)), 1);

        // Class 1: support=1, predicted=2, tp=1
        assert_eq!(counts.support(bits_of(1.0)), 1);
        assert_eq!(counts.predicted(bits_of(1.0)), 2);
        assert_eq!(counts.tp(bits_of(1.0)), 1);

        // Class 2: support=3, predicted=2, tp=2
        assert_eq!(counts.support(bits_of(2.0)), 3);
        assert_eq!(counts.predicted(bits_of(2.0)), 2);
        assert_eq!(counts.tp(bits_of(2.0)), 2);
    }

    #[test]
    fn confusion_counts_spurious_predicted_label() {
        // y_pred contains label 2 which never appears in y_true. The
        // `predicted` map records it, but `classes_set` (sourced from
        // y_true) does not include it, so it is silently ignored by
        // per-class metrics that iterate `classes_set`.
        let y_true: Vec<f64> = vec![0., 0., 1., 1.];
        let y_pred: Vec<f64> = vec![0., 2., 1., 1.];
        let counts = ConfusionCounts::new(&y_true, &y_pred);

        assert_eq!(counts.classes_set().len(), 2);
        assert!(!counts.classes_set().contains(&bits_of(2.0)));

        // The spurious label is tracked in `predicted`...
        assert_eq!(counts.predicted(bits_of(2.0)), 1);
        // ...but has no support or tp entry.
        assert_eq!(counts.support(bits_of(2.0)), 0);
        assert_eq!(counts.tp(bits_of(2.0)), 0);
    }

    #[test]
    fn confusion_counts_empty_input() {
        let y_true: Vec<f64> = vec![];
        let y_pred: Vec<f64> = vec![];
        let counts = ConfusionCounts::new(&y_true, &y_pred);

        assert!(counts.classes_set().is_empty());
        assert_eq!(counts.predicted(bits_of(0.0)), 0);
        assert_eq!(counts.support(bits_of(0.0)), 0);
        assert_eq!(counts.tp(bits_of(0.0)), 0);
    }

    #[test]
    fn confusion_counts_perfect_predictions() {
        let y_true: Vec<f64> = vec![0., 1., 2., 0., 1.];
        let counts = ConfusionCounts::new(&y_true, &y_true);

        for &bits in counts.classes_set() {
            assert_eq!(counts.tp(bits), counts.support(bits));
            assert_eq!(counts.tp(bits), counts.predicted(bits));
        }
    }
}
