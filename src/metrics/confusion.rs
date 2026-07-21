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
    /// Compute per-class confusion counts in a single pass.
    pub(crate) fn from<T: RealNumber>(
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
