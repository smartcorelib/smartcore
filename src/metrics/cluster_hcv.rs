#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::base::Array1;
use crate::metrics::cluster_helpers::*;
use crate::num::Number;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// Homogeneity, completeness and V-Measure scores.
pub struct HCVScore {}

impl HCVScore {
    /// Computes Homogeneity, completeness and V-Measure scores at once.
    /// * `labels_true` - ground truth class labels to be used as a reference.
    /// * `labels_pred` - cluster labels to evaluate.    
    pub fn get_score<T: Number + Ord, V: Array1<T>>(
        &self,
        labels_true: &V,
        labels_pred: &V,
    ) -> (f64, f64, f64) {
        let entropy_c = entropy(labels_true);
        let entropy_k = entropy(labels_pred);
        let contingency = contingency_matrix(labels_true, labels_pred);
        let mi = mutual_info_score(&contingency);

        let homogeneity = entropy_c.map(|e| mi / e).unwrap_or(0f64);
        let completeness = entropy_k.map(|e| mi / e).unwrap_or(0f64);

        let v_measure_score = if homogeneity + completeness == 0f64 {
            0f64
        } else {
            2f64 * homogeneity * completeness / (1f64 * homogeneity + completeness)
        };

        (homogeneity, completeness, v_measure_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn homogeneity_score() {
        let v1 = vec![0, 0, 1, 1, 2, 0, 4];
        let v2 = vec![1, 0, 0, 0, 0, 1, 0];
        let scores = HCVScore {}.get_score(&v1, &v2);

        assert!((0.2548 - scores.0).abs() < 1e-4);
        assert!((0.5440 - scores.1).abs() < 1e-4);
        assert!((0.3471 - scores.2).abs() < 1e-4);
    }
}
