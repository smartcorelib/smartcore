use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::RealNumber;
use crate::metrics::cluster_helpers::*;

#[derive(Serialize, Deserialize, Debug)]
/// Mean Absolute Error
pub struct HCVScore {}

impl HCVScore {
    /// Computes mean absolute error
    /// * `y_true` - Ground truth (correct) target values.
    /// * `y_pred` - Estimated target values.    
    pub fn get_score<T: RealNumber, V: BaseVector<T>>(
        &self,
        labels_true: &V,
        labels_pred: &V,
    ) -> (T, T, T) {
        let labels_true = labels_true.to_vec();
        let labels_pred = labels_pred.to_vec();
        let entropy_c = entropy(&labels_true);
        let entropy_k = entropy(&labels_pred);
        let contingency = contingency_matrix(&labels_true, &labels_pred);
        let mi: T = mutual_info_score(&contingency);

        let homogeneity = entropy_c.map(|e| mi / e).unwrap_or(T::one());
        let completeness = entropy_k.map(|e| mi / e).unwrap_or(T::one());

        let v_measure_score = if homogeneity + completeness == T::zero() {
            T::zero()
        } else {
            T::two() * homogeneity * completeness / (T::one() * homogeneity + completeness)
        };

        (homogeneity, completeness, v_measure_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homogeneity_score() {
        let v1 = vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 4.0];
        let v2 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let scores = HCVScore {}.get_score(&v1, &v2);

        assert!((0.2548f32 - scores.0).abs() < 1e-4);
        assert!((0.5440f32 - scores.1).abs() < 1e-4);
        assert!((0.3471f32 - scores.2).abs() < 1e-4);
    }
}
