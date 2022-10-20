use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::ArrayView1;
use crate::metrics::cluster_helpers::*;
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

use crate::metrics::Metrics;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// Homogeneity, completeness and V-Measure scores.
pub struct HCVScore<T> {
    _phantom: PhantomData<T>,
    homogeneity: Option<f64>,
    completeness: Option<f64>,
    v_measure: Option<f64>,
}

impl<T: Number + Ord> HCVScore<T> {
    /// return homogenity score
    pub fn homogeneity(&self) -> Option<f64> {
        self.homogeneity
    }
    /// return completeness score
    pub fn completeness(&self) -> Option<f64> {
        self.completeness
    }
    /// return v_measure score
    pub fn v_measure(&self) -> Option<f64> {
        self.v_measure
    }
    /// run computation for measures
    pub fn compute(&mut self, y_true: &dyn ArrayView1<T>, y_pred: &dyn ArrayView1<T>) -> () {
        let entropy_c: Option<f64> = entropy(y_true);
        let entropy_k: Option<f64> = entropy(y_pred);
        let contingency = contingency_matrix(y_true, y_pred);
        let mi = mutual_info_score(&contingency);

        let homogeneity = entropy_c.map(|e| mi / e).unwrap_or(0f64);
        let completeness = entropy_k.map(|e| mi / e).unwrap_or(0f64);

        let v_measure_score = if homogeneity + completeness == 0f64 {
            0f64
        } else {
            2.0f64 * homogeneity * completeness / (1.0f64 * homogeneity + completeness)
        };

        self.homogeneity = Some(homogeneity);
        self.completeness = Some(completeness);
        self.v_measure = Some(v_measure_score);
    }
}

impl<T: Number + Ord> Metrics<T> for HCVScore<T> {
    /// create a typed object to call HCVScore functions
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
            homogeneity: Option::None,
            completeness: Option::None,
            v_measure: Option::None,
        }
    }
    fn new_with(_parameter: f64) -> Self {
        Self {
            _phantom: PhantomData,
            homogeneity: Option::None,
            completeness: Option::None,
            v_measure: Option::None,
        }
    }
    /// Computes Homogeneity, completeness and V-Measure scores at once.
    /// * `y_true` - ground truth class labels to be used as a reference.
    /// * `y_pred` - cluster labels to evaluate.    
    fn get_score(&self, y_true: &dyn ArrayView1<T>, y_pred: &dyn ArrayView1<T>) -> f64 {
        // this functions should not be used for this struct
        // use homogeneity(), completeness(), v_measure()
        // TODO: implement Metrics -> Result<T, Failed>
        0f64
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
        let mut scores = HCVScore::new();
        scores.compute(&v1, &v2);

        assert!((0.2548 - scores.homogeneity.unwrap() as f64).abs() < 1e-4);
        assert!((0.5440 - scores.completeness.unwrap() as f64).abs() < 1e-4);
        assert!((0.3471 - scores.v_measure.unwrap() as f64).abs() < 1e-4);
    }
}
