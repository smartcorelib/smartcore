use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::FloatExt;
use crate::metrics::precision::Precision;
use crate::metrics::recall::Recall;

#[derive(Serialize, Deserialize, Debug)]
pub struct F1 {}

impl F1 {
    pub fn get_score<T: FloatExt, V: BaseVector<T>>(&self, y_true: &V, y_pred: &V) -> T {
        if y_true.len() != y_pred.len() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.len(),
                y_pred.len()
            );
        }
        let beta2 = T::one();

        let p = Precision {}.get_score(y_true, y_pred);
        let r = Recall {}.get_score(y_true, y_pred);

        (T::one() + beta2) * (p * r) / (beta2 * p + r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f1() {
        let y_pred: Vec<f64> = vec![0., 0., 1., 1., 1., 1.];
        let y_true: Vec<f64> = vec![0., 1., 1., 0., 1., 0.];

        let score1: f64 = F1 {}.get_score(&y_pred, &y_true);
        let score2: f64 = F1 {}.get_score(&y_true, &y_true);

        assert!((score1 - 0.57142857).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
