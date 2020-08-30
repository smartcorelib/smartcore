use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::RealNumber;

#[derive(Serialize, Deserialize, Debug)]
pub struct Accuracy {}

impl Accuracy {
    pub fn get_score<T: RealNumber, V: BaseVector<T>>(&self, y_true: &V, y_pred: &V) -> T {
        if y_true.len() != y_pred.len() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.len(),
                y_pred.len()
            );
        }

        let n = y_true.len();

        let mut positive = 0;
        for i in 0..n {
            if y_true.get(i) == y_pred.get(i) {
                positive += 1;
            }
        }

        T::from_i64(positive).unwrap() / T::from_usize(n).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy() {
        let y_pred: Vec<f64> = vec![0., 2., 1., 3.];
        let y_true: Vec<f64> = vec![0., 1., 2., 3.];

        let score1: f64 = Accuracy {}.get_score(&y_pred, &y_true);
        let score2: f64 = Accuracy {}.get_score(&y_true, &y_true);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
