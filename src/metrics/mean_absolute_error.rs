use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::FloatExt;

#[derive(Serialize, Deserialize, Debug)]
pub struct MeanAbsoluteError {}

impl MeanAbsoluteError {
    pub fn get_score<T: FloatExt, V: BaseVector<T>>(&self, y_true: &V, y_pred: &V) -> T {
        if y_true.len() != y_pred.len() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.len(),
                y_pred.len()
            );
        }

        let n = y_true.len();
        let mut ras = T::zero();
        for i in 0..n {
            ras = ras + (y_true.get(i) - y_pred.get(i)).abs();
        }

        ras / T::from_usize(n).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_absolute_error() {
        let y_true: Vec<f64> = vec![3., -0.5, 2., 7.];
        let y_pred: Vec<f64> = vec![2.5, 0.0, 2., 8.];

        let score1: f64 = MeanAbsoluteError {}.get_score(&y_pred, &y_true);
        let score2: f64 = MeanAbsoluteError {}.get_score(&y_true, &y_true);

        println!("{}", score1);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 0.0).abs() < 1e-8);
    }
}
