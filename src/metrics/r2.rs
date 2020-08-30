use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::RealNumber;

#[derive(Serialize, Deserialize, Debug)]
pub struct R2 {}

impl R2 {
    pub fn get_score<T: RealNumber, V: BaseVector<T>>(&self, y_true: &V, y_pred: &V) -> T {
        if y_true.len() != y_pred.len() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.len(),
                y_pred.len()
            );
        }

        let n = y_true.len();

        let mut mean = T::zero();

        for i in 0..n {
            mean = mean + y_true.get(i);
        }

        mean = mean / T::from_usize(n).unwrap();

        let mut ss_tot = T::zero();
        let mut ss_res = T::zero();

        for i in 0..n {
            let y_i = y_true.get(i);
            let f_i = y_pred.get(i);
            ss_tot = ss_tot + (y_i - mean).square();
            ss_res = ss_res + (y_i - f_i).square();
        }

        T::one() - (ss_res / ss_tot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2() {
        let y_true: Vec<f64> = vec![3., -0.5, 2., 7.];
        let y_pred: Vec<f64> = vec![2.5, 0.0, 2., 8.];

        let score1: f64 = R2 {}.get_score(&y_true, &y_pred);
        let score2: f64 = R2 {}.get_score(&y_true, &y_true);

        assert!((score1 - 0.948608137).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
