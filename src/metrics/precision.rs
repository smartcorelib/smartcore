use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::FloatExt;

#[derive(Serialize, Deserialize, Debug)]
pub struct Precision {}

impl Precision {
    pub fn get_score<T: FloatExt, V: BaseVector<T>>(&self, y_true: &V, y_prod: &V) -> T {
        if y_true.len() != y_prod.len() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.len(),
                y_prod.len()
            );
        }

        let mut tp = 0;
        let mut p = 0;
        let n = y_true.len();
        for i in 0..n {
            if y_true.get(i) != T::zero() && y_true.get(i) != T::one() {
                panic!(
                    "Precision can only be applied to binary classification: {}",
                    y_true.get(i)
                );
            }

            if y_prod.get(i) != T::zero() && y_prod.get(i) != T::one() {
                panic!(
                    "Precision can only be applied to binary classification: {}",
                    y_prod.get(i)
                );
            }

            if y_prod.get(i) == T::one() {
                p += 1;

                if y_true.get(i) == T::one() {
                    tp += 1;
                }
            }
        }

        T::from_i64(tp).unwrap() / T::from_i64(p).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision() {
        let y_true: Vec<f64> = vec![0., 1., 1., 0.];
        let y_pred: Vec<f64> = vec![0., 0., 1., 1.];

        let score1: f64 = Precision {}.get_score(&y_pred, &y_true);
        let score2: f64 = Precision {}.get_score(&y_pred, &y_pred);

        assert!((score1 - 0.5).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
