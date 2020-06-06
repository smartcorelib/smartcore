use serde::{Deserialize, Serialize};

use crate::math::num::FloatExt;

use super::Distance;

#[derive(Serialize, Deserialize, Debug)]
pub struct Hamming {}

impl<T: PartialEq, F: FloatExt> Distance<Vec<T>, F> for Hamming {
    fn distance(&self, x: &Vec<T>, y: &Vec<T>) -> F {
        if x.len() != y.len() {
            panic!("Input vector sizes are different");
        }

        let mut dist = 0;
        for i in 0..x.len() {
            if x[i] != y[i] {
                dist += 1;
            }
        }

        F::from_i64(dist).unwrap() / F::from_usize(x.len()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minkowski_distance() {
        let a = vec![1, 0, 0, 1, 0, 0, 1];
        let b = vec![1, 1, 0, 0, 1, 0, 1];

        let h: f64 = Hamming {}.distance(&a, &b);

        assert!((h - 0.42857142).abs() < 1e-8);
    }
}
