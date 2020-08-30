use serde::{Deserialize, Serialize};

use crate::math::num::RealNumber;

use super::Distance;

#[derive(Serialize, Deserialize, Debug)]
pub struct Manhattan {}

impl<T: RealNumber> Distance<Vec<T>, T> for Manhattan {
    fn distance(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        if x.len() != y.len() {
            panic!("Input vector sizes are different");
        }

        let mut dist = T::zero();
        for i in 0..x.len() {
            dist = dist + (x[i] - y[i]).abs();
        }

        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manhattan_distance() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let l1: f64 = Manhattan {}.distance(&a, &b);

        assert!((l1 - 9.0).abs() < 1e-8);
    }
}
