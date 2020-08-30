use serde::{Deserialize, Serialize};

use crate::math::num::RealNumber;

use super::Distance;

#[derive(Serialize, Deserialize, Debug)]
pub struct Euclidian {}

impl Euclidian {
    pub fn squared_distance<T: RealNumber>(x: &Vec<T>, y: &Vec<T>) -> T {
        if x.len() != y.len() {
            panic!("Input vector sizes are different.");
        }

        let mut sum = T::zero();
        for i in 0..x.len() {
            sum = sum + (x[i] - y[i]).powf(T::two());
        }

        sum
    }
}

impl<T: RealNumber> Distance<Vec<T>, T> for Euclidian {
    fn distance(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        Euclidian::squared_distance(x, y).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_distance() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let l2: f64 = Euclidian {}.distance(&a, &b);

        assert!((l2 - 5.19615242).abs() < 1e-8);
    }
}
