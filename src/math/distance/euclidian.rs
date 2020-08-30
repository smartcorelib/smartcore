//! # Euclidian Metric Distance
//!
//! The Euclidean distance (L2) between two points \\( x \\) and \\( y \\) in n-space is defined as
//!
//! \\[ d(x, y) = \sqrt{\sum_{i=1}^n (x-y)^2} \\]
//!
//! Example:
//!
//! ```
//! use smartcore::math::distance::Distance;
//! use smartcore::math::distance::euclidian::Euclidian;
//!
//! let x = vec![1., 1.];
//! let y = vec![2., 2.];
//!
//! let l2: f64 = Euclidian{}.distance(&x, &y);
//! ```
//!
//! <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
use serde::{Deserialize, Serialize};

use crate::math::num::RealNumber;

use super::Distance;

/// Euclidean distance is a measure of the true straight line distance between two points in Euclidean n-space.
#[derive(Serialize, Deserialize, Debug)]
pub struct Euclidian {}

impl Euclidian {
    pub(crate) fn squared_distance<T: RealNumber>(x: &Vec<T>, y: &Vec<T>) -> T {
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
