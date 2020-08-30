//! # Minkowski Distance
//!
//! The Minkowski distance  of order _p_ (where _p_ is an integer) is a metric in a normed vector space which can be considered as a generalization of both the Euclidean distance and the Manhattan distance.
//! The Manhattan distance between two points \\(x \in ℝ^n \\) and \\( y \in ℝ^n \\) in n-dimensional space is defined as:
//!
//! \\[ d(x, y) = \left(\sum_{i=0}^n \lvert x_i - y_i \rvert^p\right)^{1/p} \\]
//!
//! Example:
//!
//! ```
//! use smartcore::math::distance::Distance;
//! use smartcore::math::distance::minkowski::Minkowski;
//!
//! let x = vec![1., 1.];
//! let y = vec![2., 2.];
//!
//! let l1: f64 = Minkowski { p: 1 }.distance(&x, &y);
//! let l2: f64 = Minkowski { p: 2 }.distance(&x, &y);
//!
//! ```
//! <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

use serde::{Deserialize, Serialize};

use crate::math::num::RealNumber;

use super::Distance;

/// Defines the Minkowski distance of order `p`
#[derive(Serialize, Deserialize, Debug)]
pub struct Minkowski {
    /// order, integer
    pub p: u16,
}

impl<T: RealNumber> Distance<Vec<T>, T> for Minkowski {
    fn distance(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        if x.len() != y.len() {
            panic!("Input vector sizes are different");
        }
        if self.p < 1 {
            panic!("p must be at least 1");
        }

        let mut dist = T::zero();
        let p_t = T::from_u16(self.p).unwrap();

        for i in 0..x.len() {
            let d = (x[i] - y[i]).abs();
            dist = dist + d.powf(p_t);
        }

        dist.powf(T::one() / p_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minkowski_distance() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let l1: f64 = Minkowski { p: 1 }.distance(&a, &b);
        let l2: f64 = Minkowski { p: 2 }.distance(&a, &b);
        let l3: f64 = Minkowski { p: 3 }.distance(&a, &b);

        assert!((l1 - 9.0).abs() < 1e-8);
        assert!((l2 - 5.19615242).abs() < 1e-8);
        assert!((l3 - 4.32674871).abs() < 1e-8);
    }

    #[test]
    #[should_panic(expected = "p must be at least 1")]
    fn minkowski_distance_negative_p() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let _: f64 = Minkowski { p: 0 }.distance(&a, &b);
    }
}
