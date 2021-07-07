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
//! let l1: f64 = Minkowski::new(1).distance(&x, &y);
//! let l2: f64 = Minkowski::new(2).distance(&x, &y);
//!
//! ```
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::linalg::base::ArrayView1;
use crate::num::Number;

use super::Distance;

/// Defines the Minkowski distance of order `p`
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Minkowski<T: Number> {
    /// order, integer
    pub p: u16,
    _t: PhantomData<T>
}

impl<T: Number> Minkowski<T> {

    pub fn new(p: u16) -> Minkowski<T> {
        Minkowski {p, _t: PhantomData}
    }
}

impl<T: Number, A: ArrayView1<T>> Distance<A> for Minkowski<T> {
    fn distance(&self, x: &A, y: &A) -> f64 {
        if x.shape() != y.shape() {
            panic!("Input vector sizes are different");
        }
        if self.p < 1 {
            panic!("p must be at least 1");
        }

        let mut dist = 0f64;
        let p_t = self.p as f64;

        let dist: f64 = x.iterator(0).zip(y.iterator(0)).map(|(&a, &b)| (a - b).to_f64().unwrap().abs().powf(p_t)).sum();
        
        dist.powf(1f64 / p_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn minkowski_distance() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let l1: f64 = Minkowski::new(1).distance(&a, &b);
        let l2: f64 = Minkowski::new(2).distance(&a, &b);
        let l3: f64 = Minkowski::new(3).distance(&a, &b);

        assert!((l1 - 9.0).abs() < 1e-8);
        assert!((l2 - 5.19615242).abs() < 1e-8);
        assert!((l3 - 4.32674871).abs() < 1e-8);
    }

    #[test]
    #[should_panic(expected = "p must be at least 1")]
    fn minkowski_distance_negative_p() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let _: f64 = Minkowski::new(0).distance(&a, &b);
    }
}
