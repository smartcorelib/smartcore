//! # Manhattan Distance
//!
//! The Manhattan distance between two points \\(x \in ℝ^n \\) and \\( y \in ℝ^n \\) in n-dimensional space is the sum of the distances in each dimension.
//!
//! \\[ d(x, y) = \sum_{i=0}^n \lvert x_i - y_i \rvert \\]
//!
//! Example:
//!
//! ```
//! use smartcore::math::distance::Distance;
//! use smartcore::math::distance::manhattan::Manhattan;
//!
//! let x = vec![1., 1.];
//! let y = vec![2., 2.];
//!
//! let l1: f64 = Manhattan {}.distance(&x, &y);
//! ```
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::math::num::RealNumber;

use super::Distance;

/// Manhattan distance
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Manhattan {}

impl<T: RealNumber> Distance<Vec<T>, T> for Manhattan {
    fn distance(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        if x.len() != y.len() {
            panic!("Input vector sizes are different");
        }

        let mut dist = T::zero();
        for i in 0..x.len() {
            dist += (x[i] - y[i]).abs();
        }

        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn manhattan_distance() {
        let a = vec![1., 2., 3.];
        let b = vec![4., 5., 6.];

        let l1: f64 = Manhattan {}.distance(&a, &b);

        assert!((l1 - 9.0).abs() < 1e-8);
    }
}
