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
//! let l1: f64 = Manhattan::new().distance(&x, &y);
//! ```
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::linalg::base::ArrayView1;
use crate::num::Number;

use super::Distance;

/// Manhattan distance
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Manhattan<T: Number> {
    _t: PhantomData<T>
}

impl<T: Number> Manhattan<T> {

    pub fn new() -> Manhattan<T> {
        Manhattan {_t: PhantomData}
    }
}

impl<T: Number, A: ArrayView1<T>> Distance<A> for Manhattan<T> {
    fn distance(&self, x: &A, y: &A) -> f64 {
        if x.shape() != y.shape() {
            panic!("Input vector sizes are different");
        }        

        let dist: f64 = x.iterator(0).zip(y.iterator(0)).map(|(&a, &b)| (a - b).to_f64().unwrap().abs()).sum();

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

        let l1: f64 = Manhattan::new().distance(&a, &b);

        assert!((l1 - 9.0).abs() < 1e-8);
    }
}
