//! # Euclidian Metric Distance
//!
//! The Euclidean distance (L2) between two points \\( x \\) and \\( y \\) in n-space is defined as
//!
//! \\[ d(x, y) = \sqrt{\sum_{i=1}^n (x-y)^2} \\]
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::distance::Distance;
//! use smartcore::metrics::distance::euclidian::Euclidian;
//!
//! let x = vec![1., 1.];
//! let y = vec![2., 2.];
//!
//! let l2: f64 = Euclidian::new().distance(&x, &y);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::basenum::Number;

use super::Distance;

/// Euclidean distance is a measure of the true straight line distance between two points in Euclidean n-space.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Euclidian<T> {
    _t: PhantomData<T>,
}

impl<T: Number> Default for Euclidian<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Number> Euclidian<T> {
    /// instatiate the initial structure
    pub fn new() -> Euclidian<T> {
        Euclidian { _t: PhantomData }
    }

    /// return sum of squared distances
    #[inline]
    pub(crate) fn squared_distance<A: ArrayView1<T>>(x: &A, y: &A) -> f64 {
        if x.shape() != y.shape() {
            panic!("Input vector sizes are different.");
        }

        let sum: f64 = x
            .iterator(0)
            .zip(y.iterator(0))
            .map(|(&a, &b)| {
                let r = a - b;
                (r * r).to_f64().unwrap()
            })
            .sum();

        sum
    }
}

impl<T: Number, A: ArrayView1<T>> Distance<A> for Euclidian<T> {
    fn distance(&self, x: &A, y: &A) -> f64 {
        Euclidian::squared_distance(x, y).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn squared_distance() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];

        let l2: f64 = Euclidian::new().distance(&a, &b);

        assert!((l2 - 5.19615242).abs() < 1e-8);
    }
}
