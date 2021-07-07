//! # Hamming Distance
//!
//! Hamming Distance measures the similarity between two integer-valued vectors of the same length.
//! Given two vectors \\( x \in ℝ^n \\), \\( y \in ℝ^n \\) the hamming distance between \\( x \\) and \\( y \\), \\( d(x, y) \\), is the number of places where \\( x \\) and \\( y \\) differ.
//!
//! Example:
//!
//! ```
//! use smartcore::math::distance::Distance;
//! use smartcore::math::distance::hamming::Hamming;
//!
//! let a = vec![1, 0, 0, 1, 0, 0, 1];
//! let b = vec![1, 1, 0, 0, 1, 0, 1];
//!
//! let h: f64 = Hamming::new().distance(&a, &b);
//!
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use crate::linalg::base::ArrayView1;
use crate::num::Number;
use super::Distance;

/// While comparing two integer-valued vectors of equal length, Hamming distance is the number of bit positions in which the two bits are different
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Hamming<T: Number> {
    _t: PhantomData<T>
}

impl<T: Number> Hamming<T> {

    pub fn new() -> Hamming<T> {
        Hamming {_t: PhantomData}
    }
}

impl<T: Number, A: ArrayView1<T>> Distance<A> for Hamming<T> {
    fn distance(&self, x: &A, y: &A) -> f64 {
        if x.shape() != y.shape() {
            panic!("Input vector sizes are different");
        }
        
        let dist: usize = x.iterator(0).zip(y.iterator(0)).map(|(a, b)| {
            match a != b {
                true => 1,
                false => 0
            }}).sum();        

        dist as f64 / x.shape() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn hamming_distance() {
        let a = vec![1, 0, 0, 1, 0, 0, 1];
        let b = vec![1, 1, 0, 0, 1, 0, 1];

        let h: f64 = Hamming::new().distance(&a, &b);

        assert!((h - 0.42857142).abs() < 1e-8);
    }
}
