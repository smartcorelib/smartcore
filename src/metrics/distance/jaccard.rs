//! # Jaccard Distance
//!
//! Jaccard Distance measures dissimilarity between two integer-valued vectors of the same length.
//! Given two vectors \\( x \in ℝ^n \\), \\( y \in ℝ^n \\) the Jaccard distance between \\( x \\) and \\( y \\) is defined as
//!
//! \\[ d(x, y) = 1 - \frac{|x \cap y|}{|x \cup y|} \\]
//!
//! where \\(|x \cap y|\\) is the number of positions where both vectors are non-zero,
//! and \\(|x \cup y|\\) is the number of positions where at least one of the vectors is non-zero.
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::distance::Distance;
//! use smartcore::metrics::distance::jaccard::Jaccard;
//!
//! let a = vec![1, 0, 1, 1];
//! let b = vec![1, 1, 0, 1];
//!
//! let j: f64 = Jaccard::new().distance(&a, &b);
//!
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use super::Distance;
use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::basenum::Number;

/// Jaccard distance between two integer-valued vectors
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Jaccard<T: Number> {
    _t: PhantomData<T>,
}

impl<T: Number> Jaccard<T> {
    /// instatiate the initial structure
    pub fn new() -> Jaccard<T> {
        Jaccard { _t: PhantomData }
    }
}

impl<T: Number> Default for Jaccard<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Number, A: ArrayView1<T>> Distance<A> for Jaccard<T> {
    fn distance(&self, x: &A, y: &A) -> f64 {
        if x.shape() != y.shape() {
            panic!("Input vector sizes are different");
        }

        let (intersection, union): (usize, usize) = x
            .iterator(0)
            .zip(y.iterator(0))
            .map(|(a, b)| {
                let a_nz = *a != T::zero();
                let b_nz = *b != T::zero();

                match (a_nz, b_nz) {
                    (true, true) => (1, 1),
                    (true, false) | (false, true) => (0, 1),
                    (false, false) => (0, 0),
                }
            })
            .fold((0, 0), |acc, v| (acc.0 + v.0, acc.1 + v.1));

        if union == 0 {
            0.0
        } else {
            1.0 - intersection as f64 / union as f64
        }
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
    fn jaccard_distance() {
        let a = vec![1, 0, 1, 1];
        let b = vec![1, 1, 0, 1];

        let j: f64 = Jaccard::new().distance(&a, &b);

        assert!((j - 0.5).abs() < 1e-8);
    }

    #[test]
    fn jaccard_identical_vectors() {
        let a = vec![1, 0, 1, 0];
        let b = vec![1, 0, 1, 0];

        let j: f64 = Jaccard::new().distance(&a, &b);

        assert!((j - 0.0).abs() < 1e-8);
    }

    #[test]
    fn jaccard_both_zero_vectors() {
        let a = vec![0, 0, 0];
        let b = vec![0, 0, 0];

        let j: f64 = Jaccard::new().distance(&a, &b);

        assert!((j - 0.0).abs() < 1e-8);
    }

    #[test]
    fn jaccard_symmetry() {
        let a = vec![1, 0, 1, 1];
        let b = vec![0, 1, 1, 0];

        let j = Jaccard::new();

        let d1 = j.distance(&a, &b);
        let d2 = j.distance(&b, &a);

        assert!((d1 - d2).abs() < 1e-12);
    }
}
