//! # Cosine Distance Metric
//!
//! The cosine distance between two points \\( x \\) and \\( y \\) in n-space is defined as:
//!
//! \\[ d(x, y) = 1 - \frac{x \cdot y}{||x|| ||y||} \\]
//!
//! where \\( x \cdot y \\) is the dot product of the vectors, and \\( ||x|| \\) and \\( ||y|| \\)
//! are their respective magnitudes (Euclidean norms).
//!
//! Cosine distance measures the angular dissimilarity between vectors, ranging from 0 to 2.
//! A value of 0 indicates identical direction (parallel vectors), while larger values indicate
//! greater angular separation.
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::distance::Distance;
//! use smartcore::metrics::distance::cosine::Cosine;
//!
//! let x = vec![1., 1.];
//! let y = vec![2., 2.];
//!
//! let cosine_dist: f64 = Cosine::new().distance(&x, &y);
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

/// Cosine distance is a measure of the angular dissimilarity between two non-zero vectors in n-space.
/// It is defined as 1 minus the cosine similarity of the vectors.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Cosine<T> {
    _t: PhantomData<T>,
}

impl<T: Number> Default for Cosine<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Number> Cosine<T> {
    /// Instantiate the initial structure
    pub fn new() -> Cosine<T> {
        Cosine { _t: PhantomData }
    }

    /// Calculate the dot product of two vectors using smartcore's ArrayView1 trait
    #[inline]
    pub(crate) fn dot_product<A: ArrayView1<T>>(x: &A, y: &A) -> f64 {
        if x.shape() != y.shape() {
            panic!("Input vector sizes are different.");
        }

        // Use the built-in dot product method from ArrayView1 trait
        x.dot(y).to_f64().unwrap()
    }

    /// Calculate the squared magnitude (norm squared) of a vector
    #[inline]
    #[expect(dead_code)]
    pub(crate) fn squared_magnitude<A: ArrayView1<T>>(x: &A) -> f64 {
        x.iterator(0)
            .map(|&a| {
                let val = a.to_f64().unwrap();
                val * val
            })
            .sum()
    }

    /// Calculate the magnitude (Euclidean norm) of a vector using smartcore's norm2 method
    #[inline]
    pub(crate) fn magnitude<A: ArrayView1<T>>(x: &A) -> f64 {
        // Use the built-in norm2 method from ArrayView1 trait
        x.norm2()
    }

    /// Calculate cosine similarity between two vectors
    #[inline]
    pub(crate) fn cosine_similarity<A: ArrayView1<T>>(x: &A, y: &A) -> f64 {
        let dot_product = Self::dot_product(x, y);
        let magnitude_x = Self::magnitude(x);
        let magnitude_y = Self::magnitude(y);

        if magnitude_x == 0.0 || magnitude_y == 0.0 {
            return f64::MIN;
        }

        dot_product / (magnitude_x * magnitude_y)
    }
}

impl<T: Number, A: ArrayView1<T>> Distance<A> for Cosine<T> {
    fn distance(&self, x: &A, y: &A) -> f64 {
        let similarity = Cosine::cosine_similarity(x, y);
        1.0 - similarity
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
    fn cosine_distance_identical_vectors() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];

        let dist: f64 = Cosine::new().distance(&a, &b);

        assert!((dist - 0.0).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_distance_orthogonal_vectors() {
        let a = vec![1, 0];
        let b = vec![0, 1];

        let dist: f64 = Cosine::new().distance(&a, &b);

        assert!((dist - 1.0).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_distance_opposite_vectors() {
        let a = vec![1, 2, 3];
        let b = vec![-1, -2, -3];

        let dist: f64 = Cosine::new().distance(&a, &b);

        assert!((dist - 2.0).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_distance_general_case() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 1.0, 3.0];

        let dist: f64 = Cosine::new().distance(&a, &b);

        // Expected cosine similarity: (1*2 + 2*1 + 3*3) / (sqrt(1+4+9) * sqrt(4+1+9))
        // = (2 + 2 + 9) / (sqrt(14) * sqrt(14)) = 13/14 ≈ 0.9286
        // So cosine distance = 1 - 13/14 = 1/14 ≈ 0.0714
        let expected_dist = 1.0 - (13.0 / 14.0);
        assert!((dist - expected_dist).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[should_panic(expected = "Input vector sizes are different.")]
    fn cosine_distance_different_sizes() {
        let a = vec![1, 2];
        let b = vec![1, 2, 3];

        let _dist: f64 = Cosine::new().distance(&a, &b);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_distance_zero_vector() {
        let a = vec![0, 0, 0];
        let b = vec![1, 2, 3];

        let dist: f64 = Cosine::new().distance(&a, &b);
        assert!(dist > 1e300)
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_distance_float_precision() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];

        let dist: f64 = Cosine::new().distance(&a, &b);

        // Calculate expected value manually
        let dot_product = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // = 32
        let mag_a = (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0_f64).sqrt(); // = sqrt(14)
        let mag_b = (4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0_f64).sqrt(); // = sqrt(77)
        let expected_similarity = dot_product / (mag_a * mag_b);
        let expected_distance = 1.0 - expected_similarity;

        assert!((dist - expected_distance).abs() < 1e-6);
    }
}
