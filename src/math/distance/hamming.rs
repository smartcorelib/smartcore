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
//! let h: f64 = Hamming {}.distance(&a, &b);
//!
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::math::num::RealNumber;

use super::Distance;

/// While comparing two integer-valued vectors of equal length, Hamming distance is the number of bit positions in which the two bits are different
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Hamming {}

impl<T: PartialEq, F: RealNumber> Distance<Vec<T>, F> for Hamming {
    fn distance(&self, x: &Vec<T>, y: &Vec<T>) -> F {
        if x.len() != y.len() {
            panic!("Input vector sizes are different");
        }

        let mut dist = 0;
        for i in 0..x.len() {
            if x[i] != y[i] {
                dist += 1;
            }
        }

        F::from_i64(dist).unwrap() / F::from_usize(x.len()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_distance() {
        let a = vec![1, 0, 0, 1, 0, 0, 1];
        let b = vec![1, 1, 0, 0, 1, 0, 1];

        let h: f64 = Hamming {}.distance(&a, &b);

        assert!((h - 0.42857142).abs() < 1e-8);
    }
}
