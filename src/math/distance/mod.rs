//! # Collection of Distance Functions
//!
//! Many algorithms in machine learning require a measure of distance between data points. Distance metric (or metric) is a function that defines a distance between a pair of point elements of a set.
//! Formally, the distance can be any metric measure that is defined as \\( d(x, y) \geq 0\\) and follows three conditions:
//! 1. \\( d(x, y) = 0 \\) if and only \\( x = y \\), positive definiteness
//! 1. \\( d(x, y) = d(y, x) \\), symmetry
//! 1. \\( d(x, y) \leq d(x, z) + d(z, y) \\), 	subadditivity or triangle inequality
//!
//! for all \\(x, y, z \in Z \\)
//!
//! A good distance metric helps to improve the performance of classification, clustering and information retrieval algorithms significantly.
//!
//! <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

/// Euclidean Distance is the straight-line distance between two points in Euclidean spacere that presents the shortest distance between these points.
pub mod euclidian;
/// Hamming Distance between two strings is the number of positions at which the corresponding symbols are different.
pub mod hamming;
/// The Mahalanobis distance is the distance between two points in multivariate space.
pub mod mahalanobis;
/// Also known as rectilinear distance, city block distance, taxicab metric.
pub mod manhattan;
/// A generalization of both the Euclidean distance and the Manhattan distance.
pub mod minkowski;

use crate::math::num::RealNumber;

/// Distance metric, a function that calculates distance between two points
pub trait Distance<T, F: RealNumber> {
    /// Calculates distance between _a_ and _b_
    fn distance(&self, a: &T, b: &T) -> F;
}

/// Multitude of distance metric functions
pub struct Distances {}

impl Distances {
    /// Euclidian distance
    pub fn euclidian() -> euclidian::Euclidian {
        euclidian::Euclidian {}
    }

    /// Minkowski distance
    /// * `p` - function order. Should be >= 1
    pub fn minkowski(p: u16) -> minkowski::Minkowski {
        minkowski::Minkowski { p: p }
    }

    /// Manhattan distance
    pub fn manhattan() -> manhattan::Manhattan {
        manhattan::Manhattan {}
    }

    /// Hamming distance
    pub fn hamming() -> hamming::Hamming {
        hamming::Hamming {}
    }
}
