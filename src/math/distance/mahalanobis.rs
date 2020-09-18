//! # Mahalanobis Distance
//!
//! The Mahalanobis distance (MD) is the distance between two points in multivariate space.
//! In a regular Euclidean space the distance between any two points can be measured with [Euclidean distance](../euclidian/index.html).
//! For uncorrelated variables, the Euclidean distance equals the MD. However, if two or more variables are correlated the measurements become impossible
//! with Euclidean distance because the axes are no longer at right angles to each other. MD on the other hand, is scale-invariant,
//! it takes into account the covariance matrix of the dataset when calculating distance between 2 points that belong to the same space as the dataset.
//!
//! MD between two vectors \\( x \in ℝ^n \\) and \\( y \in ℝ^n \\) is defined as
//! \\[ d(x, y) = \sqrt{(x - y)^TS^{-1}(x - y)}\\]
//!
//! where \\( S \\) is the covariance matrix of the dataset.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::math::distance::Distance;
//! use smartcore::math::distance::mahalanobis::Mahalanobis;
//!
//! let data = DenseMatrix::from_2d_array(&[
//!                   &[64., 580., 29.],
//!                   &[66., 570., 33.],
//!                   &[68., 590., 37.],
//!                   &[69., 660., 46.],
//!                   &[73., 600., 55.],
//! ]);
//!
//! let a = data.column_mean();
//! let b = vec![66., 640., 44.];
//!
//! let mahalanobis = Mahalanobis::new(&data);
//!
//! mahalanobis.distance(&a, &b);
//! ```
//!
//! ## References
//! * ["Introduction to Multivariate Statistical Analysis in Chemometrics", Varmuza, K., Filzmoser, P., 2016, p.46](https://www.taylorfrancis.com/books/9780429145049)
//! * ["Example of Calculating the Mahalanobis Distance", McCaffrey, J.D.](https://jamesmccaffrey.wordpress.com/2017/11/09/example-of-calculating-the-mahalanobis-distance/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#![allow(non_snake_case)]

use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::math::num::RealNumber;

use super::Distance;
use crate::linalg::Matrix;

/// Mahalanobis distance.
#[derive(Serialize, Deserialize, Debug)]
pub struct Mahalanobis<T: RealNumber, M: Matrix<T>> {
    /// covariance matrix of the dataset
    pub sigma: M,
    /// inverse of the covariance matrix
    pub sigmaInv: M,
    t: PhantomData<T>,
}

impl<T: RealNumber, M: Matrix<T>> Mahalanobis<T, M> {
    /// Constructs new instance of `Mahalanobis` from given dataset
    /// * `data` - a matrix of _NxM_ where _N_ is number of observations and _M_ is number of attributes
    pub fn new(data: &M) -> Mahalanobis<T, M> {
        let sigma = data.cov();
        let sigmaInv = sigma.lu().and_then(|lu| lu.inverse()).unwrap();
        Mahalanobis {
            sigma: sigma,
            sigmaInv: sigmaInv,
            t: PhantomData,
        }
    }

    /// Constructs new instance of `Mahalanobis` from given covariance matrix
    /// * `cov` - a covariance matrix
    pub fn new_from_covariance(cov: &M) -> Mahalanobis<T, M> {
        let sigma = cov.clone();
        let sigmaInv = sigma.lu().and_then(|lu| lu.inverse()).unwrap();
        Mahalanobis {
            sigma: sigma,
            sigmaInv: sigmaInv,
            t: PhantomData,
        }
    }
}

impl<T: RealNumber, M: Matrix<T>> Distance<Vec<T>, T> for Mahalanobis<T, M> {
    fn distance(&self, x: &Vec<T>, y: &Vec<T>) -> T {
        let (nrows, ncols) = self.sigma.shape();
        if x.len() != nrows {
            panic!(
                "Array x[{}] has different dimension with Sigma[{}][{}].",
                x.len(),
                nrows,
                ncols
            );
        }

        if y.len() != nrows {
            panic!(
                "Array y[{}] has different dimension with Sigma[{}][{}].",
                y.len(),
                nrows,
                ncols
            );
        }

        println!("{}", self.sigmaInv);

        let n = x.len();
        let mut z = vec![T::zero(); n];
        for i in 0..n {
            z[i] = x[i] - y[i];
        }

        // np.dot(np.dot((a-b),VI),(a-b).T)
        let mut s = T::zero();
        for j in 0..n {
            for i in 0..n {
                s = s + self.sigmaInv.get(i, j) * z[i] * z[j];
            }
        }

        s.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn mahalanobis_distance() {
        let data = DenseMatrix::from_2d_array(&[
            &[64., 580., 29.],
            &[66., 570., 33.],
            &[68., 590., 37.],
            &[69., 660., 46.],
            &[73., 600., 55.],
        ]);

        let a = data.column_mean();
        let b = vec![66., 640., 44.];

        let mahalanobis = Mahalanobis::new(&data);

        let md: f64 = mahalanobis.distance(&a, &b);

        assert!((md - 5.33).abs() < 1e-2);
    }
}
