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
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::linalg::base::ArrayView2;
//! use smartcore::metrics::distance::Distance;
//! use smartcore::metrics::distance::mahalanobis::Mahalanobis;
//!
//! let data = DenseMatrix::from_2d_array(&[
//!                   &[64., 580., 29.],
//!                   &[66., 570., 33.],
//!                   &[68., 590., 37.],
//!                   &[69., 660., 46.],
//!                   &[73., 600., 55.],
//! ]);
//!
//! let a = data.mean(0);
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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

use super::Distance;
use crate::linalg::basic::arrays::{Array, Array2, ArrayView1};
use crate::linalg::basic::matrix::DenseMatrix;
use crate::linalg::traits::lu::LUDecomposable;
use crate::numbers::basenum::Number;

/// Mahalanobis distance.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Mahalanobis<T: Number, M: Array2<f64>> {
    /// covariance matrix of the dataset
    pub sigma: M,
    /// inverse of the covariance matrix
    pub sigmaInv: M,
    _t: PhantomData<T>,
}

impl<T: Number, M: Array2<f64> + LUDecomposable<f64>> Mahalanobis<T, M> {
    /// Constructs new instance of `Mahalanobis` from given dataset
    /// * `data` - a matrix of _NxM_ where _N_ is number of observations and _M_ is number of attributes
    pub fn new<X: Array2<T>>(data: &X) -> Mahalanobis<T, M> {
        let (_, m) = data.shape();
        let mut sigma = M::zeros(m, m);
        data.cov(&mut sigma);
        let sigmaInv = sigma.lu().and_then(|lu| lu.inverse()).unwrap();
        Mahalanobis {
            sigma,
            sigmaInv,
            _t: PhantomData,
        }
    }

    /// Constructs new instance of `Mahalanobis` from given covariance matrix
    /// * `cov` - a covariance matrix
    pub fn new_from_covariance<X: Array2<f64> + LUDecomposable<f64>>(cov: &X) -> Mahalanobis<T, X> {
        let sigma = cov.clone();
        let sigmaInv = sigma.lu().and_then(|lu| lu.inverse()).unwrap();
        Mahalanobis {
            sigma,
            sigmaInv,
            _t: PhantomData,
        }
    }
}

impl<T: Number, A: ArrayView1<T>> Distance<A> for Mahalanobis<T, DenseMatrix<f64>> {
    fn distance(&self, x: &A, y: &A) -> f64 {
        let (nrows, ncols) = self.sigma.shape();
        if x.shape() != nrows {
            panic!(
                "Array x[{}] has different dimension with Sigma[{}][{}].",
                x.shape(),
                nrows,
                ncols
            );
        }

        if y.shape() != nrows {
            panic!(
                "Array y[{}] has different dimension with Sigma[{}][{}].",
                y.shape(),
                nrows,
                ncols
            );
        }

        let n = x.shape();

        let z: Vec<f64> = x
            .iterator(0)
            .zip(y.iterator(0))
            .map(|(&a, &b)| (a - b).to_f64().unwrap())
            .collect();

        // np.dot(np.dot((a-b),VI),(a-b).T)
        let mut s = 0f64;
        for j in 0..n {
            for i in 0..n {
                s += *self.sigmaInv.get((i, j)) * z[i] * z[j];
            }
        }

        s.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::ArrayView2;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mahalanobis_distance() {
        let data = DenseMatrix::from_2d_array(&[
            &[64., 580., 29.],
            &[66., 570., 33.],
            &[68., 590., 37.],
            &[69., 660., 46.],
            &[73., 600., 55.],
        ]);

        let a = data.mean(0);
        let b = vec![66., 640., 44.];

        let mahalanobis = Mahalanobis::new(&data);

        let md: f64 = mahalanobis.distance(&a, &b);

        assert!((md - 5.33).abs() < 1e-2);
    }
}
