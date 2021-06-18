//! # Cholesky Decomposition
//!
//! every positive definite matrix \\(A \in R^{n \times n}\\) can be factored as
//!
//! \\[A = R^TR\\]
//!
//! where \\(R\\) is upper triangular matrix with positive diagonal elements
//!
//! Example:
//! ```
//! use smartcore::linalg::dense::matrix::DenseMatrix;
//! use crate::smartcore::linalg::cholesky_n::*;
//!
//! let A = DenseMatrix::from_2d_array(&[
//!                 &[25., 15., -5.],
//!                 &[15., 18., 0.],
//!                 &[-5., 0., 11.]
//!         ]);
//!
//! let cholesky = A.cholesky().unwrap();
//! let lower_triangular: DenseMatrix<f64> = cholesky.L();
//! let upper_triangular: DenseMatrix<f64> = cholesky.U();
//! ```
//!
//! ## References:
//! * ["No bullshit guide to linear algebra", Ivan Savov, 2016, 7.6 Matrix decompositions](https://minireference.com/)
//! * ["Numerical Recipes: The Art of Scientific Computing",  Press W.H., Teukolsky S.A., Vetterling W.T, Flannery B.P, 3rd ed., 2.9 Cholesky Decomposition](http://numerical.recipes/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#![allow(non_snake_case)]

use std::fmt::Debug;
use std::marker::PhantomData;

use crate::error::{Failed, FailedError};
use crate::linalg::base::Array2;
use crate::num::FloatNumber;

#[derive(Debug, Clone)]
/// Results of Cholesky decomposition.
pub struct Cholesky<T: FloatNumber, M: Array2<T>> {
    R: M,
    t: PhantomData<T>,
}

impl<T: FloatNumber, M: Array2<T>> Cholesky<T, M> {
    pub(crate) fn new(R: M) -> Cholesky<T, M> {
        Cholesky { R, t: PhantomData }
    }

    /// Get lower triangular matrix.
    pub fn L(&self) -> M {
        let (n, _) = self.R.shape();
        let mut R = M::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                if j <= i {
                    R.set((i, j), *self.R.get((i, j)));
                }
            }
        }
        R
    }

    /// Get upper triangular matrix.
    pub fn U(&self) -> M {
        let (n, _) = self.R.shape();
        let mut R = M::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                if j <= i {
                    R.set((j, i), *self.R.get((i, j)));
                }
            }
        }
        R
    }

    /// Solves Ax = b
    pub(crate) fn solve(&self, mut b: M) -> Result<M, Failed> {
        let (bn, m) = b.shape();
        let (rn, _) = self.R.shape();

        if bn != rn {
            return Err(Failed::because(
                FailedError::SolutionFailed,
                &"Can\'t solve Ax = b for x. Number of rows in b != number of rows in R."
                    .to_string(),
            ));
        }

        for k in 0..bn {
            for j in 0..m {
                for i in 0..k {
                    b.sub_element_mut((k, j), *b.get((i, j)) * *self.R.get((k, i)));
                }
                b.div_element_mut((k, j), *self.R.get((k, k)));
            }
        }

        for k in (0..bn).rev() {
            for j in 0..m {
                for i in k + 1..bn {
                    b.sub_element_mut((k, j), *b.get((i, j)) * *self.R.get((i, k)));
                }
                b.div_element_mut((k, j), *self.R.get((k, k)));
            }
        }
        Ok(b)
    }
}

/// Trait that implements Cholesky decomposition routine for any matrix.
pub trait CholeskyDecomposableMatrix<T: FloatNumber>: Array2<T> {
    /// Compute the Cholesky decomposition of a matrix.
    fn cholesky(&self) -> Result<Cholesky<T, Self>, Failed> {
        self.clone().cholesky_mut()
    }

    /// Compute the Cholesky decomposition of a matrix. The input matrix
    /// will be used for factorization.
    fn cholesky_mut(mut self) -> Result<Cholesky<T, Self>, Failed> {
        let (m, n) = self.shape();

        if m != n {
            return Err(Failed::because(
                FailedError::DecompositionFailed,
                &"Can\'t do Cholesky decomposition on a non-square matrix".to_string(),
            ));
        }

        for j in 0..n {
            let mut d = T::zero();
            for k in 0..j {
                let mut s = T::zero();
                for i in 0..k {
                    s += *self.get((k, i)) * *self.get((j, i));
                }
                s = (*self.get((j, k)) - s) / *self.get((k, k));
                self.set((j, k), s);
                d += s * s;
            }
            d = *self.get((j, j)) - d;

            if d < T::zero() {
                return Err(Failed::because(
                    FailedError::DecompositionFailed,
                    &"The matrix is not positive definite.".to_string(),
                ));
            }

            self.set((j, j), d.sqrt());
        }

        Ok(Cholesky::new(self))
    }

    /// Solves Ax = b
    fn cholesky_solve_mut(self, b: Self) -> Result<Self, Failed> {
        self.cholesky_mut().and_then(|qr| qr.solve(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::dense::matrix::DenseMatrix;
    use approx::relative_eq;
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cholesky_decompose() {
        let a = DenseMatrix::from_2d_array(&[&[25., 15., -5.], &[15., 18., 0.], &[-5., 0., 11.]]);
        let l =
            DenseMatrix::from_2d_array(&[&[5.0, 0.0, 0.0], &[3.0, 3.0, 0.0], &[-1.0, 1.0, 3.0]]);
        let u =
            DenseMatrix::from_2d_array(&[&[5.0, 3.0, -1.0], &[0.0, 3.0, 1.0], &[0.0, 0.0, 3.0]]);
        let cholesky = a.cholesky().unwrap();

        assert!(relative_eq!(cholesky.L().abs(), l.abs(), epsilon = 1e-4));
        assert!(relative_eq!(cholesky.U().abs(), u.abs(), epsilon = 1e-4));
        assert!(relative_eq!(
            cholesky.L().matmul(&cholesky.U()).abs(),
            a.abs(),
            epsilon = 1e-4
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cholesky_solve_mut() {
        let a = DenseMatrix::from_2d_array(&[&[25., 15., -5.], &[15., 18., 0.], &[-5., 0., 11.]]);
        let b = DenseMatrix::from_2d_array(&[&[40., 51., 28.]]);
        let expected = DenseMatrix::from_2d_array(&[&[1.0, 2.0, 3.0]]);

        let cholesky = a.cholesky().unwrap();

        assert!(relative_eq!(
            cholesky.solve(b.transpose()).unwrap().transpose(),
            expected,
            epsilon = 1e-4
        ));
    }
}
