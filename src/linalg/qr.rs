//! # QR Decomposition
//!
//! Any real square matrix \\(A \in R^{n \times n}\\) can be decomposed as a product of an orthogonal matrix \\(Q\\) and an upper triangular matrix \\(R\\):
//!
//! \\[A = QR\\]
//!
//! Example:
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::linalg::qr::*;
//!
//! let A = DenseMatrix::from_2d_array(&[
//!                 &[0.9, 0.4, 0.7],
//!                 &[0.4, 0.5, 0.3],
//!                 &[0.7, 0.3, 0.8]
//!         ]);
//!
//! let qr = A.qr().unwrap();
//! let orthogonal: DenseMatrix<f64> = qr.Q();
//! let triangular: DenseMatrix<f64> = qr.R();
//! ```
//!
//! ## References:
//! * ["No bullshit guide to linear algebra", Ivan Savov, 2016, 7.6 Matrix decompositions](https://minireference.com/)
//! * ["Numerical Recipes: The Art of Scientific Computing",  Press W.H., Teukolsky S.A., Vetterling W.T, Flannery B.P, 3rd ed., 2.10 QR Decomposition](http://numerical.recipes/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#![allow(non_snake_case)]

use crate::error::Failed;
use crate::linalg::BaseMatrix;
use crate::math::num::RealNumber;
use std::fmt::Debug;

#[derive(Debug, Clone)]
/// Results of QR decomposition.
pub struct QR<T: RealNumber, M: BaseMatrix<T>> {
    QR: M,
    tau: Vec<T>,
    singular: bool,
}

impl<T: RealNumber, M: BaseMatrix<T>> QR<T, M> {
    pub(crate) fn new(QR: M, tau: Vec<T>) -> QR<T, M> {
        let mut singular = false;
        for tau_elem in tau.iter() {
            if *tau_elem == T::zero() {
                singular = true;
                break;
            }
        }

        QR { QR, tau, singular }
    }

    /// Get upper triangular matrix.
    pub fn R(&self) -> M {
        let (_, n) = self.QR.shape();
        let mut R = M::zeros(n, n);
        for i in 0..n {
            R.set(i, i, self.tau[i]);
            for j in i + 1..n {
                R.set(i, j, self.QR.get(i, j));
            }
        }
        R
    }

    /// Get an orthogonal matrix.
    pub fn Q(&self) -> M {
        let (m, n) = self.QR.shape();
        let mut Q = M::zeros(m, n);
        let mut k = n - 1;
        loop {
            Q.set(k, k, T::one());
            for j in k..n {
                if self.QR.get(k, k) != T::zero() {
                    let mut s = T::zero();
                    for i in k..m {
                        s += self.QR.get(i, k) * Q.get(i, j);
                    }
                    s = -s / self.QR.get(k, k);
                    for i in k..m {
                        Q.add_element_mut(i, j, s * self.QR.get(i, k));
                    }
                }
            }
            if k == 0 {
                break;
            } else {
                k -= 1;
            }
        }
        Q
    }

    fn solve(&self, mut b: M) -> Result<M, Failed> {
        let (m, n) = self.QR.shape();
        let (b_nrows, b_ncols) = b.shape();

        if b_nrows != m {
            panic!(
                "Row dimensions do not agree: A is {} x {}, but B is {} x {}",
                m, n, b_nrows, b_ncols
            );
        }

        if self.singular {
            panic!("Matrix is rank deficient.");
        }

        for k in 0..n {
            for j in 0..b_ncols {
                let mut s = T::zero();
                for i in k..m {
                    s += self.QR.get(i, k) * b.get(i, j);
                }
                s = -s / self.QR.get(k, k);
                for i in k..m {
                    b.add_element_mut(i, j, s * self.QR.get(i, k));
                }
            }
        }

        for k in (0..n).rev() {
            for j in 0..b_ncols {
                b.set(k, j, b.get(k, j) / self.tau[k]);
            }

            for i in 0..k {
                for j in 0..b_ncols {
                    b.sub_element_mut(i, j, b.get(k, j) * self.QR.get(i, k));
                }
            }
        }

        Ok(b)
    }
}

/// Trait that implements QR decomposition routine for any matrix.
pub trait QRDecomposableMatrix<T: RealNumber>: BaseMatrix<T> {
    /// Compute the QR decomposition of a matrix.
    fn qr(&self) -> Result<QR<T, Self>, Failed> {
        self.clone().qr_mut()
    }

    /// Compute the QR decomposition of a matrix. The input matrix
    /// will be used for factorization.
    fn qr_mut(mut self) -> Result<QR<T, Self>, Failed> {
        let (m, n) = self.shape();

        let mut r_diagonal: Vec<T> = vec![T::zero(); n];

        for (k, r_diagonal_k) in r_diagonal.iter_mut().enumerate().take(n) {
            let mut nrm = T::zero();
            for i in k..m {
                nrm = nrm.hypot(self.get(i, k));
            }

            if nrm.abs() > T::epsilon() {
                if self.get(k, k) < T::zero() {
                    nrm = -nrm;
                }
                for i in k..m {
                    self.div_element_mut(i, k, nrm);
                }
                self.add_element_mut(k, k, T::one());

                for j in k + 1..n {
                    let mut s = T::zero();
                    for i in k..m {
                        s += self.get(i, k) * self.get(i, j);
                    }
                    s = -s / self.get(k, k);
                    for i in k..m {
                        self.add_element_mut(i, j, s * self.get(i, k));
                    }
                }
            }
            *r_diagonal_k = -nrm;
        }

        Ok(QR::new(self, r_diagonal))
    }

    /// Solves Ax = b
    fn qr_solve_mut(self, b: Self) -> Result<Self, Failed> {
        self.qr_mut().and_then(|qr| qr.solve(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn decompose() {
        let a = DenseMatrix::from_2d_array(&[&[0.9, 0.4, 0.7], &[0.4, 0.5, 0.3], &[0.7, 0.3, 0.8]]);
        let q = DenseMatrix::from_2d_array(&[
            &[-0.7448, 0.2436, 0.6212],
            &[-0.331, -0.9432, -0.027],
            &[-0.5793, 0.2257, -0.7832],
        ]);
        let r = DenseMatrix::from_2d_array(&[
            &[-1.2083, -0.6373, -1.0842],
            &[0.0, -0.3064, 0.0682],
            &[0.0, 0.0, -0.1999],
        ]);
        let qr = a.qr().unwrap();
        assert!(qr.Q().abs().approximate_eq(&q.abs(), 1e-4));
        assert!(qr.R().abs().approximate_eq(&r.abs(), 1e-4));
    }

    #[test]
    fn qr_solve_mut() {
        let a = DenseMatrix::from_2d_array(&[&[0.9, 0.4, 0.7], &[0.4, 0.5, 0.3], &[0.7, 0.3, 0.8]]);
        let b = DenseMatrix::from_2d_array(&[&[0.5, 0.2], &[0.5, 0.8], &[0.5, 0.3]]);
        let expected_w = DenseMatrix::from_2d_array(&[
            &[-0.2027027, -1.2837838],
            &[0.8783784, 2.2297297],
            &[0.4729730, 0.6621622],
        ]);
        let w = a.qr_solve_mut(b).unwrap();
        assert!(w.approximate_eq(&expected_w, 1e-2));
    }
}
