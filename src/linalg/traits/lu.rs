//! # LU Decomposition
//!
//! Decomposes a square matrix into a product of two triangular matrices:
//!
//! \\[A = LU\\]
//!
//! where \\(U\\) is an upper triangular matrix and \\(L\\) is a lower triangular matrix.
//! and \\(Q{-1}\\) is the inverse of the matrix comprised of the eigenvectors. The LU decomposition is used to obtain more efficient solutions to equations of the form
//!
//! \\[Ax = b\\]
//!
//! Example:
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::linalg::traits::lu::*;
//!
//! let A = DenseMatrix::from_2d_array(&[
//!                  &[1., 2., 3.],
//!                  &[0., 1., 5.],
//!                  &[5., 6., 0.]
//!         ]);
//!
//! let lu = A.lu().unwrap();
//! let lower: DenseMatrix<f64> = lu.L();
//! let upper: DenseMatrix<f64> = lu.U();
//! ```
//!
//! ## References:
//! * ["No bullshit guide to linear algebra", Ivan Savov, 2016, 7.6 Matrix decompositions](https://minireference.com/)
//! * ["Numerical Recipes: The Art of Scientific Computing",  Press W.H., Teukolsky S.A., Vetterling W.T, Flannery B.P, 3rd ed., 2.3.1 Performing the LU Decomposition](http://numerical.recipes/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#![allow(non_snake_case)]

use std::cmp::Ordering;
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::error::Failed;
use crate::linalg::basic::arrays::Array2;
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;
#[derive(Debug, Clone)]
/// Result of LU decomposition.
pub struct LU<T: Number + RealNumber, M: Array2<T>> {
    LU: M,
    pivot: Vec<usize>,
    #[allow(dead_code)]
    pivot_sign: i8,
    singular: bool,
    phantom: PhantomData<T>,
}

impl<T: Number + RealNumber, M: Array2<T>> LU<T, M> {
    pub(crate) fn new(LU: M, pivot: Vec<usize>, pivot_sign: i8) -> LU<T, M> {
        let (_, n) = LU.shape();

        let mut singular = false;
        for j in 0..n {
            if LU.get((j, j)) == &T::zero() {
                singular = true;
                break;
            }
        }

        LU {
            LU,
            pivot,
            pivot_sign,
            singular,
            phantom: PhantomData,
        }
    }

    /// Get lower triangular matrix
    pub fn L(&self) -> M {
        let (n_rows, n_cols) = self.LU.shape();
        let mut L = M::zeros(n_rows, n_cols);

        for i in 0..n_rows {
            for j in 0..n_cols {
                match i.cmp(&j) {
                    Ordering::Greater => L.set((i, j), *self.LU.get((i, j))),
                    Ordering::Equal => L.set((i, j), T::one()),
                    Ordering::Less => L.set((i, j), T::zero()),
                }
            }
        }

        L
    }

    /// Get upper triangular matrix
    pub fn U(&self) -> M {
        let (n_rows, n_cols) = self.LU.shape();
        let mut U = M::zeros(n_rows, n_cols);

        for i in 0..n_rows {
            for j in 0..n_cols {
                if i <= j {
                    U.set((i, j), *self.LU.get((i, j)));
                } else {
                    U.set((i, j), T::zero());
                }
            }
        }

        U
    }

    /// Pivot vector
    pub fn pivot(&self) -> M {
        let (_, n) = self.LU.shape();
        let mut piv = M::zeros(n, n);

        for i in 0..n {
            piv.set((i, self.pivot[i]), T::one());
        }

        piv
    }

    /// Returns matrix inverse
    pub fn inverse(&self) -> Result<M, Failed> {
        let (m, n) = self.LU.shape();

        if m != n {
            panic!("Matrix is not square: {}x{}", m, n);
        }

        let mut inv = M::zeros(n, n);

        for i in 0..n {
            inv.set((i, i), T::one());
        }

        self.solve(inv)
    }

    fn solve(&self, mut b: M) -> Result<M, Failed> {
        let (m, n) = self.LU.shape();
        let (b_m, b_n) = b.shape();

        if b_m != m {
            panic!(
                "Row dimensions do not agree: A is {} x {}, but B is {} x {}",
                m, n, b_m, b_n
            );
        }

        if self.singular {
            panic!("Matrix is singular.");
        }

        let mut X = M::zeros(b_m, b_n);

        for j in 0..b_n {
            for i in 0..m {
                X.set((i, j), *b.get((self.pivot[i], j)));
            }
        }

        for k in 0..n {
            for i in k + 1..n {
                for j in 0..b_n {
                    X.sub_element_mut((i, j), *X.get((k, j)) * *self.LU.get((i, k)));
                }
            }
        }

        for k in (0..n).rev() {
            for j in 0..b_n {
                X.div_element_mut((k, j), *self.LU.get((k, k)));
            }

            for i in 0..k {
                for j in 0..b_n {
                    X.sub_element_mut((i, j), *X.get((k, j)) * *self.LU.get((i, k)));
                }
            }
        }

        for j in 0..b_n {
            for i in 0..m {
                b.set((i, j), *X.get((i, j)));
            }
        }

        Ok(b)
    }
}

/// Trait that implements LU decomposition routine for any matrix.
pub trait LUDecomposable<T: Number + RealNumber>: Array2<T> {
    /// Compute the LU decomposition of a square matrix.
    fn lu(&self) -> Result<LU<T, Self>, Failed> {
        self.clone().lu_mut()
    }

    /// Compute the LU decomposition of a square matrix. The input matrix
    /// will be used for factorization.
    fn lu_mut(mut self) -> Result<LU<T, Self>, Failed> {
        let (m, n) = self.shape();

        let mut piv = (0..m).collect::<Vec<_>>();

        let mut pivsign = 1;
        let mut LUcolj = vec![T::zero(); m];

        for j in 0..n {
            for (i, LUcolj_i) in LUcolj.iter_mut().enumerate().take(m) {
                *LUcolj_i = *self.get((i, j));
            }

            for i in 0..m {
                let kmax = usize::min(i, j);
                let mut s = T::zero();
                for (k, LUcolj_k) in LUcolj.iter().enumerate().take(kmax) {
                    s += *self.get((i, k)) * (*LUcolj_k);
                }

                LUcolj[i] -= s;
                self.set((i, j), LUcolj[i]);
            }

            let mut p = j;
            for i in j + 1..m {
                if LUcolj[i].abs() > LUcolj[p].abs() {
                    p = i;
                }
            }
            if p != j {
                for k in 0..n {
                    self.swap((p, k), (j, k));
                }
                piv.swap(p, j);
                pivsign = -pivsign;
            }

            if j < m && self.get((j, j)) != &T::zero() {
                for i in j + 1..m {
                    self.div_element_mut((i, j), *self.get((j, j)));
                }
            }
        }

        Ok(LU::new(self, piv, pivsign))
    }

    /// Solves Ax = b
    fn lu_solve_mut(self, b: Self) -> Result<Self, Failed> {
        self.lu_mut().and_then(|lu| lu.solve(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use approx::relative_eq;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn decompose() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[0., 1., 5.], &[5., 6., 0.]]);
        let expected_L =
            DenseMatrix::from_2d_array(&[&[1., 0., 0.], &[0., 1., 0.], &[0.2, 0.8, 1.]]);
        let expected_U =
            DenseMatrix::from_2d_array(&[&[5., 6., 0.], &[0., 1., 5.], &[0., 0., -1.]]);
        let expected_pivot =
            DenseMatrix::from_2d_array(&[&[0., 0., 1.], &[0., 1., 0.], &[1., 0., 0.]]);
        let lu = a.lu().unwrap();
        assert!(relative_eq!(lu.L(), expected_L, epsilon = 1e-4));
        assert!(relative_eq!(lu.U(), expected_U, epsilon = 1e-4));
        assert!(relative_eq!(lu.pivot(), expected_pivot, epsilon = 1e-4));
    }
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn inverse() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[0., 1., 5.], &[5., 6., 0.]]);
        let expected =
            DenseMatrix::from_2d_array(&[&[-6.0, 3.6, 1.4], &[5.0, -3.0, -1.0], &[-1.0, 0.8, 0.2]]);
        let a_inv = a.lu().and_then(|lu| lu.inverse()).unwrap();
        assert!(relative_eq!(a_inv, expected, epsilon = 1e-4));
    }
}
