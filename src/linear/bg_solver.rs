//! This is a generic solver for Ax = b type of equation
//!
//! Example:
//! ```
//! use smartcore::linalg::basic::arrays::Array1;
//! use smartcore::linalg::basic::arrays::Array2;
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::linear::bg_solver::*;
//! use smartcore::numbers::floatnum::FloatNumber;
//! use smartcore::linear::bg_solver::BiconjugateGradientSolver;
//!
//! pub struct BGSolver {}
//! impl<'a, T: FloatNumber, X: Array2<T>> BiconjugateGradientSolver<'a, T, X> for BGSolver {}
//!
//! let a = DenseMatrix::from_2d_array(&[&[25., 15., -5.], &[15., 18., 0.], &[-5., 0., 11.]]);
//! let b = vec![40., 51., 28.];
//! let expected = vec![1.0, 2.0, 3.0];
//! let mut x = Vec::zeros(3);
//! let solver = BGSolver {};
//! let err: f64 = solver.solve_mut(&a, &b, &mut x, 1e-6, 6).unwrap();
//! ```
//!
//! for more information take a look at [this Wikipedia article](https://en.wikipedia.org/wiki/Biconjugate_gradient_method)
//! and [this paper](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array, Array1, Array2, ArrayView1, MutArrayView1};
use crate::numbers::floatnum::FloatNumber;

///
pub trait BiconjugateGradientSolver<'a, T: FloatNumber, X: Array2<T>> {
    ///
    fn solve_mut(
        &self,
        a: &'a X,
        b: &Vec<T>,
        x: &mut Vec<T>,
        tol: T,
        max_iter: usize,
    ) -> Result<T, Failed> {
        if tol <= T::zero() {
            return Err(Failed::fit("tolerance shoud be > 0"));
        }

        if max_iter == 0 {
            return Err(Failed::fit("maximum number of iterations should be > 0"));
        }

        let n = b.shape();

        let mut r = Vec::zeros(n);
        let mut rr = Vec::zeros(n);
        let mut z = Vec::zeros(n);
        let mut zz = Vec::zeros(n);

        self.mat_vec_mul(a, x, &mut r);

        for j in 0..n {
            r[j] = b[j] - r[j];
            rr[j] = r[j];
        }

        let bnrm = b.norm(2f64);
        self.solve_preconditioner(a, &r[..], &mut z[..]);

        let mut p = Vec::zeros(n);
        let mut pp = Vec::zeros(n);
        let mut bkden = T::zero();
        let mut err = T::zero();

        for iter in 1..max_iter {
            let mut bknum = T::zero();

            self.solve_preconditioner(a, &rr, &mut zz);
            for j in 0..n {
                bknum += z[j] * rr[j];
            }
            if iter == 1 {
                p[..n].copy_from_slice(&z[..n]);
                pp[..n].copy_from_slice(&zz[..n]);
            } else {
                let bk = bknum / bkden;
                for j in 0..n {
                    p[j] = bk * pp[j] + z[j];
                    pp[j] = bk * pp[j] + zz[j];
                }
            }
            bkden = bknum;
            self.mat_vec_mul(a, &p, &mut z);
            let mut akden = T::zero();
            for j in 0..n {
                akden += z[j] * pp[j];
            }
            let ak = bknum / akden;
            self.mat_t_vec_mul(a, &pp, &mut zz);
            for j in 0..n {
                x[j] += ak * p[j];
                r[j] -= ak * z[j];
                rr[j] -= ak * zz[j];
            }
            self.solve_preconditioner(a, &r, &mut z);
            err = T::from_f64(r.norm(2f64) / bnrm).unwrap();

            if err <= tol {
                break;
            }
        }

        Ok(err)
    }

    ///
    fn solve_preconditioner(&self, a: &'a X, b: &[T], x: &mut [T]) {
        let diag = Self::diag(a);
        let n = diag.len();

        for (i, diag_i) in diag.iter().enumerate().take(n) {
            if *diag_i != T::zero() {
                x[i] = b[i] / *diag_i;
            } else {
                x[i] = b[i];
            }
        }
    }

    /// y = Ax
    fn mat_vec_mul(&self, a: &X, x: &Vec<T>, y: &mut Vec<T>) {
        y.copy_from(&x.xa(false, a));
    }

    /// y = Atx
    fn mat_t_vec_mul(&self, a: &X, x: &Vec<T>, y: &mut Vec<T>) {
        y.copy_from(&x.xa(true, a));
    }

    ///
    fn diag(a: &X) -> Vec<T> {
        let (nrows, ncols) = a.shape();
        let n = nrows.min(ncols);

        let mut d = Vec::with_capacity(n);
        for i in 0..n {
            d.push(*a.get((i, i)));
        }

        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::Array2;
    use crate::linalg::basic::matrix::DenseMatrix;

    pub struct BGSolver {}

    impl<T: FloatNumber, X: Array2<T>> BiconjugateGradientSolver<'_, T, X> for BGSolver {}

    #[test]
    fn bg_solver() {
        let a = DenseMatrix::from_2d_array(&[&[25., 15., -5.], &[15., 18., 0.], &[-5., 0., 11.]]);
        let b = vec![40., 51., 28.];
        let expected = vec![1.0, 2.0, 3.0];

        let mut x = Vec::zeros(3);

        let solver = BGSolver {};

        let err: f64 = solver.solve_mut(&a, &b, &mut x, 1e-6, 6).unwrap();

        assert!(x
            .iter()
            .zip(expected.iter())
            .all(|(&a, &b)| (a - b).abs() < 1e-4));
        assert!((err - 0.0).abs() < 1e-4);
    }
}
