//! This is a generic solver for Ax = b type of equation
//!
//! for more information take a look at [this Wikipedia article](https://en.wikipedia.org/wiki/Biconjugate_gradient_method)
//! and [this paper](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
use crate::error::Failed;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;

pub trait BiconjugateGradientSolver<T: RealNumber, M: Matrix<T>> {
    fn solve_mut(&self, a: &M, b: &M, x: &mut M, tol: T, max_iter: usize) -> Result<T, Failed> {
        if tol <= T::zero() {
            return Err(Failed::fit("tolerance shoud be > 0"));
        }

        if max_iter == 0 {
            return Err(Failed::fit("maximum number of iterations should be > 0"));
        }

        let (n, _) = b.shape();

        let mut r = M::zeros(n, 1);
        let mut rr = M::zeros(n, 1);
        let mut z = M::zeros(n, 1);
        let mut zz = M::zeros(n, 1);

        self.mat_vec_mul(a, x, &mut r);

        for j in 0..n {
            r.set(j, 0, b.get(j, 0) - r.get(j, 0));
            rr.set(j, 0, r.get(j, 0));
        }

        let bnrm = b.norm(T::two());
        self.solve_preconditioner(a, &r, &mut z);

        let mut p = M::zeros(n, 1);
        let mut pp = M::zeros(n, 1);
        let mut bkden = T::zero();
        let mut err = T::zero();

        for iter in 1..max_iter {
            let mut bknum = T::zero();

            self.solve_preconditioner(a, &rr, &mut zz);
            for j in 0..n {
                bknum += z.get(j, 0) * rr.get(j, 0);
            }
            if iter == 1 {
                for j in 0..n {
                    p.set(j, 0, z.get(j, 0));
                    pp.set(j, 0, zz.get(j, 0));
                }
            } else {
                let bk = bknum / bkden;
                for j in 0..n {
                    p.set(j, 0, bk * p.get(j, 0) + z.get(j, 0));
                    pp.set(j, 0, bk * pp.get(j, 0) + zz.get(j, 0));
                }
            }
            bkden = bknum;
            self.mat_vec_mul(a, &p, &mut z);
            let mut akden = T::zero();
            for j in 0..n {
                akden += z.get(j, 0) * pp.get(j, 0);
            }
            let ak = bknum / akden;
            self.mat_t_vec_mul(a, &pp, &mut zz);
            for j in 0..n {
                x.set(j, 0, x.get(j, 0) + ak * p.get(j, 0));
                r.set(j, 0, r.get(j, 0) - ak * z.get(j, 0));
                rr.set(j, 0, rr.get(j, 0) - ak * zz.get(j, 0));
            }
            self.solve_preconditioner(a, &r, &mut z);
            err = r.norm(T::two()) / bnrm;

            if err <= tol {
                break;
            }
        }

        Ok(err)
    }

    fn solve_preconditioner(&self, a: &M, b: &M, x: &mut M) {
        let diag = Self::diag(a);
        let n = diag.len();

        for (i, diag_i) in diag.iter().enumerate().take(n) {
            if *diag_i != T::zero() {
                x.set(i, 0, b.get(i, 0) / *diag_i);
            } else {
                x.set(i, 0, b.get(i, 0));
            }
        }
    }

    // y = Ax
    fn mat_vec_mul(&self, a: &M, x: &M, y: &mut M) {
        y.copy_from(&a.matmul(x));
    }

    // y = Atx
    fn mat_t_vec_mul(&self, a: &M, x: &M, y: &mut M) {
        y.copy_from(&a.ab(true, x, false));
    }

    fn diag(a: &M) -> Vec<T> {
        let (nrows, ncols) = a.shape();
        let n = nrows.min(ncols);

        let mut d = Vec::with_capacity(n);
        for i in 0..n {
            d.push(a.get(i, i));
        }

        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    pub struct BGSolver {}

    impl<T: RealNumber, M: Matrix<T>> BiconjugateGradientSolver<T, M> for BGSolver {}

    #[test]
    fn bg_solver() {
        let a = DenseMatrix::from_2d_array(&[&[25., 15., -5.], &[15., 18., 0.], &[-5., 0., 11.]]);
        let b = DenseMatrix::from_2d_array(&[&[40., 51., 28.]]);
        let expected = DenseMatrix::from_2d_array(&[&[1.0, 2.0, 3.0]]);

        let mut x = DenseMatrix::zeros(3, 1);

        let solver = BGSolver {};

        let err: f64 = solver
            .solve_mut(&a, &b.transpose(), &mut x, 1e-6, 6)
            .unwrap();

        assert!(x.transpose().approximate_eq(&expected, 1e-4));
        assert!((err - 0.0).abs() < 1e-4);
    }
}
