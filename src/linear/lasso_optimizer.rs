//! An Interior-Point Method for Large-Scale l1-Regularized Least Squares
//!
//! This is a specialized interior-point method for solving large-scale 1-regularized LSPs that uses the
//! preconditioned conjugate gradients algorithm to compute the search direction.
//!
//! The interior-point method can solve large sparse problems, with a million variables and observations, in a few tens of minutes on a PC.
//! It can efficiently solve large dense problems, that arise in sparse signal recovery with orthogonal transforms, by exploiting fast algorithms for these transforms.
//!
//! ## References:
//! * ["An Interior-Point Method for Large-Scale l1-Regularized Least Squares",  K. Koh, M. Lustig, S. Boyd, D. Gorinevsky](https://web.stanford.edu/~boyd/papers/pdf/l1_ls.pdf)
//! * [Simple Matlab Solver for l1-regularized Least Squares Problems](https://web.stanford.edu/~boyd/l1_ls/)
//!

use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::linear::bg_solver::BiconjugateGradientSolver;
use crate::math::num::RealNumber;

pub struct InteriorPointOptimizer<T: RealNumber, M: Matrix<T>> {
    ata: M,
    d1: Vec<T>,
    d2: Vec<T>,
    prb: Vec<T>,
    prs: Vec<T>,
}

impl<T: RealNumber, M: Matrix<T>> InteriorPointOptimizer<T, M> {
    pub fn new(a: &M, n: usize) -> InteriorPointOptimizer<T, M> {
        InteriorPointOptimizer {
            ata: a.ab(true, a, false),
            d1: vec![T::zero(); n],
            d2: vec![T::zero(); n],
            prb: vec![T::zero(); n],
            prs: vec![T::zero(); n],
        }
    }

    pub fn optimize(
        &mut self,
        x: &M,
        y: &M::RowVector,
        lambda: T,
        max_iter: usize,
        tol: T,
    ) -> Result<M, Failed> {
        let (n, p) = x.shape();
        let p_f64 = T::from_usize(p).unwrap();

        let lambda = lambda.max(T::epsilon());

        //parameters
        let pcgmaxi = 5000;
        let min_pcgtol = T::from_f64(0.1).unwrap();
        let eta = T::from_f64(1E-3).unwrap();
        let alpha = T::from_f64(0.01).unwrap();
        let beta = T::from_f64(0.5).unwrap();
        let gamma = T::from_f64(-0.25).unwrap();
        let mu = T::two();

        let y = M::from_row_vector(y.sub_scalar(y.mean())).transpose();

        let mut max_ls_iter = 100;
        let mut pitr = 0;
        let mut w = M::zeros(p, 1);
        let mut neww = w.clone();
        let mut u = M::ones(p, 1);
        let mut newu = u.clone();

        let mut f = M::fill(p, 2, -T::one());
        let mut newf = f.clone();

        let mut q1 = vec![T::zero(); p];
        let mut q2 = vec![T::zero(); p];

        let mut dx = M::zeros(p, 1);
        let mut du = M::zeros(p, 1);
        let mut dxu = M::zeros(2 * p, 1);
        let mut grad = M::zeros(2 * p, 1);

        let mut nu = M::zeros(n, 1);
        let mut dobj = T::zero();
        let mut s = T::infinity();
        let mut t = T::one()
            .max(T::one() / lambda)
            .min(T::two() * p_f64 / T::from(1e-3).unwrap());

        for ntiter in 0..max_iter {
            let mut z = x.matmul(&w);

            for i in 0..n {
                z.set(i, 0, z.get(i, 0) - y.get(i, 0));
                nu.set(i, 0, T::two() * z.get(i, 0));
            }

            // CALCULATE DUALITY GAP
            let xnu = x.ab(true, &nu, false);
            let max_xnu = xnu.norm(T::infinity());
            if max_xnu > lambda {
                let lnu = lambda / max_xnu;
                nu.mul_scalar_mut(lnu);
            }

            let pobj = z.dot(&z) + lambda * w.norm(T::one());
            dobj = dobj.max(gamma * nu.dot(&nu) - nu.dot(&y));

            let gap = pobj - dobj;

            // STOPPING CRITERION
            if gap / dobj < tol {
                break;
            }

            // UPDATE t
            if s >= T::half() {
                t = t.max((T::two() * p_f64 * mu / gap).min(mu * t));
            }

            // CALCULATE NEWTON STEP
            for i in 0..p {
                let q1i = T::one() / (u.get(i, 0) + w.get(i, 0));
                let q2i = T::one() / (u.get(i, 0) - w.get(i, 0));
                q1[i] = q1i;
                q2[i] = q2i;
                self.d1[i] = (q1i * q1i + q2i * q2i) / t;
                self.d2[i] = (q1i * q1i - q2i * q2i) / t;
            }

            let mut gradphi = x.ab(true, &z, false);

            for i in 0..p {
                let g1 = T::two() * gradphi.get(i, 0) - (q1[i] - q2[i]) / t;
                let g2 = lambda - (q1[i] + q2[i]) / t;
                gradphi.set(i, 0, g1);
                grad.set(i, 0, -g1);
                grad.set(i + p, 0, -g2);
            }

            for i in 0..p {
                self.prb[i] = T::two() + self.d1[i];
                self.prs[i] = self.prb[i] * self.d1[i] - self.d2[i].powi(2);
            }

            let normg = grad.norm2();
            let mut pcgtol = min_pcgtol.min(eta * gap / T::one().min(normg));
            if ntiter != 0 && pitr == 0 {
                pcgtol *= min_pcgtol;
            }

            let error = self.solve_mut(x, &grad, &mut dxu, pcgtol, pcgmaxi)?;
            if error > pcgtol {
                pitr = pcgmaxi;
            }

            for i in 0..p {
                dx.set(i, 0, dxu.get(i, 0));
                du.set(i, 0, dxu.get(i + p, 0));
            }

            // BACKTRACKING LINE SEARCH
            let phi = z.dot(&z) + lambda * u.sum() - Self::sumlogneg(&f) / t;
            s = T::one();
            let gdx = grad.dot(&dxu);

            let lsiter = 0;
            while lsiter < max_ls_iter {
                for i in 0..p {
                    neww.set(i, 0, w.get(i, 0) + s * dx.get(i, 0));
                    newu.set(i, 0, u.get(i, 0) + s * du.get(i, 0));
                    newf.set(i, 0, neww.get(i, 0) - newu.get(i, 0));
                    newf.set(i, 1, -neww.get(i, 0) - newu.get(i, 0));
                }

                if newf.max() < T::zero() {
                    let mut newz = x.matmul(&neww);
                    for i in 0..n {
                        newz.set(i, 0, newz.get(i, 0) - y.get(i, 0));
                    }

                    let newphi = newz.dot(&newz) + lambda * newu.sum() - Self::sumlogneg(&newf) / t;
                    if newphi - phi <= alpha * s * gdx {
                        break;
                    }
                }
                s = beta * s;
                max_ls_iter += 1;
            }

            if lsiter == max_ls_iter {
                return Err(Failed::fit(
                    "Exceeded maximum number of iteration for interior point optimizer",
                ));
            }

            w.copy_from(&neww);
            u.copy_from(&newu);
            f.copy_from(&newf);
        }

        Ok(w)
    }

    fn sumlogneg(f: &M) -> T {
        let (n, _) = f.shape();
        let mut sum = T::zero();
        for i in 0..n {
            sum += (-f.get(i, 0)).ln();
            sum += (-f.get(i, 1)).ln();
        }
        sum
    }
}

impl<'a, T: RealNumber, M: Matrix<T>> BiconjugateGradientSolver<T, M>
    for InteriorPointOptimizer<T, M>
{
    fn solve_preconditioner(&self, a: &M, b: &M, x: &mut M) {
        let (_, p) = a.shape();

        for i in 0..p {
            x.set(
                i,
                0,
                (self.d1[i] * b.get(i, 0) - self.d2[i] * b.get(i + p, 0)) / self.prs[i],
            );
            x.set(
                i + p,
                0,
                (-self.d2[i] * b.get(i, 0) + self.prb[i] * b.get(i + p, 0)) / self.prs[i],
            );
        }
    }

    fn mat_vec_mul(&self, _: &M, x: &M, y: &mut M) {
        let (_, p) = self.ata.shape();
        let atax = self.ata.matmul(&x.slice(0..p, 0..1));

        for i in 0..p {
            y.set(
                i,
                0,
                T::two() * atax.get(i, 0) + self.d1[i] * x.get(i, 0) + self.d2[i] * x.get(i + p, 0),
            );
            y.set(
                i + p,
                0,
                self.d2[i] * x.get(i, 0) + self.d1[i] * x.get(i + p, 0),
            );
        }
    }

    fn mat_t_vec_mul(&self, a: &M, x: &M, y: &mut M) {
        self.mat_vec_mul(a, x, y);
    }
}
