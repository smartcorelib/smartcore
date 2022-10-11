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
use crate::linalg::basic::arrays::{Array1, Array2, ArrayView1, MutArray, MutArrayView1};
use crate::linear::bg_solver::BiconjugateGradientSolver;
use crate::numbers::floatnum::FloatNumber;

///
pub struct InteriorPointOptimizer<T: FloatNumber, X: Array2<T>> {
    ata: X,
    d1: Vec<T>,
    d2: Vec<T>,
    prb: Vec<T>,
    prs: Vec<T>,
}

///
impl<T: FloatNumber, X: Array2<T>> InteriorPointOptimizer<T, X> {
    ///
    pub fn new(a: &X, n: usize) -> InteriorPointOptimizer<T, X> {
        InteriorPointOptimizer {
            ata: a.ab(true, a, false),
            d1: vec![T::zero(); n],
            d2: vec![T::zero(); n],
            prb: vec![T::zero(); n],
            prs: vec![T::zero(); n],
        }
    }

    ///
    pub fn optimize(
        &mut self,
        x: &X,
        y: &Vec<T>,
        lambda: T,
        max_iter: usize,
        tol: T,
    ) -> Result<Vec<T>, Failed> {
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

        // let y = M::from_row_vector(y.sub_scalar(y.mean())).transpose();
        let y = y.sub_scalar(T::from_f64(y.mean()).unwrap());

        let mut max_ls_iter = 100;
        let mut pitr = 0;
        let mut w = Vec::zeros(p);
        let mut neww = w.clone();
        let mut u = Vec::ones(p);
        let mut newu = u.clone();

        let mut f = X::fill(p, 2, -T::one());
        let mut newf = f.clone();

        let mut q1 = vec![T::zero(); p];
        let mut q2 = vec![T::zero(); p];

        let mut dx = Vec::zeros(p);
        let mut du = Vec::zeros(p);
        let mut dxu = Vec::zeros(2 * p);
        let mut grad = Vec::zeros(2 * p);

        let mut nu = Vec::zeros(n);
        let mut dobj = T::zero();
        let mut s = T::infinity();
        let mut t = T::one()
            .max(T::one() / lambda)
            .min(T::two() * p_f64 / T::from(1e-3).unwrap());

        let lambda_f64 = lambda.to_f64().unwrap();

        for ntiter in 0..max_iter {
            let mut z = w.xa(true, x);

            for i in 0..n {
                z[i] -= y[i];
                nu[i] = T::two() * z[i];
            }

            // CALCULATE DUALITY GAP
            let xnu = nu.xa(false, x);
            let max_xnu = xnu.norm(std::f64::INFINITY);
            if max_xnu > lambda_f64 {
                let lnu = T::from_f64(lambda_f64 / max_xnu).unwrap();
                nu.mul_scalar_mut(lnu);
            }

            let pobj = z.dot(&z) + lambda * T::from_f64(w.norm(1f64)).unwrap();
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
                let q1i = T::one() / (u[i] + w[i]);
                let q2i = T::one() / (u[i] - w[i]);
                q1[i] = q1i;
                q2[i] = q2i;
                self.d1[i] = (q1i * q1i + q2i * q2i) / t;
                self.d2[i] = (q1i * q1i - q2i * q2i) / t;
            }

            let mut gradphi = z.xa(false, x);

            for i in 0..p {
                let g1 = T::two() * gradphi[i] - (q1[i] - q2[i]) / t;
                let g2 = lambda - (q1[i] + q2[i]) / t;
                gradphi[i] = g1;
                grad[i] = -g1;
                grad[i + p] = -g2;
            }

            for i in 0..p {
                self.prb[i] = T::two() + self.d1[i];
                self.prs[i] = self.prb[i] * self.d1[i] - self.d2[i].powi(2);
            }

            let normg = T::from_f64(grad.norm2()).unwrap();
            let mut pcgtol = min_pcgtol.min(eta * gap / T::one().min(normg));
            if ntiter != 0 && pitr == 0 {
                pcgtol *= min_pcgtol;
            }

            let error = self.solve_mut(x, &grad, &mut dxu, pcgtol, pcgmaxi)?;
            if error > pcgtol {
                pitr = pcgmaxi;
            }

            for i in 0..p {
                dx[i] = dxu[i];
                du[i] = dxu[i + p];
            }

            // BACKTRACKING LINE SEARCH
            let phi = z.dot(&z) + lambda * u.sum() - Self::sumlogneg(&f) / t;
            s = T::one();
            let gdx = grad.dot(&dxu);

            let lsiter = 0;
            while lsiter < max_ls_iter {
                for i in 0..p {
                    neww[i] = w[i] + s * dx[i];
                    newu[i] = u[i] + s * du[i];
                    newf.set((i, 0), neww[i] - newu[i]);
                    newf.set((i, 1), -neww[i] - newu[i]);
                }

                if newf
                    .iterator(0)
                    .fold(T::neg_infinity(), |max, v| v.max(max))
                    < T::zero()
                {
                    let mut newz = neww.xa(true, x);
                    for i in 0..n {
                        newz[i] -= y[i];
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

    ///
    fn sumlogneg(f: &X) -> T {
        let (n, _) = f.shape();
        let mut sum = T::zero();
        for i in 0..n {
            sum += (-*f.get((i, 0))).ln();
            sum += (-*f.get((i, 1))).ln();
        }
        sum
    }
}

///
impl<'a, T: FloatNumber, X: Array2<T>> BiconjugateGradientSolver<T, X>
    for InteriorPointOptimizer<T, X>
{
    ///
    fn solve_preconditioner(&self, a: &X, b: &Vec<T>, x: &mut Vec<T>) {
        let (_, p) = a.shape();

        for i in 0..p {
            x[i] = (self.d1[i] * b[i] - self.d2[i] * b[i + p]) / self.prs[i];
            x[i + p] = (-self.d2[i] * b[i] + self.prb[i] * b[i + p]) / self.prs[i];
        }
    }

    ///
    fn mat_vec_mul(&self, _: &X, x: &Vec<T>, y: &mut Vec<T>) {
        let (_, p) = self.ata.shape();
        let x_slice = Vec::from_slice(x.slice(0..p).as_ref());
        let atax = x_slice.xa(true, &self.ata);

        for i in 0..p {
            y[i] = T::two() * atax[i] + self.d1[i] * x[i] + self.d2[i] * x[i + p];
            y[i + p] = self.d2[i] * x[i] + self.d1[i] * x[i + p];
        }
    }

    ///
    fn mat_t_vec_mul(&self, a: &X, x: &Vec<T>, y: &mut Vec<T>) {
        self.mat_vec_mul(a, x, y);
    }
}
