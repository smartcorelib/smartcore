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

/// Interior Point Optimizer
pub struct InteriorPointOptimizer<T: FloatNumber, X: Array2<T>> {
    ata: X,
    d1: Vec<T>,
    d2: Vec<T>,
    prb: Vec<T>,
    prs: Vec<T>,
}

impl<T: FloatNumber, X: Array2<T>> InteriorPointOptimizer<T, X> {
    /// Initialize a new Interior Point Optimizer
    pub fn new(a: &X, n: usize) -> InteriorPointOptimizer<T, X> {
        InteriorPointOptimizer {
            ata: a.ab(true, a, false),
            d1: vec![T::zero(); n],
            d2: vec![T::zero(); n],
            prb: vec![T::zero(); n],
            prs: vec![T::zero(); n],
        }
    }

    /// Run the optimization
    pub fn optimize(
        &mut self,
        x: &X,
        y: &Vec<T>,
        lambda: T,
        max_iter: usize,
        tol: T,
        fit_intercept: bool,
    ) -> Result<Vec<T>, Failed> {
        let (n, p) = x.shape();
        let p_f64 = T::from_usize(p).unwrap();

        let lambda = lambda.max(T::epsilon());

        //parameters
        let max_ls_iter = 100;
        let pcgmaxi = 5000;
        let min_pcgtol = T::from_f64(0.1).unwrap();
        let eta = T::from_f64(1E-3).unwrap();
        let alpha = T::from_f64(0.01).unwrap();
        let beta = T::from_f64(0.5).unwrap();
        let gamma = T::from_f64(-0.25).unwrap();
        let mu = T::two();

        // let y = M::from_row_vector(y.sub_scalar(y.mean_by())).transpose();
        let y = if fit_intercept {
            y.sub_scalar(T::from_f64(y.mean_by()).unwrap())
        } else {
            y.to_owned()
        };

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
            let max_xnu = xnu.norm(f64::INFINITY);
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

            dx[..p].copy_from_slice(&dxu[..p]);
            du[..p].copy_from_slice(&dxu[p..(p + p)]);

            // BACKTRACKING LINE SEARCH
            let phi = z.dot(&z) + lambda * u.sum() - Self::sumlogneg(&f) / t;
            s = T::one();
            let gdx = grad.dot(&dxu);

            let mut lsiter = 0;
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
                lsiter += 1;
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

impl<'a, T: FloatNumber, X: Array2<T>> BiconjugateGradientSolver<'a, T, X>
    for InteriorPointOptimizer<T, X>
{
    fn solve_preconditioner(&self, a: &'a X, b: &[T], x: &mut [T]) {
        let (_, p) = a.shape();

        for i in 0..p {
            x[i] = (self.d1[i] * b[i] - self.d2[i] * b[i + p]) / self.prs[i];
            x[i + p] = (-self.d2[i] * b[i] + self.prb[i] * b[i + p]) / self.prs[i];
        }
    }

    fn mat_vec_mul(&self, _: &X, x: &Vec<T>, y: &mut Vec<T>) {
        let (_, p) = self.ata.shape();
        let x_slice = Vec::from_slice(x.slice(0..p).as_ref());
        let atax = x_slice.xa(true, &self.ata);

        for i in 0..p {
            y[i] = T::two() * atax[i] + self.d1[i] * x[i] + self.d2[i] * x[i + p];
            y[i + p] = self.d2[i] * x[i] + self.d1[i] * x[i + p];
        }
    }

    fn mat_t_vec_mul(&self, a: &X, x: &Vec<T>, y: &mut Vec<T>) {
        self.mat_vec_mul(a, x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn new_builds_ata_with_correct_shape() {
        // a is 4x2 -> ata = a^T a is 2x2
        let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.]]).unwrap();
        let opt: InteriorPointOptimizer<f64, DenseMatrix<f64>> = InteriorPointOptimizer::new(&a, 2);
        assert_eq!(opt.ata.shape(), (2, 2));
        // ata[0][0] = sum of column 0 squared = 1+9+25+49 = 84
        assert!((opt.ata.get((0, 0)) - 84.0).abs() < 1e-10);
        // ata[1][1] = sum of column 1 squared = 4+16+36+64 = 120
        assert!((opt.ata.get((1, 1)) - 120.0).abs() < 1e-10);
        // internal scratch vectors sized to p=2
        assert_eq!(opt.d1.len(), 2);
        assert_eq!(opt.d2.len(), 2);
        assert_eq!(opt.prb.len(), 2);
        assert_eq!(opt.prs.len(), 2);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn optimize_with_zero_lambda_recovers_least_squares() {
        // y = 2*x0 + 3*x1 (exact, no noise). With lambda -> 0 (clamped to epsilon
        // internally) the l1-regularized LS solution should approach [2, 3].
        let x = DenseMatrix::from_2d_array(&[&[1., 0.], &[0., 1.], &[1., 1.], &[2., 1.]]).unwrap();
        let y = vec![2.0, 3.0, 5.0, 7.0];

        let mut opt: InteriorPointOptimizer<f64, DenseMatrix<f64>> =
            InteriorPointOptimizer::new(&x, 2);
        let w = opt.optimize(&x, &y, 1e-10, 100, 1e-8, false).unwrap();

        assert_eq!(w.len(), 2);
        assert!((w[0] - 2.0).abs() < 1e-3, "w[0]={} expected ~2.0", w[0]);
        assert!((w[1] - 3.0).abs() < 1e-3, "w[1]={} expected ~3.0", w[1]);
    }
}
