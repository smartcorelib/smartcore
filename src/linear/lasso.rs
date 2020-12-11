//! # Lasso
//!
//! [Linear regression](../linear_regression/index.html) is the standard algorithm for predicting a quantitative response \\(y\\) on the basis of a linear combination of explanatory variables \\(X\\)
//! that assumes that there is approximately a linear relationship between \\(X\\) and \\(y\\).
//! Lasso is an extension to linear regression that adds L1 regularization term to the loss function during training.
//!
//! Similar to [ridge regression](../ridge_regression/index.html), the lasso shrinks the coefficient estimates towards zero when. However, in the case of the lasso, the l1 penalty has the effect of
//! forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter \\(\alpha\\) is sufficiently large.
//!
//! Lasso coefficient estimates solve the problem:
//!
//! \\[\underset{\beta}{minimize} \space \space \sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p \beta_jx_{ij} \right)^2 + \alpha \sum_{j=1}^p \lVert \beta_j \rVert_1\\]
//!
//! This problem is solved with an interior-point method that is comparable to coordinate descent in solving large problems with modest accuracy,
//! but is able to solve them with high accuracy with relatively small additional computational cost.
//!
//! ## References:
//!
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 6.2. Shrinkage Methods](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["An Interior-Point Method for Large-Scale l1-Regularized Least Squares",  K. Koh, M. Lustig, S. Boyd, D. Gorinevsky](https://web.stanford.edu/~boyd/papers/pdf/l1_ls.pdf)
//! * [Simple Matlab Solver for l1-regularized Least Squares Problems](https://web.stanford.edu/~boyd/l1_ls/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::linear::bg_solver::BiconjugateGradientSolver;
use crate::math::num::RealNumber;

/// Lasso regression parameters
#[derive(Serialize, Deserialize, Debug)]
pub struct LassoParameters<T: RealNumber> {
    /// Controls the strength of the penalty to the loss function.
    pub alpha: T,
    /// If true the regressors X will be normalized before regression
    /// by subtracting the mean and dividing by the standard deviation.
    pub normalize: bool,
    /// The tolerance for the optimization
    pub tol: T,
    /// The maximum number of iterations
    pub max_iter: usize,
}

#[derive(Serialize, Deserialize, Debug)]
/// Lasso regressor
pub struct Lasso<T: RealNumber, M: Matrix<T>> {
    coefficients: M,
    intercept: T,
}

struct InteriorPointOptimizer<T: RealNumber, M: Matrix<T>> {
    ata: M,
    d1: Vec<T>,
    d2: Vec<T>,
    prb: Vec<T>,
    prs: Vec<T>,
}

impl<T: RealNumber> Default for LassoParameters<T> {
    fn default() -> Self {
        LassoParameters {
            alpha: T::one(),
            normalize: true,
            tol: T::from_f64(1e-4).unwrap(),
            max_iter: 1000,
        }
    }
}

impl<T: RealNumber, M: Matrix<T>> PartialEq for Lasso<T, M> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficients == other.coefficients
            && (self.intercept - other.intercept).abs() <= T::epsilon()
    }
}

impl<T: RealNumber, M: Matrix<T>> Lasso<T, M> {
    /// Fits Lasso regression to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target values
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: LassoParameters<T>,
    ) -> Result<Lasso<T, M>, Failed> {
        let (n, p) = x.shape();

        if n <= p {
            return Err(Failed::fit(
                "Number of rows in X should be >= number of columns in X",
            ));
        }

        if parameters.alpha < T::zero() {
            return Err(Failed::fit("alpha should be >= 0"));
        }

        if parameters.tol <= T::zero() {
            return Err(Failed::fit("tol should be > 0"));
        }

        if parameters.max_iter == 0 {
            return Err(Failed::fit("max_iter should be > 0"));
        }

        if y.len() != n {
            return Err(Failed::fit("Number of rows in X should = len(y)"));
        }

        let (w, b) = if parameters.normalize {
            let (scaled_x, col_mean, col_std) = Self::rescale_x(x)?;

            let mut optimizer = InteriorPointOptimizer::new(&scaled_x, p);

            let mut w = optimizer.optimize(&scaled_x, y, &parameters)?;

            for (j, col_std_j) in col_std.iter().enumerate().take(p) {
                w.set(j, 0, w.get(j, 0) / *col_std_j);
            }

            let mut b = T::zero();

            for (i, col_mean_i) in col_mean.iter().enumerate().take(p) {
                b += w.get(i, 0) * *col_mean_i;
            }

            b = y.mean() - b;
            (w, b)
        } else {
            let mut optimizer = InteriorPointOptimizer::new(x, p);

            let w = optimizer.optimize(x, y, &parameters)?;

            (w, y.mean())
        };

        Ok(Lasso {
            intercept: b,
            coefficients: w,
        })
    }

    /// Predict target values from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        let (nrows, _) = x.shape();
        let mut y_hat = x.matmul(&self.coefficients);
        y_hat.add_mut(&M::fill(nrows, 1, self.intercept));
        Ok(y_hat.transpose().to_row_vector())
    }

    /// Get estimates regression coefficients
    pub fn coefficients(&self) -> &M {
        &self.coefficients
    }

    /// Get estimate of intercept
    pub fn intercept(&self) -> T {
        self.intercept
    }

    fn rescale_x(x: &M) -> Result<(M, Vec<T>, Vec<T>), Failed> {
        let col_mean = x.mean(0);
        let col_std = x.std(0);

        for (i, col_std_i) in col_std.iter().enumerate() {
            if (*col_std_i - T::zero()).abs() < T::epsilon() {
                return Err(Failed::fit(&format!(
                    "Cannot rescale constant column {}",
                    i
                )));
            }
        }

        let mut scaled_x = x.clone();
        scaled_x.scale_mut(&col_mean, &col_std, 0);
        Ok((scaled_x, col_mean, col_std))
    }
}

impl<T: RealNumber, M: Matrix<T>> InteriorPointOptimizer<T, M> {
    fn new(a: &M, n: usize) -> InteriorPointOptimizer<T, M> {
        InteriorPointOptimizer {
            ata: a.ab(true, a, false),
            d1: vec![T::zero(); n],
            d2: vec![T::zero(); n],
            prb: vec![T::zero(); n],
            prs: vec![T::zero(); n],
        }
    }

    fn optimize(
        &mut self,
        x: &M,
        y: &M::RowVector,
        parameters: &LassoParameters<T>,
    ) -> Result<M, Failed> {
        let (n, p) = x.shape();
        let p_f64 = T::from_usize(p).unwrap();

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
            .max(T::one() / parameters.alpha)
            .min(T::two() * p_f64 / T::from(1e-3).unwrap());

        for ntiter in 0..parameters.max_iter {
            let mut z = x.matmul(&w);

            for i in 0..n {
                z.set(i, 0, z.get(i, 0) - y.get(i, 0));
                nu.set(i, 0, T::two() * z.get(i, 0));
            }

            // CALCULATE DUALITY GAP
            let xnu = x.ab(true, &nu, false);
            let max_xnu = xnu.norm(T::infinity());
            if max_xnu > parameters.alpha {
                let lnu = parameters.alpha / max_xnu;
                nu.mul_scalar_mut(lnu);
            }

            let pobj = z.dot(&z) + parameters.alpha * w.norm(T::one());
            dobj = dobj.max(gamma * nu.dot(&nu) - nu.dot(&y));

            let gap = pobj - dobj;

            // STOPPING CRITERION
            if gap / dobj < parameters.tol {
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
                let g2 = parameters.alpha - (q1[i] + q2[i]) / t;
                gradphi.set(i, 0, g1);
                grad.set(i, 0, -g1);
                grad.set(i + p, 0, -g2);
            }

            for i in 0..p {
                self.prb[i] = T::two() + self.d1[i];
                self.prs[i] = self.prb[i] * self.d1[i] - self.d2[i] * self.d2[i];
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
            let phi = z.dot(&z) + parameters.alpha * u.sum() - Self::sumlogneg(&f) / t;
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

                    let newphi = newz.dot(&newz) + parameters.alpha * newu.sum()
                        - Self::sumlogneg(&newf) / t;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::*;
    use crate::metrics::mean_absolute_error;

    #[test]
    fn lasso_fit_predict() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
            &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);

        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let y_hat = Lasso::fit(
            &x,
            &y,
            LassoParameters {
                alpha: 0.1,
                normalize: true,
                tol: 1e-4,
                max_iter: 1000,
            },
        )
        .and_then(|lr| lr.predict(&x))
        .unwrap();

        assert!(mean_absolute_error(&y_hat, &y) < 2.0);

        let y_hat = Lasso::fit(
            &x,
            &y,
            LassoParameters {
                alpha: 0.1,
                normalize: false,
                tol: 1e-4,
                max_iter: 1000,
            },
        )
        .and_then(|lr| lr.predict(&x))
        .unwrap();

        assert!(mean_absolute_error(&y_hat, &y) < 2.0);
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
            &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);

        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let lr = Lasso::fit(&x, &y, Default::default()).unwrap();

        let deserialized_lr: Lasso<f64, DenseMatrix<f64>> =
            serde_json::from_str(&serde_json::to_string(&lr).unwrap()).unwrap();

        assert_eq!(lr, deserialized_lr);
    }
}
