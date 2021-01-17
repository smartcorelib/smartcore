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

#[cfg(feature = "serde")] use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::linear::lasso_optimizer::InteriorPointOptimizer;
use crate::math::num::RealNumber;

/// Lasso regression parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// Lasso regressor
pub struct Lasso<T: RealNumber, M: Matrix<T>> {
    coefficients: M,
    intercept: T,
}

impl<T: RealNumber> LassoParameters<T> {
    /// Regularization parameter.
    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }
    /// If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the standard deviation.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
    /// The tolerance for the optimization
    pub fn with_tol(mut self, tol: T) -> Self {
        self.tol = tol;
        self
    }
    /// The maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
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

impl<T: RealNumber, M: Matrix<T>> SupervisedEstimator<M, M::RowVector, LassoParameters<T>>
    for Lasso<T, M>
{
    fn fit(x: &M, y: &M::RowVector, parameters: LassoParameters<T>) -> Result<Self, Failed> {
        Lasso::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>> Predictor<M, M::RowVector> for Lasso<T, M> {
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
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

        let l1_reg = parameters.alpha * T::from_usize(n).unwrap();

        let (w, b) = if parameters.normalize {
            let (scaled_x, col_mean, col_std) = Self::rescale_x(x)?;

            let mut optimizer = InteriorPointOptimizer::new(&scaled_x, p);

            let mut w =
                optimizer.optimize(&scaled_x, y, l1_reg, parameters.max_iter, parameters.tol)?;

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

            let w = optimizer.optimize(x, y, l1_reg, parameters.max_iter, parameters.tol)?;

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

        let y_hat = Lasso::fit(&x, &y, Default::default())
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
