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
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::base::{Array1, Array2, ArrayView1};
use crate::linear::lasso_optimizer::InteriorPointOptimizer;
use crate::num::{FloatNumber, Number};

/// Lasso regression parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LassoParameters {
    /// Controls the strength of the penalty to the loss function.
    pub alpha: f64,
    /// If true the regressors X will be normalized before regression
    /// by subtracting the mean and dividing by the standard deviation.
    pub normalize: bool,
    /// The tolerance for the optimization
    pub tol: f64,
    /// The maximum number of iterations
    pub max_iter: usize,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// Lasso regressor
pub struct Lasso<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    coefficients: X,
    intercept: TX,
    _phantom_ty: PhantomData<TY>,
    _phantom_y: PhantomData<Y>,
}

impl LassoParameters {
    /// Regularization parameter.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    /// If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the standard deviation.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
    /// The tolerance for the optimization
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// The maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl Default for LassoParameters {
    fn default() -> Self {
        LassoParameters {
            alpha: 1f64,
            normalize: true,
            tol: 1e-4,
            max_iter: 1000,
        }
    }
}

impl<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq for Lasso<TX, TY, X, Y> {
    fn eq(&self, other: &Self) -> bool {
        self.intercept == other.intercept
            && self.coefficients.shape() == other.coefficients.shape()
            && self
                .coefficients
                .iterator(0)
                .zip(other.coefficients.iterator(0))
                .all(|(&a, &b)| (a - b).abs() <= TX::epsilon())
    }
}

impl<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, LassoParameters> for Lasso<TX, TY, X, Y>
{
    fn fit(x: &X, y: &Y, parameters: LassoParameters) -> Result<Self, Failed> {
        Lasso::fit(x, y, parameters)
    }
}

impl<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> Predictor<X, Y>
    for Lasso<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> Lasso<TX, TY, X, Y> {
    /// Fits Lasso regression to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target values
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(x: &X, y: &Y, parameters: LassoParameters) -> Result<Lasso<TX, TY, X, Y>, Failed> {
        let (n, p) = x.shape();

        if n <= p {
            return Err(Failed::fit(
                "Number of rows in X should be >= number of columns in X",
            ));
        }

        if parameters.alpha < 0f64 {
            return Err(Failed::fit("alpha should be >= 0"));
        }

        if parameters.tol <= 0f64 {
            return Err(Failed::fit("tol should be > 0"));
        }

        if parameters.max_iter == 0 {
            return Err(Failed::fit("max_iter should be > 0"));
        }

        if y.shape() != n {
            return Err(Failed::fit("Number of rows in X should = len(y)"));
        }

        let y: Vec<TX> = y.iterator(0).map(|&v| TX::from(v).unwrap()).collect();

        let l1_reg = TX::from_f64(parameters.alpha * n as f64).unwrap();

        let (w, b) = if parameters.normalize {
            let (scaled_x, col_mean, col_std) = Self::rescale_x(x)?;

            let mut optimizer = InteriorPointOptimizer::new(&scaled_x, p);

            let mut w = optimizer.optimize(
                &scaled_x,
                &y,
                l1_reg,
                parameters.max_iter,
                TX::from_f64(parameters.tol).unwrap(),
            )?;

            for (j, col_std_j) in col_std.iter().enumerate().take(p) {
                w[j] /= *col_std_j;
            }

            let mut b = TX::zero();

            for (i, col_mean_i) in col_mean.iter().enumerate().take(p) {
                b += w[i] * *col_mean_i;
            }

            b = TX::from_f64(y.mean()).unwrap() - b;
            (X::from_column(&w), b)
        } else {
            let mut optimizer = InteriorPointOptimizer::new(x, p);

            let w = optimizer.optimize(
                x,
                &y,
                l1_reg,
                parameters.max_iter,
                TX::from_f64(parameters.tol).unwrap(),
            )?;

            (X::from_column(&w), TX::from_f64(y.mean()).unwrap())
        };

        Ok(Lasso {
            intercept: b,
            coefficients: w,
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
        })
    }

    /// Predict target values from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let (nrows, _) = x.shape();
        let mut y_hat = x.matmul(&self.coefficients);
        let bias = X::fill(nrows, 1, self.intercept);
        y_hat.add_mut(&bias);
        Ok(Y::from_iterator(
            y_hat.iterator(0).map(|&v| TY::from(v).unwrap()),
            nrows,
        ))
    }

    /// Get estimates regression coefficients
    pub fn coefficients(&self) -> &X {
        &self.coefficients
    }

    /// Get estimate of intercept
    pub fn intercept(&self) -> TX {
        self.intercept
    }

    fn rescale_x(x: &X) -> Result<(X, Vec<TX>, Vec<TX>), Failed> {
        let col_mean: Vec<TX> = x
            .mean(0)
            .iter()
            .map(|&v| TX::from_f64(v).unwrap())
            .collect();
        let col_std: Vec<TX> = x.std(0).iter().map(|&v| TX::from_f64(v).unwrap()).collect();

        for (i, col_std_i) in col_std.iter().enumerate() {
            if (*col_std_i - TX::zero()).abs() < TX::epsilon() {
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
    use crate::linalg::dense::matrix::DenseMatrix;
    use crate::metrics::mean_absolute_error;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "serde")]
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

        let deserialized_lr: Lasso<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            serde_json::from_str(&serde_json::to_string(&lr).unwrap()).unwrap();

        assert_eq!(lr, deserialized_lr);
    }
}
