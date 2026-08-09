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
//! \\[\underset{\beta}{minimize} \space \space \frac{1}{n} \sum_{i=1}^n \left( y_i - \beta_0 - \sum_{j=1}^p \beta_jx_{ij} \right)^2 + \alpha \sum_{j=1}^p \lVert \beta_j \rVert_1\\]
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
use crate::linalg::basic::arrays::{Array1, Array2, ArrayView1};
use crate::linear::lasso_optimizer::InteriorPointOptimizer;
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

/// Lasso regression parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LassoParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Controls the strength of the penalty to the loss function.
    pub alpha: f64,
    #[cfg_attr(feature = "serde", serde(default))]
    /// If true the regressors X will be normalized before regression
    /// by subtracting the mean and dividing by the standard deviation.
    pub normalize: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The tolerance for the optimization
    pub tol: f64,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The maximum number of iterations
    pub max_iter: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// If false, force the intercept parameter (beta_0) to be zero.
    pub fit_intercept: bool,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// Lasso regressor
pub struct Lasso<TX: FloatNumber + RealNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    coefficients: Option<X>,
    intercept: Option<TX>,
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

    /// If false, force the intercept parameter (beta_0) to be zero.
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
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
            fit_intercept: true,
        }
    }
}

impl<TX: FloatNumber + RealNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for Lasso<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        self.intercept == other.intercept
            && self.coefficients().shape() == other.coefficients().shape()
            && self
                .coefficients()
                .iterator(0)
                .zip(other.coefficients().iterator(0))
                .all(|(&a, &b)| (a - b).abs() <= TX::epsilon())
    }
}

impl<TX: FloatNumber + RealNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, LassoParameters> for Lasso<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
        }
    }

    fn fit(x: &X, y: &Y, parameters: LassoParameters) -> Result<Self, Failed> {
        Lasso::fit(x, y, parameters)
    }
}

impl<TX: FloatNumber + RealNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> Predictor<X, Y>
    for Lasso<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

/// Lasso grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LassoSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Controls the strength of the penalty to the loss function.
    pub alpha: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// If true the regressors X will be normalized before regression
    /// by subtracting the mean and dividing by the standard deviation.
    pub normalize: Vec<bool>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The tolerance for the optimization
    pub tol: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The maximum number of iterations
    pub max_iter: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// If false, force the intercept parameter (beta_0) to be zero.
    pub fit_intercept: Vec<bool>,
}

/// Lasso grid search iterator
pub struct LassoSearchParametersIterator {
    lasso_search_parameters: LassoSearchParameters,
    current_alpha: usize,
    current_normalize: usize,
    current_tol: usize,
    current_max_iter: usize,
    current_fit_intercept: usize,
}

impl IntoIterator for LassoSearchParameters {
    type Item = LassoParameters;
    type IntoIter = LassoSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        LassoSearchParametersIterator {
            lasso_search_parameters: self,
            current_alpha: 0,
            current_normalize: 0,
            current_tol: 0,
            current_max_iter: 0,
            current_fit_intercept: 0,
        }
    }
}

impl Iterator for LassoSearchParametersIterator {
    type Item = LassoParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_alpha == self.lasso_search_parameters.alpha.len()
            && self.current_normalize == self.lasso_search_parameters.normalize.len()
            && self.current_tol == self.lasso_search_parameters.tol.len()
            && self.current_max_iter == self.lasso_search_parameters.max_iter.len()
            && self.current_fit_intercept == self.lasso_search_parameters.fit_intercept.len()
        {
            return None;
        }

        let next = LassoParameters {
            alpha: self.lasso_search_parameters.alpha[self.current_alpha],
            normalize: self.lasso_search_parameters.normalize[self.current_normalize],
            tol: self.lasso_search_parameters.tol[self.current_tol],
            max_iter: self.lasso_search_parameters.max_iter[self.current_max_iter],
            fit_intercept: self.lasso_search_parameters.fit_intercept[self.current_fit_intercept],
        };

        if self.current_alpha + 1 < self.lasso_search_parameters.alpha.len() {
            self.current_alpha += 1;
        } else if self.current_normalize + 1 < self.lasso_search_parameters.normalize.len() {
            self.current_alpha = 0;
            self.current_normalize += 1;
        } else if self.current_tol + 1 < self.lasso_search_parameters.tol.len() {
            self.current_alpha = 0;
            self.current_normalize = 0;
            self.current_tol += 1;
        } else if self.current_max_iter + 1 < self.lasso_search_parameters.max_iter.len() {
            self.current_alpha = 0;
            self.current_normalize = 0;
            self.current_tol = 0;
            self.current_max_iter += 1;
        } else if self.current_fit_intercept + 1 < self.lasso_search_parameters.fit_intercept.len()
        {
            self.current_alpha = 0;
            self.current_normalize = 0;
            self.current_tol = 0;
            self.current_max_iter = 0;
            self.current_fit_intercept += 1;
        } else {
            self.current_alpha += 1;
            self.current_normalize += 1;
            self.current_tol += 1;
            self.current_max_iter += 1;
            self.current_fit_intercept += 1;
        }

        Some(next)
    }
}

impl Default for LassoSearchParameters {
    fn default() -> Self {
        let default_params = LassoParameters::default();

        LassoSearchParameters {
            alpha: vec![default_params.alpha],
            normalize: vec![default_params.normalize],
            tol: vec![default_params.tol],
            max_iter: vec![default_params.max_iter],
            fit_intercept: vec![default_params.fit_intercept],
        }
    }
}

impl<TX: FloatNumber + RealNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> Lasso<TX, TY, X, Y> {
    /// Fits Lasso regression to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target values
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(x: &X, y: &Y, parameters: LassoParameters) -> Result<Lasso<TX, TY, X, Y>, Failed> {
        let (n, p) = x.shape();

        if n < p {
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
                parameters.fit_intercept,
            )?;

            for (j, col_std_j) in col_std.iter().enumerate().take(p) {
                w[j] /= *col_std_j;
            }

            let b = if parameters.fit_intercept {
                let mut xw_mean = TX::zero();
                for (i, col_mean_i) in col_mean.iter().enumerate().take(p) {
                    xw_mean += w[i] * *col_mean_i;
                }

                Some(TX::from_f64(y.mean_by()).unwrap() - xw_mean)
            } else {
                None
            };
            (X::from_column(&w), b)
        } else {
            let mut optimizer = InteriorPointOptimizer::new(x, p);

            let w = optimizer.optimize(
                x,
                &y,
                l1_reg,
                parameters.max_iter,
                TX::from_f64(parameters.tol).unwrap(),
                parameters.fit_intercept,
            )?;

            (
                X::from_column(&w),
                if parameters.fit_intercept {
                    Some(TX::from_f64(y.mean_by()).unwrap())
                } else {
                    None
                },
            )
        };

        Ok(Lasso {
            intercept: b,
            coefficients: Some(w),
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
        })
    }

    /// Predict target values from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let (nrows, _) = x.shape();
        let mut y_hat = x.matmul(self.coefficients());
        let bias = X::fill(nrows, 1, self.intercept.unwrap());
        y_hat.add_mut(&bias);
        Ok(Y::from_iterator(
            y_hat.iterator(0).map(|&v| TY::from(v).unwrap()),
            nrows,
        ))
    }

    /// Get estimates regression coefficients
    pub fn coefficients(&self) -> &X {
        self.coefficients.as_ref().unwrap()
    }

    /// Get estimate of intercept
    pub fn intercept(&self) -> &TX {
        self.intercept.as_ref().unwrap()
    }

    fn rescale_x(x: &X) -> Result<(X, Vec<TX>, Vec<TX>), Failed> {
        let col_mean: Vec<TX> = x
            .mean_by(0)
            .iter()
            .map(|&v| TX::from_f64(v).unwrap())
            .collect();
        let col_std: Vec<TX> = x
            .std_dev(0)
            .iter()
            .map(|&v| TX::from_f64(v).unwrap())
            .collect();

        for (i, col_std_i) in col_std.iter().enumerate() {
            if (*col_std_i - TX::zero()).abs() < TX::epsilon() {
                return Err(Failed::fit(&format!("Cannot rescale constant column {i}")));
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
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::metrics::mean_absolute_error;

    #[test]
    fn search_parameters() {
        let parameters = LassoSearchParameters {
            alpha: vec![0., 1.],
            max_iter: vec![10, 100],
            fit_intercept: vec![false, true],
            ..Default::default()
        };

        let mut iter = parameters.clone().into_iter();
        for current_fit_intercept in 0..parameters.fit_intercept.len() {
            for current_max_iter in 0..parameters.max_iter.len() {
                for current_alpha in 0..parameters.alpha.len() {
                    let next = iter.next().unwrap();
                    assert_eq!(next.alpha, parameters.alpha[current_alpha]);
                    assert_eq!(next.max_iter, parameters.max_iter[current_max_iter]);
                    assert_eq!(
                        next.fit_intercept,
                        parameters.fit_intercept[current_fit_intercept]
                    );
                }
            }
        }
        assert!(iter.next().is_none());
    }

    fn get_example_x_y() -> (DenseMatrix<f64>, Vec<f64>) {
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
        ])
        .unwrap();

        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        (x, y)
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn lasso_fit_predict() {
        let (x, y) = get_example_x_y();

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
                fit_intercept: true,
            },
        )
        .and_then(|lr| lr.predict(&x))
        .unwrap();

        assert!(mean_absolute_error(&y_hat, &y) < 2.0);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_full_rank_x() {
        // x: randn(3,3) * 10, demean, then round to 2 decimal points
        // y = x @ [10.0, 0.2, -3.0], round to 2 decimal points
        let param = LassoParameters::default()
            .with_normalize(false)
            .with_alpha(200.0);
        let x = DenseMatrix::from_2d_array(&[
            &[-8.9, -2.24, 8.89],
            &[-4.02, 8.89, 12.33],
            &[12.92, -6.65, -21.22],
        ])
        .unwrap();

        let y = vec![-116.12, -75.41, 191.53];
        let w = Lasso::fit(&x, &y, param)
            .unwrap()
            .coefficients()
            .iterator(0)
            .copied()
            .collect();

        let expected_w = vec![5.20289531, 0., -5.32823882]; // by coordinate descent
        assert!(mean_absolute_error(&w, &expected_w) < 1e-3); // actual mean_absolute_error is about 2e-4
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_fit_intercept() {
        let (x, y) = get_example_x_y();
        let fit_result = Lasso::fit(
            &x,
            &y,
            LassoParameters {
                alpha: 0.1,
                normalize: false,
                tol: 1e-8,
                max_iter: 1000,
                fit_intercept: false,
            },
        )
        .unwrap();

        let w = fit_result.coefficients().iterator(0).copied().collect();
        // by sklearn LassoLars. coordinate descent doesn't converge well
        let expected_w = vec![
            0.18335684,
            0.02106526,
            0.00703214,
            -1.35952542,
            0.09295222,
            0.,
        ];
        assert!(mean_absolute_error(&w, &expected_w) < 1e-6);
        assert_eq!(fit_result.intercept, None);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let (x, y) = get_example_x_y();
        let lr = Lasso::fit(&x, &y, Default::default()).unwrap();

        let deserialized_lr: Lasso<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            postcard::from_bytes(&postcard::to_allocvec(&lr).unwrap()).unwrap();

        assert_eq!(lr, deserialized_lr);
    }
}
