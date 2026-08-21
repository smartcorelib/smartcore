//! # Linear Regression
//!
//! Linear regression is a very straightforward approach for predicting a quantitative response \\(y\\) on the basis of a linear combination of explanatory variables \\(X\\).
//! Linear regression assumes that there is approximately a linear relationship between \\(X\\) and \\(y\\). Formally, we can write this linear relationship as
//!
//! \\[y \approx \beta_0 + \sum_{i=1}^n \beta_iX_i + \epsilon\\]
//!
//! where \\(\epsilon\\) is a mean-zero random error term and the regression coefficients \\(\beta_0, \beta_0, ... \beta_n\\) are unknown, and must be estimated.
//!
//! While regression coefficients can be estimated directly by solving
//!
//! \\[\hat{\beta} = (X^TX)^{-1}X^Ty \\]
//!
//! the \\((X^TX)^{-1}\\) term is both computationally expensive and numerically unstable. An alternative approach is to use a matrix decomposition to avoid this operation.
//! `smartcore` uses [SVD](../../linalg/svd/index.html) and [QR](../../linalg/qr/index.html) matrix decomposition to find estimates of \\(\hat{\beta}\\).
//! The QR decomposition is more computationally efficient and more numerically stable than calculating the normal equation directly,
//! but does not work for all data matrices. Unlike the QR decomposition, all matrices have an SVD decomposition.
//!
//! ### Examples
//!
//! #### Single-Output Regression
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::linear::linear_regression::*;
//!
//! // Longley dataset (https://www.statsmodels.org/stable/datasets/generated/longley.html)
//! let x = DenseMatrix::from_2d_array(&[
//!               &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
//!               &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
//!               &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
//!               &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
//!               &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
//!               &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
//!               &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
//!               &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
//!               &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
//!               &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
//!               &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
//!               &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
//!               &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
//!               &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
//!               &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
//!               &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
//!          ]).unwrap();
//!
//! let y: Vec<f64> = vec![83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0,
//!           100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9];
//!
//! let lr = LinearRegression::fit(&x, &y,
//!             LinearRegressionParameters::default().
//!             with_solver(LinearRegressionSolverName::QR)).unwrap();
//!
//! let y_hat = lr.predict(&x).unwrap();
//! ```
//!
//! #### Multi-Output Regression
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::linear::linear_regression::*;
//!
//! let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[2.0, 1.0], &[3.0, 4.0], &[4.0, 3.0]]).unwrap();
//! let y = DenseMatrix::from_2d_array(&[&[4.0, 1.5], &[5.0, 3.0], &[10.0, 0.5], &[11.0, 2.0]]).unwrap();
//!
//! let model = LinearRegression::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit_matrix(
//!     &x,
//!     &y,
//!     Default::default(),
//! ).unwrap();
//!
//! let y_hat = model.predict_matrix(&x).unwrap();
//! ```
//!
//! ## References:
//!
//! * ["Pattern Recognition and Machine Learning", C.M. Bishop, Linear Models for Regression](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 3. Linear Regression](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["Numerical Recipes: The Art of Scientific Computing",  Press W.H., Teukolsky S.A., Vetterling W.T, Flannery B.P, 3rd ed., Section 15.4 General Linear Least Squares](http://numerical.recipes/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::linalg::traits::qr::QRDecomposable;
use crate::linalg::traits::svd::SVDDecomposable;
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Default, Clone, Eq, PartialEq)]
/// Approach to use for estimation of regression coefficients. QR is more efficient but SVD is more stable.
pub enum LinearRegressionSolverName {
    /// QR decomposition, see [QR](../../linalg/qr/index.html)
    QR,
    #[default]
    /// SVD decomposition, see [SVD](../../linalg/svd/index.html)
    SVD,
}

/// Linear Regression parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LinearRegressionParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Solver to use for estimation of regression coefficients.
    pub solver: LinearRegressionSolverName,
}

impl Default for LinearRegressionParameters {
    fn default() -> Self {
        LinearRegressionParameters {
            solver: LinearRegressionSolverName::SVD,
        }
    }
}

/// Linear Regression
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct LinearRegression<
    TX: Number + RealNumber,
    TY: Number,
    X: Array2<TX> + QRDecomposable<TX> + SVDDecomposable<TX>,
    Y,
> {
    coefficients: Option<X>,
    intercept: Option<X>,
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_y: PhantomData<Y>,
}

impl LinearRegressionParameters {
    /// Solver to use for estimation of regression coefficients.
    pub fn with_solver(mut self, solver: LinearRegressionSolverName) -> Self {
        self.solver = solver;
        self
    }
}

/// Linear Regression grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LinearRegressionSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Solver to use for estimation of regression coefficients.
    pub solver: Vec<LinearRegressionSolverName>,
}

/// Linear Regression grid search iterator
pub struct LinearRegressionSearchParametersIterator {
    linear_regression_search_parameters: LinearRegressionSearchParameters,
    current_solver: usize,
}

impl IntoIterator for LinearRegressionSearchParameters {
    type Item = LinearRegressionParameters;
    type IntoIter = LinearRegressionSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        LinearRegressionSearchParametersIterator {
            linear_regression_search_parameters: self,
            current_solver: 0,
        }
    }
}

impl Iterator for LinearRegressionSearchParametersIterator {
    type Item = LinearRegressionParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_solver == self.linear_regression_search_parameters.solver.len() {
            return None;
        }

        let next = LinearRegressionParameters {
            solver: self.linear_regression_search_parameters.solver[self.current_solver].clone(),
        };

        self.current_solver += 1;

        Some(next)
    }
}

impl Default for LinearRegressionSearchParameters {
    fn default() -> Self {
        let default_params = LinearRegressionParameters::default();

        LinearRegressionSearchParameters {
            solver: vec![default_params.solver],
        }
    }
}

impl<
    TX: Number + RealNumber,
    TY: Number,
    X: Array2<TX> + QRDecomposable<TX> + SVDDecomposable<TX>,
    Y,
> PartialEq for LinearRegression<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        let intercepts_match = match (&self.intercept, &other.intercept) {
            (Some(a), Some(b)) => {
                a.shape() == b.shape()
                    && a.iterator(0)
                        .zip(b.iterator(0))
                        .all(|(&x, &y)| (x - y).abs() <= TX::epsilon())
            }
            (None, None) => true,
            _ => false,
        };

        let coefficients_match = match (&self.coefficients, &other.coefficients) {
            (Some(a), Some(b)) => {
                a.shape() == b.shape()
                    && a.iterator(0)
                        .zip(b.iterator(0))
                        .all(|(&x, &y)| (x - y).abs() <= TX::epsilon())
            }
            (None, None) => true,
            _ => false,
        };

        intercepts_match && coefficients_match
    }
}

impl<
    TX: Number + RealNumber,
    TY: Number,
    X: Array2<TX> + QRDecomposable<TX> + SVDDecomposable<TX>,
    Y: Array1<TY>,
> SupervisedEstimator<X, Y, LinearRegressionParameters> for LinearRegression<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            coefficients: Option::None,
            intercept: Option::None,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
        }
    }

    fn fit(x: &X, y: &Y, parameters: LinearRegressionParameters) -> Result<Self, Failed> {
        LinearRegression::fit(x, y, parameters)
    }
}

impl<
    TX: Number + RealNumber,
    TY: Number,
    X: Array2<TX> + QRDecomposable<TX> + SVDDecomposable<TX>,
    Y: Array1<TY>,
> Predictor<X, Y> for LinearRegression<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<
    TX: Number + RealNumber,
    TY: Number,
    X: Array2<TX> + QRDecomposable<TX> + SVDDecomposable<TX>,
    Y,
> LinearRegression<TX, TY, X, Y>
{
    /// Fits Linear Regression to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - _NxK_ matrix with _N_ observations and _K_ target values in each observation.
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.
    pub fn fit_matrix<YM: Array2<TX>>(
        x: &X,
        y: &YM,
        parameters: LinearRegressionParameters,
    ) -> Result<LinearRegression<TX, TY, X, Y>, Failed> {
        let (x_nrows, num_attributes) = x.shape();
        let (y_nrows, num_targets) = y.shape();

        if x_nrows != y_nrows {
            return Err(Failed::fit(
                "Number of rows of X doesn't match number of rows of Y",
            ));
        }

        // Augment X with column of ones for intercept fitting
        let a = x.h_stack(&X::ones(x_nrows, 1));

        // Convert Y into X storage; both iterator(0) and from_iterator(axis = 0)
        // traverse logical row-major order for every Array2 implementation
        let y_x = X::from_iterator(y.iterator(0).copied(), y_nrows, num_targets, 0);

        // Solve A * W = Y for W matrix of size (num_attributes + 1) x K
        let w = match parameters.solver {
            LinearRegressionSolverName::QR => a.qr_solve_mut(y_x)?,
            LinearRegressionSolverName::SVD => a.svd_solve_mut(y_x)?,
        };

        let weights = X::from_slice(w.slice(0..num_attributes, 0..num_targets).as_ref());
        let intercept = X::from_slice(
            w.slice(num_attributes..num_attributes + 1, 0..num_targets)
                .as_ref(),
        );

        Ok(LinearRegression {
            intercept: Some(intercept),
            coefficients: Some(weights),
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
        })
    }

    /// Predict target values from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict_matrix(&self, x: &X) -> Result<X, Failed> {
        let (nrows, _) = x.shape();

        let intercept = self.intercept_matrix();
        let (_, num_targets) = intercept.shape();

        let mut y_hat = x.matmul(self.coefficients());

        // Tile the 1xK intercept across all rows, then add in one pass
        let bias = X::from_iterator(
            (0..nrows).flat_map(|_| intercept.iterator(0).copied()),
            nrows,
            num_targets,
            0,
        );
        y_hat.add_mut(&bias);

        Ok(y_hat)
    }

    /// Get estimates regression coefficients
    pub fn coefficients(&self) -> &X {
        self.coefficients.as_ref().unwrap()
    }

    /// Get estimate of intercept (first target for multi-output models)
    pub fn intercept(&self) -> &TX {
        self.intercept.as_ref().unwrap().get((0, 0))
    }

    /// Get estimates of intercepts as a _1xK_ matrix
    pub fn intercept_matrix(&self) -> &X {
        self.intercept.as_ref().unwrap()
    }
}

impl<
    TX: Number + RealNumber,
    TY: Number,
    X: Array2<TX> + QRDecomposable<TX> + SVDDecomposable<TX>,
    Y: Array1<TY>,
> LinearRegression<TX, TY, X, Y>
{
    /// Fits Linear Regression to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target values
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: LinearRegressionParameters,
    ) -> Result<LinearRegression<TX, TY, X, Y>, Failed> {
        let y_mat = X::from_iterator(
            y.iterator(0).map(|&v| TX::from(v).unwrap()),
            y.shape(),
            1,
            0,
        );
        Self::fit_matrix(x, &y_mat, parameters)
    }

    /// Predict target values from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let y_hat_mat = self.predict_matrix(x)?;
        let (nrows, _) = x.shape();
        Ok(Y::from_iterator(
            y_hat_mat.iterator(0).map(|&v| TY::from(v).unwrap()),
            nrows,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn search_parameters() {
        let parameters = LinearRegressionSearchParameters {
            solver: vec![
                LinearRegressionSolverName::QR,
                LinearRegressionSolverName::SVD,
            ],
        };
        let mut iter = parameters.into_iter();
        assert_eq!(iter.next().unwrap().solver, LinearRegressionSolverName::QR);
        assert_eq!(iter.next().unwrap().solver, LinearRegressionSolverName::SVD);
        assert!(iter.next().is_none());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn ols_fit_predict() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
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
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8,
        ];

        let lr = LinearRegression::fit(
            &x,
            &y,
            LinearRegressionParameters {
                solver: LinearRegressionSolverName::QR,
            },
        )
        .unwrap();

        let y_hat_qr = lr.predict(&x).unwrap();

        let y_hat_svd = LinearRegression::fit(&x, &y, Default::default())
            .and_then(|lr| lr.predict(&x))
            .unwrap();

        assert!(
            y.iter()
                .zip(y_hat_qr.iter())
                .all(|(&a, &b)| (a - b).abs() <= 5.0)
        );
        assert!(
            y.iter()
                .zip(y_hat_svd.iter())
                .all(|(&a, &b)| (a - b).abs() <= 5.0)
        );

        // Confirm single-output intercept getter works properly
        let intercept: f64 = *lr.intercept();
        assert!((intercept - 83.0).abs() > 0.0);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
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
        ])
        .unwrap();

        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let lr = LinearRegression::fit(&x, &y, Default::default()).unwrap();

        let deserialized_lr: LinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            postcard::from_bytes(&postcard::to_allocvec(&lr).unwrap()).unwrap();

        assert_eq!(lr, deserialized_lr);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn multi_output_serde() {
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[2.0, 1.0], &[3.0, 4.0], &[4.0, 3.0]])
            .unwrap();
        let y = DenseMatrix::from_2d_array(&[&[4.0, 1.5], &[5.0, 3.0], &[10.0, 0.5], &[11.0, 2.0]])
            .unwrap();

        let lr = LinearRegression::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit_matrix(
            &x,
            &y,
            Default::default(),
        )
        .unwrap();

        let deserialized_lr: LinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            postcard::from_bytes(&postcard::to_allocvec(&lr).unwrap()).unwrap();

        assert_eq!(lr, deserialized_lr);
        assert_eq!(
            lr.intercept_matrix().shape(),
            deserialized_lr.intercept_matrix().shape()
        );
    }

    #[test]
    fn multi_output_ols_fit_predict() {
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[2.0, 1.0], &[3.0, 4.0], &[4.0, 3.0]])
            .unwrap();

        let y = DenseMatrix::from_2d_array(&[&[4.0, 1.5], &[5.0, 3.0], &[10.0, 0.5], &[11.0, 2.0]])
            .unwrap();

        // Test SVD solver path
        let model_svd = LinearRegression::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit_matrix(
            &x,
            &y,
            LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::SVD),
        )
        .unwrap();
        let pred_svd = model_svd.predict_matrix(&x).unwrap();

        // Test QR solver path
        let model_qr = LinearRegression::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit_matrix(
            &x,
            &y,
            LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::QR),
        )
        .unwrap();
        let pred_qr = model_qr.predict_matrix(&x).unwrap();

        assert_eq!(pred_svd.shape(), (4, 2));
        assert_eq!(pred_qr.shape(), (4, 2));
        assert_eq!(model_svd.coefficients().shape(), (2, 2));

        // Both solvers should agree on well-conditioned data
        assert!(
            pred_svd
                .iterator(0)
                .zip(pred_qr.iterator(0))
                .all(|(&a, &b)| (a - b).abs() <= 1e-8)
        );
    }

    #[test]
    fn multi_output_numerical_correctness() {
        // Target model: Y_1 = 2*X_1 + 1*X_2 + 3, Y_2 = 1*X_1 + 3*X_2 + 2
        let x = DenseMatrix::from_2d_array(&[
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[2.0, 2.0],
            &[3.0, 1.0],
            &[1.0, 4.0],
        ])
        .unwrap();

        let y = DenseMatrix::from_2d_array(&[
            &[5.0, 3.0],
            &[4.0, 5.0],
            &[9.0, 10.0],
            &[10.0, 8.0],
            &[9.0, 15.0],
        ])
        .unwrap();

        let model = LinearRegression::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit_matrix(
            &x,
            &y,
            Default::default(),
        )
        .unwrap();

        let coefs = model.coefficients();
        let intercept = model.intercept_matrix();

        // Check slope matrix (2 x 2)
        assert!((coefs.get((0, 0)) - 2.0).abs() < 1e-5);
        assert!((coefs.get((1, 0)) - 1.0).abs() < 1e-5);
        assert!((coefs.get((0, 1)) - 1.0).abs() < 1e-5);
        assert!((coefs.get((1, 1)) - 3.0).abs() < 1e-5);

        // Check intercept vector (1 x 2)
        assert!((intercept.get((0, 0)) - 3.0).abs() < 1e-5);
        assert!((intercept.get((0, 1)) - 2.0).abs() < 1e-5);

        // Predictions should reproduce the exact linear targets
        let y_hat = model.predict_matrix(&x).unwrap();
        for i in 0..y.shape().0 {
            for j in 0..y.shape().1 {
                assert!((*y_hat.get((i, j)) - *y.get((i, j))).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn fit_matrix_dimension_mismatch() {
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[2.0, 1.0]]).unwrap();
        let y = DenseMatrix::from_2d_array(&[&[4.0, 1.5]]).unwrap();

        let result = LinearRegression::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit_matrix(
            &x,
            &y,
            Default::default(),
        );

        assert!(result.is_err());
    }
}
