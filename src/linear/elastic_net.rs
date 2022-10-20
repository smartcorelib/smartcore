#![allow(clippy::needless_range_loop)]
//! # Elastic Net
//!
//! Elastic net is an extension of [linear regression](../linear_regression/index.html) that adds regularization penalties to the loss function during training.
//! Just like in ordinary linear regression you assume a linear relationship between input variables and the target variable.
//! Unlike linear regression elastic net adds regularization penalties to the loss function during training.
//! In particular, the elastic net coefficient estimates \\(\beta\\) are the values that minimize
//!
//! \\[L(\alpha, \beta) = \vert \boldsymbol{y} - \boldsymbol{X}\beta\vert^2 + \lambda_1 \vert \beta \vert^2 + \lambda_2 \vert \beta \vert_1\\]
//!
//! where \\(\lambda_1 = \\alpha l_{1r}\\), \\(\lambda_2 = \\alpha (1 -  l_{1r})\\) and \\(l_{1r}\\) is the l1 ratio, elastic net mixing parameter.
//!
//! In essense, elastic net combines both the [L1](../lasso/index.html) and [L2](../ridge_regression/index.html) penalties during training,
//! which can result in better performance than a model with either one or the other penalty on some problems.
//! The elastic net is particularly useful when the number of predictors (p) is much bigger than the number of observations (n).
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::linear::elastic_net::*;
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
//!          ]);
//!
//! let y: Vec<f64> = vec![83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0,
//!           100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9];
//!
//! let y_hat = ElasticNet::fit(&x, &y, Default::default()).
//!                 and_then(|lr| lr.predict(&x)).unwrap();
//! ```
//!
//! ## References:
//!
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 6.2. Shrinkage Methods](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["Regularization and variable selection via the elastic net",  Hui Zou and Trevor Hastie](https://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array, Array1, Array2, MutArray};
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;

use crate::linear::lasso_optimizer::InteriorPointOptimizer;

/// Elastic net parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ElasticNetParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Regularization parameter.
    pub alpha: f64,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The elastic net mixing parameter, with 0 <= l1_ratio <= 1.
    /// For l1_ratio = 0 the penalty is an L2 penalty.
    /// For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    pub l1_ratio: f64,
    #[cfg_attr(feature = "serde", serde(default))]
    /// If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the standard deviation.
    pub normalize: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The tolerance for the optimization
    pub tol: f64,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The maximum number of iterations
    pub max_iter: usize,
}

/// Elastic net
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct ElasticNet<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    coefficients: Option<X>,
    intercept: Option<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_y: PhantomData<Y>,
}

impl ElasticNetParameters {
    /// Regularization parameter.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    /// The elastic net mixing parameter, with 0 <= l1_ratio <= 1.
    /// For l1_ratio = 0 the penalty is an L2 penalty.
    /// For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
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

impl Default for ElasticNetParameters {
    fn default() -> Self {
        ElasticNetParameters {
            alpha: 1.0,
            l1_ratio: 0.5,
            normalize: true,
            tol: 1e-4,
            max_iter: 1000,
        }
    }
}

/// ElasticNet grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ElasticNetSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Regularization parameter.
    pub alpha: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The elastic net mixing parameter, with 0 <= l1_ratio <= 1.
    /// For l1_ratio = 0 the penalty is an L2 penalty.
    /// For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    pub l1_ratio: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// If True, the regressors X will be normalized before regression by subtracting the mean and dividing by the standard deviation.
    pub normalize: Vec<bool>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The tolerance for the optimization
    pub tol: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The maximum number of iterations
    pub max_iter: Vec<usize>,
}

/// ElasticNet grid search iterator
pub struct ElasticNetSearchParametersIterator {
    lasso_regression_search_parameters: ElasticNetSearchParameters,
    current_alpha: usize,
    current_l1_ratio: usize,
    current_normalize: usize,
    current_tol: usize,
    current_max_iter: usize,
}

impl IntoIterator for ElasticNetSearchParameters {
    type Item = ElasticNetParameters;
    type IntoIter = ElasticNetSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        ElasticNetSearchParametersIterator {
            lasso_regression_search_parameters: self,
            current_alpha: 0,
            current_l1_ratio: 0,
            current_normalize: 0,
            current_tol: 0,
            current_max_iter: 0,
        }
    }
}

impl Iterator for ElasticNetSearchParametersIterator {
    type Item = ElasticNetParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_alpha == self.lasso_regression_search_parameters.alpha.len()
            && self.current_l1_ratio == self.lasso_regression_search_parameters.l1_ratio.len()
            && self.current_normalize == self.lasso_regression_search_parameters.normalize.len()
            && self.current_tol == self.lasso_regression_search_parameters.tol.len()
            && self.current_max_iter == self.lasso_regression_search_parameters.max_iter.len()
        {
            return None;
        }

        let next = ElasticNetParameters {
            alpha: self.lasso_regression_search_parameters.alpha[self.current_alpha],
            l1_ratio: self.lasso_regression_search_parameters.alpha[self.current_l1_ratio],
            normalize: self.lasso_regression_search_parameters.normalize[self.current_normalize],
            tol: self.lasso_regression_search_parameters.tol[self.current_tol],
            max_iter: self.lasso_regression_search_parameters.max_iter[self.current_max_iter],
        };

        if self.current_alpha + 1 < self.lasso_regression_search_parameters.alpha.len() {
            self.current_alpha += 1;
        } else if self.current_l1_ratio + 1 < self.lasso_regression_search_parameters.l1_ratio.len()
        {
            self.current_alpha = 0;
            self.current_l1_ratio += 1;
        } else if self.current_normalize + 1
            < self.lasso_regression_search_parameters.normalize.len()
        {
            self.current_alpha = 0;
            self.current_l1_ratio = 0;
            self.current_normalize += 1;
        } else if self.current_tol + 1 < self.lasso_regression_search_parameters.tol.len() {
            self.current_alpha = 0;
            self.current_l1_ratio = 0;
            self.current_normalize = 0;
            self.current_tol += 1;
        } else if self.current_max_iter + 1 < self.lasso_regression_search_parameters.max_iter.len()
        {
            self.current_alpha = 0;
            self.current_l1_ratio = 0;
            self.current_normalize = 0;
            self.current_tol = 0;
            self.current_max_iter += 1;
        } else {
            self.current_alpha += 1;
            self.current_l1_ratio += 1;
            self.current_normalize += 1;
            self.current_tol += 1;
            self.current_max_iter += 1;
        }

        Some(next)
    }
}

impl Default for ElasticNetSearchParameters {
    fn default() -> Self {
        let default_params = ElasticNetParameters::default();

        ElasticNetSearchParameters {
            alpha: vec![default_params.alpha],
            l1_ratio: vec![default_params.l1_ratio],
            normalize: vec![default_params.normalize],
            tol: vec![default_params.tol],
            max_iter: vec![default_params.max_iter],
        }
    }
}

impl<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for ElasticNet<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        if self.intercept() != other.intercept() {
            return false
        }
        if self.coefficients().shape() != other.coefficients().shape() {
            return false
        }
        self
            .coefficients()
            .iterator(0)
            .zip(other.coefficients().iterator(0))
            .all(|(&a, &b)| (a - b).abs() <= TX::epsilon())
    }
}

impl<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, ElasticNetParameters> for ElasticNet<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
        }
    }

    fn fit(x: &X, y: &Y, parameters: ElasticNetParameters) -> Result<Self, Failed> {
        ElasticNet::fit(x, y, parameters)
    }
}

impl<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> Predictor<X, Y>
    for ElasticNet<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: FloatNumber, TY: Number, X: Array2<TX>, Y: Array1<TY>> ElasticNet<TX, TY, X, Y> {
    /// Fits elastic net regression to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target values
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: ElasticNetParameters,
    ) -> Result<ElasticNet<TX, TY, X, Y>, Failed> {
        let (n, p) = x.shape();

        if y.shape() != n {
            return Err(Failed::fit("Number of rows in X should = len(y)"));
        }

        let n_float = n as f64;

        let l1_reg = TX::from_f64(parameters.alpha * parameters.l1_ratio * n_float).unwrap();
        let l2_reg =
            TX::from_f64(parameters.alpha * (1.0 - parameters.l1_ratio) * n_float).unwrap();

        let y_mean = TX::from_f64(y.mean()).unwrap();

        let (w, b) = if parameters.normalize {
            let (scaled_x, col_mean, col_std) = Self::rescale_x(x)?;

            let (x, y, gamma) = Self::augment_x_and_y(&scaled_x, y, l2_reg);

            let mut optimizer = InteriorPointOptimizer::new(&x, p);

            let mut w = optimizer.optimize(
                &x,
                &y,
                l1_reg * gamma,
                parameters.max_iter,
                TX::from_f64(parameters.tol).unwrap(),
            )?;

            for i in 0..p {
                w.set(i, gamma * *w.get(i) / col_std[i]);
            }

            let mut b = TX::zero();

            for i in 0..p {
                b += *w.get(i) * col_mean[i];
            }

            b = y_mean - b;

            (X::from_column(&w), b)
        } else {
            let (x, y, gamma) = Self::augment_x_and_y(x, y, l2_reg);

            let mut optimizer = InteriorPointOptimizer::new(&x, p);

            let mut w = optimizer.optimize(
                &x,
                &y,
                l1_reg * gamma,
                parameters.max_iter,
                TX::from_f64(parameters.tol).unwrap(),
            )?;

            for i in 0..p {
                w.set(i, gamma * *w.get(i));
            }

            (X::from_column(&w), y_mean)
        };

        Ok(ElasticNet {
            intercept: Some(b),
            coefficients: Some(w),
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
        })
    }

    /// Predict target values from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let (nrows, _) = x.shape();
        let mut y_hat = x.matmul(self.coefficients.as_ref().unwrap());
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

    fn augment_x_and_y(x: &X, y: &Y, l2_reg: TX) -> (X, Vec<TX>, TX) {
        let (n, p) = x.shape();

        let gamma = TX::one() / (TX::one() + l2_reg).sqrt();
        let padding = gamma * l2_reg.sqrt();

        let mut y2 = Vec::<TX>::zeros(n + p);
        for i in 0..y.shape() {
            y2.set(i, TX::from(*y.get(i)).unwrap());
        }

        let mut x2 = X::zeros(n + p, p);

        for j in 0..p {
            for i in 0..n {
                x2.set((i, j), gamma * *x.get((i, j)));
            }

            x2.set((j + n, j), padding);
        }

        (x2, y2, gamma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::metrics::mean_absolute_error;

    #[test]
    fn search_parameters() {
        let parameters = ElasticNetSearchParameters {
            alpha: vec![0., 1.],
            max_iter: vec![10, 100],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.alpha, 0.);
        assert_eq!(next.max_iter, 10);
        let next = iter.next().unwrap();
        assert_eq!(next.alpha, 1.);
        assert_eq!(next.max_iter, 10);
        let next = iter.next().unwrap();
        assert_eq!(next.alpha, 0.);
        assert_eq!(next.max_iter, 100);
        let next = iter.next().unwrap();
        assert_eq!(next.alpha, 1.);
        assert_eq!(next.max_iter, 100);
        assert!(iter.next().is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn elasticnet_longley() {
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

        let y_hat = ElasticNet::fit(
            &x,
            &y,
            ElasticNetParameters {
                alpha: 1.0,
                l1_ratio: 0.5,
                normalize: false,
                tol: 1e-4,
                max_iter: 1000,
            },
        )
        .and_then(|lr| lr.predict(&x))
        .unwrap();

        assert!(mean_absolute_error(&y_hat, &y) < 30.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn elasticnet_fit_predict1() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0, 1931.0, 1.2232755825400514],
            &[1.0, 1933.0, 1.1379726120972395],
            &[2.0, 1920.0, 1.4366265120543429],
            &[3.0, 1918.0, 1.206005737827858],
            &[4.0, 1934.0, 1.436613542400669],
            &[5.0, 1918.0, 1.1594588621640636],
            &[6.0, 1933.0, 1.19809994745985],
            &[7.0, 1918.0, 1.3396363871645678],
            &[8.0, 1931.0, 1.2535342096493207],
            &[9.0, 1933.0, 1.3101281563456293],
            &[10.0, 1922.0, 1.3585833349920762],
            &[11.0, 1930.0, 1.4830786699709897],
            &[12.0, 1916.0, 1.4919891143094546],
            &[13.0, 1915.0, 1.259655137451551],
            &[14.0, 1932.0, 1.3979191428724789],
            &[15.0, 1917.0, 1.3686634746782371],
            &[16.0, 1932.0, 1.381658454569724],
            &[17.0, 1918.0, 1.4054969025700674],
            &[18.0, 1929.0, 1.3271699396384906],
            &[19.0, 1915.0, 1.1373332337674806],
        ]);

        let y: Vec<f64> = vec![
            1.48, 2.72, 4.52, 5.72, 5.25, 4.07, 3.75, 4.75, 6.77, 4.72, 6.78, 6.79, 8.3, 7.42,
            10.2, 7.92, 7.62, 8.06, 9.06, 9.29,
        ];

        let l1_model = ElasticNet::fit(
            &x,
            &y,
            ElasticNetParameters {
                alpha: 1.0,
                l1_ratio: 1.0,
                normalize: true,
                tol: 1e-4,
                max_iter: 1000,
            },
        )
        .unwrap();

        let l2_model = ElasticNet::fit(
            &x,
            &y,
            ElasticNetParameters {
                alpha: 1.0,
                l1_ratio: 0.0,
                normalize: true,
                tol: 1e-4,
                max_iter: 1000,
            },
        )
        .unwrap();

        let mae_l1 = mean_absolute_error(&l1_model.predict(&x).unwrap(), &y);
        let mae_l2 = mean_absolute_error(&l2_model.predict(&x).unwrap(), &y);

        assert!(mae_l1 < 2.0);
        assert!(mae_l2 < 2.0);

        assert!(l1_model.coefficients().get((0, 0)) > l1_model.coefficients().get((1, 0)));
        assert!(l1_model.coefficients().get((0, 0)) > l1_model.coefficients().get((2, 0)));
    }

    // TODO: serialization for the new DenseMatrix needs to be implemented
    // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    // #[test]
    // #[cfg(feature = "serde")]
    // fn serde() {
    //     let x = DenseMatrix::from_2d_array(&[
    //         &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
    //         &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
    //         &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
    //         &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
    //         &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
    //         &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
    //         &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
    //         &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
    //         &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
    //         &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
    //         &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
    //         &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
    //         &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
    //         &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
    //         &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
    //         &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
    //     ]);

    //     let y = vec![
    //         83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
    //         114.2, 115.7, 116.9,
    //     ];

    //     let lr = ElasticNet::fit(&x, &y, Default::default()).unwrap();

    //     let deserialized_lr: ElasticNet<f64, f64, DenseMatrix<f64>, Vec<f64>> =
    //         serde_json::from_str(&serde_json::to_string(&lr).unwrap()).unwrap();

    //     assert_eq!(lr, deserialized_lr);
    // }
}
