//! # Extra Trees Regressor
//! An Extra-Trees (Extremely Randomized Trees) regressor is an ensemble learning method that fits multiple randomized
//! decision trees on the dataset and averages their predictions to improve accuracy and control over-fitting.
//!
//! It is similar to a standard Random Forest, but introduces more randomness in the way splits are chosen, which can
//! reduce the variance of the model and often make the training process faster.
//!
//! The two key differences from a standard Random Forest are:
//! 1. It uses the whole original dataset to build each tree instead of bootstrap samples.
//! 2. When splitting a node, it chooses a random split point for each feature, rather than the most optimal one.
//!
//! See [ensemble models](../index.html) for more details.
//!
//! Bigger number of estimators in general improves performance of the algorithm with an increased cost of training time.
//! The random sample of _m_ predictors is typically set to be \\(\sqrt{p}\\) from the full set of _p_ predictors.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::ensemble::extra_trees_regressor::*;
//!
//! // Longley dataset ([https://www.statsmodels.org/stable/datasets/generated/longley.html](https://www.statsmodels.org/stable/datasets/generated/longley.html))
//! let x = DenseMatrix::from_2d_array(&[
//!     &[234.289, 235.6, 159., 107.608, 1947., 60.323],
//!     &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
//!     &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
//!     &[284.599, 335.1, 165., 110.929, 1950., 61.187],
//!     &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
//!     &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
//!     &[365.385, 187., 354.7, 115.094, 1953., 64.989],
//!     &[363.112, 357.8, 335., 116.219, 1954., 63.761],
//!     &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
//!     &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
//!     &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
//!     &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
//!     &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
//!     &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
//!     &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
//!     &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
//! ]).unwrap();
//! let y = vec![
//!     83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2,
//!     104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9
//! ];
//!
//! let regressor = ExtraTreesRegressor::fit(&x, &y, Default::default()).unwrap();
//!
//! let y_hat = regressor.predict(&x).unwrap(); // use the same data for prediction
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

use std::default::Default;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::ensemble::base_forest_regressor::{BaseForestRegressor, BaseForestRegressorParameters};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;
use crate::tree::base_tree_regressor::Splitter;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// Parameters of the Extra Trees Regressor
/// Some parameters here are passed directly into base estimator.
pub struct ExtraTreesRegressorParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Tree max depth. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub max_depth: Option<u16>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_leaf: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to split an internal node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_split: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The number of trees in the forest.
    pub n_trees: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of random sample of predictors to use as split candidates.
    pub m: Option<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub keep_samples: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Seed used for bootstrap sampling and feature selection for each tree.
    pub seed: u64,
}

/// Extra Trees Regressor
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct ExtraTreesRegressor<
    TX: Number + FloatNumber + PartialOrd,
    TY: Number,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    forest_regressor: Option<BaseForestRegressor<TX, TY, X, Y>>,
}

impl ExtraTreesRegressorParameters {
    /// Tree max depth. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub fn with_max_depth(mut self, max_depth: u16) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }
    /// The minimum number of samples required to split an internal node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }
    /// The number of trees in the forest.
    pub fn with_n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }
    /// Number of random sample of predictors to use as split candidates.
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = Some(m);
        self
    }

    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub fn with_keep_samples(mut self, keep_samples: bool) -> Self {
        self.keep_samples = keep_samples;
        self
    }

    /// Seed used for bootstrap sampling and feature selection for each tree.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}
impl Default for ExtraTreesRegressorParameters {
    fn default() -> Self {
        ExtraTreesRegressorParameters {
            max_depth: Option::None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 10,
            m: Option::None,
            keep_samples: false,
            seed: 0,
        }
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, ExtraTreesRegressorParameters> for ExtraTreesRegressor<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            forest_regressor: Option::None,
        }
    }

    fn fit(x: &X, y: &Y, parameters: ExtraTreesRegressorParameters) -> Result<Self, Failed> {
        ExtraTreesRegressor::fit(x, y, parameters)
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    Predictor<X, Y> for ExtraTreesRegressor<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    ExtraTreesRegressor<TX, TY, X, Y>
{
    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: ExtraTreesRegressorParameters,
    ) -> Result<ExtraTreesRegressor<TX, TY, X, Y>, Failed> {
        let regressor_params = BaseForestRegressorParameters {
            max_depth: parameters.max_depth,
            min_samples_leaf: parameters.min_samples_leaf,
            min_samples_split: parameters.min_samples_split,
            n_trees: parameters.n_trees,
            m: parameters.m,
            keep_samples: parameters.keep_samples,
            seed: parameters.seed,
            bootstrap: false,
            splitter: Splitter::Random,
        };
        let forest_regressor = BaseForestRegressor::fit(x, y, regressor_params)?;

        Ok(ExtraTreesRegressor {
            forest_regressor: Some(forest_regressor),
        })
    }

    /// Predict class for `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let forest_regressor = self.forest_regressor.as_ref().unwrap();
        forest_regressor.predict(x)
    }

    /// Predict OOB classes for `x`. `x` is expected to be equal to the dataset used in training.
    pub fn predict_oob(&self, x: &X) -> Result<Y, Failed> {
        let forest_regressor = self.forest_regressor.as_ref().unwrap();
        forest_regressor.predict_oob(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::metrics::mean_squared_error;

    #[test]
    fn test_extra_trees_regressor_fit_predict() {
        // Use a simpler, more predictable dataset for unit testing.
        let x = DenseMatrix::from_2d_array(&[
            &[1., 2.],
            &[3., 4.],
            &[5., 6.],
            &[7., 8.],
            &[9., 10.],
            &[11., 12.],
            &[13., 14.],
            &[15., 16.],
        ])
        .unwrap();
        let y = vec![1., 2., 3., 4., 5., 6., 7., 8.];

        let parameters = ExtraTreesRegressorParameters::default()
            .with_n_trees(100)
            .with_seed(42);

        let regressor = ExtraTreesRegressor::fit(&x, &y, parameters).unwrap();
        let y_hat = regressor.predict(&x).unwrap();

        assert_eq!(y_hat.len(), y.len());
        // A basic check to ensure the model is learning something.
        // The error should be significantly less than the variance of y.
        let mse = mean_squared_error(&y, &y_hat);
        // With this simple dataset, the error should be very low.
        assert!(mse < 1.0);
    }

    #[test]
    fn test_fit_predict_higher_dims() {
        // Dataset with 10 features, but y is only dependent on the 3rd feature (index 2).
        let x = DenseMatrix::from_2d_array(&[
            // The 3rd column is the important one. The rest are noise.
            &[0., 0., 10., 5., 8., 1., 4., 9., 2., 7.],
            &[0., 0., 20., 1., 2., 3., 4., 5., 6., 7.],
            &[0., 0., 30., 7., 6., 5., 4., 3., 2., 1.],
            &[0., 0., 40., 9., 2., 4., 6., 8., 1., 3.],
            &[0., 0., 55., 3., 1., 8., 6., 4., 2., 9.],
            &[0., 0., 65., 2., 4., 7., 5., 3., 1., 8.],
        ])
        .unwrap();
        let y = vec![10., 20., 30., 40., 55., 65.];

        let parameters = ExtraTreesRegressorParameters::default()
            .with_n_trees(100)
            .with_seed(42);

        let regressor = ExtraTreesRegressor::fit(&x, &y, parameters).unwrap();
        let y_hat = regressor.predict(&x).unwrap();

        assert_eq!(y_hat.len(), y.len());

        let mse = mean_squared_error(&y, &y_hat);

        // The model should be able to learn this simple relationship perfectly,
        // ignoring the noise features. The MSE should be very low.
        assert!(mse < 1.0);
    }

    #[test]
    fn test_reproducibility() {
        let x = DenseMatrix::from_2d_array(&[
            &[1., 2.],
            &[3., 4.],
            &[5., 6.],
            &[7., 8.],
            &[9., 10.],
            &[11., 12.],
        ])
        .unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let params = ExtraTreesRegressorParameters::default().with_seed(42);

        let regressor1 = ExtraTreesRegressor::fit(&x, &y, params.clone()).unwrap();
        let y_hat1 = regressor1.predict(&x).unwrap();

        let regressor2 = ExtraTreesRegressor::fit(&x, &y, params.clone()).unwrap();
        let y_hat2 = regressor2.predict(&x).unwrap();

        assert_eq!(y_hat1, y_hat2);
    }
}
