//! # Random Forest Regressor
//! A random forest is an ensemble estimator that fits multiple [decision trees](../../tree/index.html) to random subsets of the dataset and averages predictions
//! to improve the predictive accuracy and control over-fitting. See [ensemble models](../index.html) for more details.
//!
//! Bigger number of estimators in general improves performance of the algorithm with an increased cost of training time.
//! The random sample of _m_ predictors is typically set to be \\(\sqrt{p}\\) from the full set of _p_ predictors.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::ensemble::random_forest_regressor::*;
//!
//! // Longley dataset (https://www.statsmodels.org/stable/datasets/generated/longley.html)
//! let x = DenseMatrix::from_2d_array(&[
//!             &[234.289, 235.6, 159., 107.608, 1947., 60.323],
//!             &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
//!             &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
//!             &[284.599, 335.1, 165., 110.929, 1950., 61.187],
//!             &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
//!             &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
//!             &[365.385, 187., 354.7, 115.094, 1953., 64.989],
//!             &[363.112, 357.8, 335., 116.219, 1954., 63.761],
//!             &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
//!             &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
//!             &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
//!             &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
//!             &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
//!             &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
//!             &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
//!             &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
//!         ]).unwrap();
//! let y = vec![
//!             83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2,
//!             104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9
//!         ];
//!
//! let regressor = RandomForestRegressor::fit(&x, &y, Default::default()).unwrap();
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

/// Validates the sample weights
fn validate_sample_weights(sample_weights: &[f64], n_rows: usize) -> Result<(), Failed> {
    if sample_weights.len() != n_rows {
        return Err(Failed::fit(
            "Number of sample weights must equal number of rows in x",
        ));
    }
    if sample_weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(Failed::fit(
            "Sample weights must be finite and non-negative",
        ));
    }
    if sample_weights.iter().sum::<f64>() <= 0.0 {
        return Err(Failed::fit("Sum of sample weights must be positive"));
    }
    Ok(())
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// Parameters of the Random Forest Regressor
/// Some parameters here are passed directly into base estimator.
#[must_use]
pub struct RandomForestRegressorParameters {
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

/// Random Forest Regressor
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct RandomForestRegressor<
    TX: Number + FloatNumber + PartialOrd,
    TY: Number,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    forest_regressor: Option<BaseForestRegressor<TX, TY, X, Y>>,
}

impl RandomForestRegressorParameters {
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
impl Default for RandomForestRegressorParameters {
    fn default() -> Self {
        RandomForestRegressorParameters {
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

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for RandomForestRegressor<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        self.forest_regressor == other.forest_regressor
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, RandomForestRegressorParameters>
    for RandomForestRegressor<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            forest_regressor: Option::None,
        }
    }

    fn fit(x: &X, y: &Y, parameters: RandomForestRegressorParameters) -> Result<Self, Failed> {
        RandomForestRegressor::fit(x, y, parameters)
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    Predictor<X, Y> for RandomForestRegressor<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

/// RandomForestRegressor grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
#[must_use]
pub struct RandomForestRegressorSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Tree max depth. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub max_depth: Vec<Option<u16>>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub min_samples_leaf: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to split an internal node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub min_samples_split: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The number of trees in the forest.
    pub n_trees: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of random sample of predictors to use as split candidates.
    pub m: Vec<Option<usize>>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub keep_samples: Vec<bool>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Seed used for bootstrap sampling and feature selection for each tree.
    pub seed: Vec<u64>,
}

/// RandomForestRegressor grid search iterator
pub struct RandomForestRegressorSearchParametersIterator {
    random_forest_regressor_search_parameters: RandomForestRegressorSearchParameters,
    current_max_depth: usize,
    current_min_samples_leaf: usize,
    current_min_samples_split: usize,
    current_n_trees: usize,
    current_m: usize,
    current_keep_samples: usize,
    current_seed: usize,
}

impl IntoIterator for RandomForestRegressorSearchParameters {
    type Item = RandomForestRegressorParameters;
    type IntoIter = RandomForestRegressorSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        RandomForestRegressorSearchParametersIterator {
            random_forest_regressor_search_parameters: self,
            current_max_depth: 0,
            current_min_samples_leaf: 0,
            current_min_samples_split: 0,
            current_n_trees: 0,
            current_m: 0,
            current_keep_samples: 0,
            current_seed: 0,
        }
    }
}

impl Iterator for RandomForestRegressorSearchParametersIterator {
    type Item = RandomForestRegressorParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_max_depth
            == self
                .random_forest_regressor_search_parameters
                .max_depth
                .len()
            && self.current_min_samples_leaf
                == self
                    .random_forest_regressor_search_parameters
                    .min_samples_leaf
                    .len()
            && self.current_min_samples_split
                == self
                    .random_forest_regressor_search_parameters
                    .min_samples_split
                    .len()
            && self.current_n_trees == self.random_forest_regressor_search_parameters.n_trees.len()
            && self.current_m == self.random_forest_regressor_search_parameters.m.len()
            && self.current_keep_samples
                == self
                    .random_forest_regressor_search_parameters
                    .keep_samples
                    .len()
            && self.current_seed == self.random_forest_regressor_search_parameters.seed.len()
        {
            return None;
        }

        let next = RandomForestRegressorParameters {
            max_depth: self.random_forest_regressor_search_parameters.max_depth
                [self.current_max_depth],
            min_samples_leaf: self
                .random_forest_regressor_search_parameters
                .min_samples_leaf[self.current_min_samples_leaf],
            min_samples_split: self
                .random_forest_regressor_search_parameters
                .min_samples_split[self.current_min_samples_split],
            n_trees: self.random_forest_regressor_search_parameters.n_trees[self.current_n_trees],
            m: self.random_forest_regressor_search_parameters.m[self.current_m],
            keep_samples: self.random_forest_regressor_search_parameters.keep_samples
                [self.current_keep_samples],
            seed: self.random_forest_regressor_search_parameters.seed[self.current_seed],
        };

        if self.current_max_depth + 1
            < self
                .random_forest_regressor_search_parameters
                .max_depth
                .len()
        {
            self.current_max_depth += 1;
        } else if self.current_min_samples_leaf + 1
            < self
                .random_forest_regressor_search_parameters
                .min_samples_leaf
                .len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf += 1;
        } else if self.current_min_samples_split + 1
            < self
                .random_forest_regressor_search_parameters
                .min_samples_split
                .len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split += 1;
        } else if self.current_n_trees + 1
            < self.random_forest_regressor_search_parameters.n_trees.len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_n_trees += 1;
        } else if self.current_m + 1 < self.random_forest_regressor_search_parameters.m.len() {
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_n_trees = 0;
            self.current_m += 1;
        } else if self.current_keep_samples + 1
            < self
                .random_forest_regressor_search_parameters
                .keep_samples
                .len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_n_trees = 0;
            self.current_m = 0;
            self.current_keep_samples += 1;
        } else if self.current_seed + 1 < self.random_forest_regressor_search_parameters.seed.len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_n_trees = 0;
            self.current_m = 0;
            self.current_keep_samples = 0;
            self.current_seed += 1;
        } else {
            self.current_max_depth += 1;
            self.current_min_samples_leaf += 1;
            self.current_min_samples_split += 1;
            self.current_n_trees += 1;
            self.current_m += 1;
            self.current_keep_samples += 1;
            self.current_seed += 1;
        }

        Some(next)
    }
}

impl Default for RandomForestRegressorSearchParameters {
    fn default() -> Self {
        let default_params = RandomForestRegressorParameters::default();

        RandomForestRegressorSearchParameters {
            max_depth: vec![default_params.max_depth],
            min_samples_leaf: vec![default_params.min_samples_leaf],
            min_samples_split: vec![default_params.min_samples_split],
            n_trees: vec![default_params.n_trees],
            m: vec![default_params.m],
            keep_samples: vec![default_params.keep_samples],
            seed: vec![default_params.seed],
        }
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    RandomForestRegressor<TX, TY, X, Y>
{
    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: RandomForestRegressorParameters,
    ) -> Result<RandomForestRegressor<TX, TY, X, Y>, Failed> {
        Self::fit_inner(x, y, None, parameters)
    }

    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    /// * `sample_weights`: sample_weights to use during fitting
    pub fn fit_with_weights(
        x: &X,
        y: &Y,
        sample_weights: &[f64],
        parameters: RandomForestRegressorParameters,
    ) -> Result<RandomForestRegressor<TX, TY, X, Y>, Failed> {
        validate_sample_weights(sample_weights, x.shape().0)?;
        Self::fit_inner(x, y, Some(sample_weights), parameters)
    }

    fn fit_inner(
        x: &X,
        y: &Y,
        sample_weights: Option<&[f64]>,
        parameters: RandomForestRegressorParameters,
    ) -> Result<RandomForestRegressor<TX, TY, X, Y>, Failed> {
        let regressor_params = BaseForestRegressorParameters {
            max_depth: parameters.max_depth,
            min_samples_leaf: parameters.min_samples_leaf,
            min_samples_split: parameters.min_samples_split,
            n_trees: parameters.n_trees,
            m: parameters.m,
            keep_samples: parameters.keep_samples,
            seed: parameters.seed,
            bootstrap: true,
            splitter: Splitter::Best,
        };
        let forest_regressor = BaseForestRegressor::fit(x, y, sample_weights, regressor_params)?;

        Ok(RandomForestRegressor {
            forest_regressor: Some(forest_regressor),
        })
    }

    /// Predict class for `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        match &self.forest_regressor {
            Some(forest) => forest.predict(x),
            None => Err(Failed::predict(
                "'fit' should be called before calling 'predict'",
            )),
        }
    }

    /// Predict OOB classes for `x`. `x` is expected to be equal to the dataset used in training.
    pub fn predict_oob(&self, x: &X) -> Result<Y, Failed> {
        match &self.forest_regressor {
            Some(forest) => forest.predict_oob(x),
            None => Err(Failed::predict(
                "'fit' should be called before calling 'predict'",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FailedError;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::metrics::mean_absolute_error;

    #[test]
    fn search_parameters() {
        let parameters = RandomForestRegressorSearchParameters {
            n_trees: vec![10, 100],
            m: vec![None, Some(1)],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 10);
        assert_eq!(next.m, None);
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 100);
        assert_eq!(next.m, None);
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 10);
        assert_eq!(next.m, Some(1));
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 100);
        assert_eq!(next.m, Some(1));
        assert!(iter.next().is_none());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_longley() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159., 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165., 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            &[365.385, 187., 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335., 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
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

        let y_hat = RandomForestRegressor::fit(
            &x,
            &y,
            RandomForestRegressorParameters {
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 1000,
                m: Option::None,
                keep_samples: false,
                seed: 87,
            },
        )
        .and_then(|rf| rf.predict(&x))
        .unwrap();

        assert!(mean_absolute_error(&y, &y_hat) < 1.0);
    }

    #[test]
    fn test_random_matrix_with_wrong_rownum() {
        let x_rand: DenseMatrix<f64> = DenseMatrix::<f64>::rand(17, 200);

        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let fail = RandomForestRegressor::fit(
            &x_rand,
            &y,
            RandomForestRegressorParameters {
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 1000,
                m: Option::None,
                keep_samples: false,
                seed: 87,
            },
        );

        assert!(fail.is_err());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_predict_longley_oob() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159., 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165., 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            &[365.385, 187., 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335., 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
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

        let regressor = RandomForestRegressor::fit(
            &x,
            &y,
            RandomForestRegressorParameters {
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 1000,
                m: Option::None,
                keep_samples: true,
                seed: 87,
            },
        )
        .unwrap();

        let y_hat = regressor.predict(&x).unwrap();
        let y_hat_oob = regressor.predict_oob(&x).unwrap();

        println!("{:?}", mean_absolute_error(&y, &y_hat));
        println!("{:?}", mean_absolute_error(&y, &y_hat_oob));

        assert!(mean_absolute_error(&y, &y_hat) < mean_absolute_error(&y, &y_hat_oob));
    }

    #[test]
    fn fit_with_weights_validates_weights() {
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator((0..6).map(|i| i as f64), 3, 2, 0);
        let y = vec![1.0_f64, 2.0, 3.0];
        let parameters = RandomForestRegressorParameters::default()
            .with_n_trees(5)
            .with_seed(42);

        // Valid weights: a zero weight is permitted when the sum is positive
        for weights in [vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.5]] {
            assert!(
                RandomForestRegressor::fit_with_weights(&x, &y, &weights, parameters.clone())
                    .is_ok(),
                "weights: {weights:?}"
            );
        }

        let wrong_length = "Number of sample weights must equal number of rows in x";
        let not_finite_or_negative = "Sample weights must be finite and non-negative";
        let zero_sum = "Sum of sample weights must be positive";
        let cases: Vec<(Vec<f64>, &str)> = vec![
            (vec![], wrong_length),
            (vec![1.0, 2.0], wrong_length),
            (vec![1.0, 2.0, 3.0, 4.0], wrong_length),
            (vec![1.0, -1.0, 3.0], not_finite_or_negative),
            (vec![1.0, f64::NAN, 3.0], not_finite_or_negative),
            (vec![1.0, f64::INFINITY, 3.0], not_finite_or_negative),
            (vec![0.0, 0.0, 0.0], zero_sum),
        ];
        for (weights, msg) in cases {
            let result =
                RandomForestRegressor::fit_with_weights(&x, &y, &weights, parameters.clone());
            assert_eq!(result.err(), Some(Failed::fit(msg)), "weights: {weights:?}");
        }
    }

    #[test]
    fn fit_with_weights_predicts_approx_weighted_mean() {
        // 20 rows, 1 feature. Stumps (max_depth = 0): each tree predicts the
        // weighted mean of y over its bootstrap sample (which is also weighted)
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator((0..20).map(|i| i as f64), 20, 1, 0);
        let y: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 10.0 }).collect();
        // Rows with y = 10 have weight 9, rows with y = 0 have weight 1.
        let sample_weights: Vec<f64> = (0..20).map(|i| if i < 10 { 1.0 } else { 9.0 }).collect();

        let parameters = RandomForestRegressorParameters::default()
            .with_max_depth(0)
            .with_n_trees(500)
            .with_seed(42);

        let forest =
            RandomForestRegressor::fit_with_weights(&x, &y, &sample_weights, parameters.clone())
                .expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");

        // with bootstrapping the weight should be close to 9, but not extremely close
        for p in y_hat.iter() {
            assert!(
                (p - 9.0).abs() < 0.1,
                "expected value reasonably close to 9, got {p}"
            );
        }

        // Without weights, the predicted value should be close to 5
        let forest = RandomForestRegressor::fit(&x, &y, parameters).expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");

        // Due to bootstrapping predicted value is not exactly 5
        for p in y_hat.iter() {
            assert!((p - 5.0).abs() < 0.1, "expected value around 5, got {p}");
        }
    }

    #[test]
    fn fit_with_same_seed_is_deterministic() {
        // 30 rows, 3 features, deterministic non-linear data
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator(
            (0..90).map(|k| ((k % 17) as f64) / (10.0 + (k % 17) as f64)),
            30,
            3,
            0,
        );
        let model_parameters = (1..=3).map(|x| x as f64).collect::<Vec<_>>();
        let y: Vec<f64> = model_parameters.xa(true, &x);
        let sample_weights: Vec<f64> = (0..30).map(|i| 1.0 + (i % 4) as f64).collect();

        let parameters = RandomForestRegressorParameters::default()
            .with_n_trees(20)
            .with_m(2) // only use 2 attributes
            .with_seed(42);

        // Without weights
        let forest_1 = RandomForestRegressor::fit(&x, &y, parameters.clone()).unwrap();
        let forest_2 = RandomForestRegressor::fit(&x, &y, parameters.clone()).unwrap();
        assert_eq!(&forest_1, &forest_2);
        assert_eq!(forest_1.predict(&x).unwrap(), forest_2.predict(&x).unwrap());

        // With weights
        let forest_1 =
            RandomForestRegressor::fit_with_weights(&x, &y, &sample_weights, parameters.clone())
                .unwrap();
        let forest_2 =
            RandomForestRegressor::fit_with_weights(&x, &y, &sample_weights, parameters).unwrap();
        assert_eq!(forest_1, forest_2);
        assert_eq!(forest_1.predict(&x).unwrap(), forest_2.predict(&x).unwrap());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159., 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165., 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            &[365.385, 187., 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335., 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
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

        let forest = RandomForestRegressor::fit(&x, &y, Default::default()).unwrap();

        let deserialized_forest: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            postcard::from_bytes(&postcard::to_allocvec(&forest).unwrap()).unwrap();

        assert_eq!(forest, deserialized_forest);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_without_fit_should_not_panic() {
        let forest: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            RandomForestRegressor::new();
        let x = DenseMatrix::from_2d_array(&[&[1.0f64]]).expect("Construction of x should work");
        let yhat = forest.predict(&x);
        assert!(yhat.is_err());
        let msg = "'fit' should be called before calling 'predict'";
        assert_eq!(yhat.err(), Some(Failed::predict(msg)));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_oob_without_fit_should_not_panic() {
        let forest: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            RandomForestRegressor::new();
        let x = DenseMatrix::from_2d_array(&[&[1.0f64]]).expect("Construction of x should work");
        let yhat = forest.predict_oob(&x);
        assert!(yhat.is_err());
        let msg = "'fit' should be called before calling 'predict'";
        assert_eq!(yhat.err(), Some(Failed::predict(msg)));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_with_empty_x_should_not_panic() {
        let parameters = RandomForestRegressorParameters::default()
            .with_n_trees(5)
            .with_seed(42);
        // (rows, columns, y): no rows, no columns, or both
        let cases: Vec<(usize, usize, Vec<f64>)> =
            vec![(0, 0, vec![]), (0, 3, vec![]), (3, 0, vec![1.0, 2.0, 3.0])];
        for (nrows, ncols, y) in cases {
            let x: DenseMatrix<f64> = DenseMatrix::new(nrows, ncols, vec![], false)
                .expect("Construction of empty x should work");
            let result = RandomForestRegressor::fit(&x, &y, parameters.clone());
            let expected = Failed::because(
                FailedError::ParametersError,
                "Training data must contain at least one sample and one feature.",
            );
            assert_eq!(result.err(), Some(expected), "shape: ({nrows}, {ncols})");
        }
    }

    mod sklearn_parity {
        use super::*;
        // sklearn parity tests.
        //
        // smartcore and numpy use different RNGs, thus the bootstrap samples and the feature
        // subsets are different. With many trees, both forests converge to the same bagged
        // predictor. Thus we compare the predictions within a tolerance.
        //
        // Reference: sklearn 1.9.1, numpy 2.4.6. Each reference value is the mean of 10 sklearn
        // runs (random_state = 0..10). The tolerance is approximately 4 x the largest standard
        // deviation of one sklearn run, for each row.
        //
        // ```python
        // rng = np.random.default_rng(0)
        // x = np.round(rng.uniform(-1, 1, (40, 4)), 4)
        // y = np.round(x[:, 0] * x[:, 1] + np.sin(3 * x[:, 2]) + 0.1 * rng.normal(size=40), 4)
        // x_probe = np.round(rng.uniform(-1, 1, (10, 4)), 4)
        // for max_features in [1.0, 2]:
        //     for seed in range(10):
        //         rf = RandomForestRegressor(n_estimators=2000, max_features=max_features,
        //             max_depth=None, min_samples_leaf=1, min_samples_split=2, bootstrap=True,
        //             oob_score=True, random_state=seed).fit(x, y)
        //         # collect rf.predict(x), rf.predict(x_probe), rf.oob_prediction_
        // ```
        //

        fn sklearn_parity_train_data() -> (DenseMatrix<f64>, Vec<f64>) {
            let x = DenseMatrix::from_2d_array(&[
                &[0.2739, -0.4604, -0.9181, -0.9669],
                &[0.6265, 0.8255, 0.2133, 0.459],
                &[0.0872, 0.8701, 0.6317, -0.9945],
                &[0.7148, -0.9328, 0.4593, -0.6487],
                &[0.7264, 0.0829, -0.4006, -0.1546],
                &[-0.9434, -0.7514, 0.3412, 0.2944],
                &[0.2308, -0.2326, 0.9944, 0.9617],
                &[0.3711, 0.3009, 0.3769, -0.2222],
                &[-0.7298, 0.443, 0.0507, -0.3795],
                &[-0.0283, 0.779, 0.8681, -0.2844],
                &[0.1431, -0.3563, 0.1886, -0.3242],
                &[-0.2168, 0.7805, -0.5457, 0.2464],
                &[-0.832, 0.6653, 0.5742, -0.5213],
                &[0.753, -0.8829, -0.3278, -0.6994],
                &[-0.0993, 0.5926, -0.5387, -0.896],
                &[-0.1909, -0.603, -0.8185, 0.1607],
                &[-0.4026, 0.344, -0.601, 0.8842],
                &[-0.2698, -0.789, 0.2582, 0.8543],
                &[-0.1192, 0.9092, -0.0002, -0.1495],
                &[0.2404, 0.9902, 0.8979, -0.0799],
                &[0.5155, -0.0052, 0.0586, 0.5716],
                &[-0.1707, 0.469, 0.4223, 0.8641],
                &[-0.7701, 0.458, 0.8548, 0.9359],
                &[-0.9706, 0.7273, 0.9624, 0.9144],
                &[-0.7025, 0.9453, 0.7799, 0.6447],
                &[-0.04, -0.5353, 0.6038, 0.8471],
                &[-0.4677, 0.0779, -0.1145, 0.862],
                &[-0.919, 0.464, 0.2287, -0.9433],
                &[0.4384, -0.968, 0.5159, 0.0255],
                &[0.8582, -0.8678, 0.6826, -0.8666],
                &[-0.3114, -0.1394, 0.9321, 0.1245],
                &[-0.4823, -0.5166, 0.7762, -0.5483],
                &[-0.7509, -0.4233, 0.1722, 0.1082],
                &[0.6194, 0.121, -0.4232, -0.1742],
                &[0.6362, 0.253, 0.9182, -0.2612],
                &[0.1052, 0.1878, 0.6966, -0.7091],
                &[-0.187, 0.8199, -0.9139, 0.6454],
                &[-0.1692, 0.6596, -0.9801, -0.2699],
                &[-0.8427, 0.3052, -0.4523, 0.4053],
                &[0.8876, -0.7464, 0.7296, -0.8811],
            ])
            .unwrap();
            let y = vec![
                -0.5144, 0.9957, 0.7839, 0.3660, -0.9022, 1.5099, 0.0804, 1.1980, -0.1768, 0.4984,
                0.3364, -1.0023, 0.5267, -1.3905, -1.0531, -0.4267, -1.0746, 0.9736, -0.1242,
                0.5237, 0.2751, 0.6806, 0.1690, -0.4747, -0.0497, 1.0539, -0.3933, 0.1634, 0.6273,
                0.0960, 0.5208, 1.0106, 0.7643, -1.0745, 0.4076, 0.9968, -0.5477, -0.3399, -1.0701,
                0.0243,
            ];
            (x, y)
        }

        fn sklearn_parity_probe_data() -> DenseMatrix<f64> {
            DenseMatrix::from_2d_array(&[
                &[-0.3606, -0.625, 0.3451, -0.6098],
                &[0.1554, 0.2045, 0.9248, -0.8555],
                &[-0.0001, 0.4882, -0.6455, -0.2239],
                &[-0.8742, 0.4518, -0.8245, -0.2098],
                &[0.747, -0.0554, 0.8252, 0.5318],
                &[0.8306, -0.7452, -0.8529, -0.8593],
                &[0.7377, 0.2681, -0.0069, -0.6729],
                &[0.3475, -0.364, 0.4218, -0.0793],
                &[0.0149, 0.5793, -0.8145, 0.1575],
                &[-0.6055, 0.6163, -0.0223, 0.9774],
            ])
            .unwrap()
        }

        fn sklearn_sample_weights() -> Vec<f64> {
            (0..40).into_iter().map(|i| ((i % 4) + 1) as f64).collect()
        }

        fn assert_close_to_sklearn(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
            assert_eq!(actual.len(), expected.len(), "{label}: length");
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() <= tol,
                    "{label}, row {i}: smartcore {a}, sklearn {e}, tol {tol}"
                );
            }
        }

        /// Fits the forest with 2000 trees and compares the predictions with sklearn.
        fn check_sklearn_parity(
            m: usize,
            train_ref: &[f64],
            probe_ref: &[f64],
            oob_ref: &[f64],
            tol: f64,
            oob_tol: f64,
            sample_weights: Option<&[f64]>,
        ) {
            let (x, y) = sklearn_parity_train_data();
            let x_probe = sklearn_parity_probe_data();

            let parameters = RandomForestRegressorParameters::default()
                .with_n_trees(2000)
                .with_m(m)
                .with_min_samples_leaf(1)
                .with_min_samples_split(2)
                .with_keep_samples(true)
                .with_seed(42);
            let forest = match sample_weights {
                None => RandomForestRegressor::fit(&x, &y, parameters).unwrap(),
                Some(sample_weights) => {
                    RandomForestRegressor::fit_with_weights(&x, &y, sample_weights, parameters)
                        .unwrap()
                }
            };

            let y_hat: Vec<f64> = forest.predict(&x).unwrap();
            assert_close_to_sklearn(&y_hat, train_ref, tol, "train");

            let y_hat_probe: Vec<f64> = forest.predict(&x_probe).unwrap();
            assert_close_to_sklearn(&y_hat_probe, probe_ref, tol, "probe");

            let y_hat_oob: Vec<f64> = forest.predict_oob(&x).unwrap();
            assert_close_to_sklearn(&y_hat_oob, oob_ref, oob_tol, "oob");
        }

        #[test]
        fn sklearn_parity_all_features() {
            // sklearn max_features = 1.0. Largest std of one run: train 0.0132, probe 0.0097,
            // oob 0.0290.
            let train_ref = [
                -0.563215, 0.790441, 0.671600, 0.447413, -0.962337, 1.182714, 0.146502, 0.956499,
                -0.019351, 0.475851, 0.536221, -0.921016, 0.566884, -1.182586, -0.953829,
                -0.492116, -0.970510, 0.906254, -0.122531, 0.492432, 0.111251, 0.701316, 0.166136,
                -0.090198, 0.168000, 0.917484, -0.318774, 0.452445, 0.701507, 0.213636, 0.519599,
                0.777545, 0.710208, -1.026581, 0.428588, 0.814387, -0.550148, -0.449774, -0.966291,
                0.151257,
            ];
            let probe_ref = [
                0.840299, 0.478916, -0.941862, -0.540659, 0.284103, -0.685080, -0.272447, 0.791806,
                -0.562334, -0.195136,
            ];
            let oob_ref = [
                -0.649343, 0.431595, 0.475531, 0.590899, -1.064646, 0.606166, 0.261989, 0.535072,
                0.255804, 0.436133, 0.880022, -0.782491, 0.635705, -0.820807, -0.774075, -0.607414,
                -0.781085, 0.788450, -0.119615, 0.437044, -0.173755, 0.739059, 0.160968, 0.594925,
                0.542656, 0.672530, -0.189747, 0.961226, 0.830511, 0.416988, 0.517550, 0.376026,
                0.616485, -0.942362, 0.465415, 0.488966, -0.554237, -0.643990, -0.784735, 0.372642,
            ];
            check_sklearn_parity(4, &train_ref, &probe_ref, &oob_ref, 0.06, 0.12, None);
        }

        #[test]
        fn sklearn_parity_two_features() {
            // sklearn max_features = 2. Largest std of one run: train 0.0193, probe 0.0137,
            // oob 0.0343.
            let train_ref = [
                -0.539971, 0.752384, 0.639444, 0.304022, -0.900858, 1.152436, 0.153326, 0.948274,
                -0.020744, 0.473991, 0.498276, -0.873499, 0.504429, -1.058965, -0.884246,
                -0.399683, -0.943511, 0.897507, -0.102778, 0.483708, 0.157492, 0.602244, 0.124176,
                -0.134138, 0.133388, 0.902843, -0.300226, 0.324471, 0.626043, 0.099969, 0.506475,
                0.816064, 0.690784, -0.939260, 0.429617, 0.800137, -0.544917, -0.442264, -0.920062,
                0.086742,
            ];
            let probe_ref = [
                0.819729, 0.481021, -0.705066, -0.572872, 0.188570, -0.723649, -0.302799, 0.719386,
                -0.522029, -0.210053,
            ];
            let oob_ref = [
                -0.584733, 0.327078, 0.387249, 0.195302, -0.898731, 0.523206, 0.280689, 0.512263,
                0.251790, 0.431326, 0.776490, -0.653868, 0.466336, -0.482361, -0.578605, -0.352370,
                -0.705269, 0.764271, -0.065520, 0.412983, -0.046812, 0.460163, 0.044958, 0.472574,
                0.448485, 0.631409, -0.138894, 0.607838, 0.623829, 0.106691, 0.481788, 0.481038,
                0.563559, -0.701948, 0.468302, 0.449402, -0.539726, -0.623343, -0.657852, 0.195843,
            ];
            check_sklearn_parity(2, &train_ref, &probe_ref, &oob_ref, 0.08, 0.14, None);
        }

        #[test]
        fn sklearn_parity_with_weights() {
            /* train: max_std=0.01455 max_dev=0.03734
             * probe: max_std=0.01280 max_dev=0.02387
             * oob: max_std=0.03646 max_dev=0.07464
             */
            let train_ref = [
                -0.579075, 0.757098, 0.681036, 0.422236, -0.966882, 1.086633, 0.123891, 1.069910,
                0.012417, 0.497026, 0.505539, -0.949550, 0.585693, -1.119453, -0.980359, -0.468459,
                -0.905580, 0.871368, -0.134145, 0.509760, 0.091451, 0.714195, 0.177695, -0.283949,
                0.336420, 0.896071, -0.324312, 0.298485, 0.762092, 0.175884, 0.548944, 0.884472,
                0.648075, -1.009675, 0.436657, 0.911007, -0.571345, -0.498412, -0.983307, 0.096344,
            ];
            let probe_ref = [
                0.894188, 0.477767, -0.950106, -0.579763, 0.357390, -0.654198, -0.218833, 0.869894,
                -0.585196, -0.204378,
            ];
            let oob_ref = [
                -0.610721, 0.467067, 0.441084, 0.654051, -0.999940, 0.545649, 0.227121, 0.542453,
                0.106474, 0.495335, 0.906098, -0.727918, 0.614949, -0.778016, -0.804240, -0.643123,
                -0.821945, 0.744144, -0.158162, 0.453301, -0.000066, 0.756773, 0.198170, 0.497569,
                0.524053, 0.703722, -0.164544, 0.868727, 0.828938, 0.275783, 0.615440, 0.383522,
                0.591260, -0.928361, 0.504742, 0.564432, -0.583036, -0.695436, -0.786008, 0.396527,
            ];

            check_sklearn_parity(
                4,
                &train_ref,
                &probe_ref,
                &oob_ref,
                0.06,
                0.20, // set a little high
                Some(&sklearn_sample_weights()),
            );
        }

        #[test]
        fn sklearn_parity_with_weights_two_features() {
            /*
             * train: max_std=0.01686 max_dev=0.03435
             * probe: max_std=0.01493 max_dev=0.03213
             * oob: max_std=0.05181 max_dev=0.08933
             */
            let train_ref = [
                -0.550424, 0.716045, 0.669342, 0.340627, -0.835593, 1.050114, 0.146434, 1.065546,
                0.013482, 0.465432, 0.480859, -0.924076, 0.454231, -0.991295, -0.909932, -0.425557,
                -0.850313, 0.847289, -0.111871, 0.509527, 0.165458, 0.572463, 0.135126, -0.302609,
                0.223818, 0.860349, -0.291573, 0.228765, 0.650378, 0.094705, 0.515905, 0.908350,
                0.658607, -0.858620, 0.452023, 0.898403, -0.559861, -0.491019, -0.931630, 0.063906,
            ];
            let probe_ref = [
                0.852481, 0.503229, -0.635851, -0.585194, 0.276241, -0.719231, -0.182757, 0.771101,
                -0.488099, -0.207485,
            ];
            let oob_ref = [
                -0.568080, 0.376018, 0.402009, 0.235654, -0.801487, 0.462343, 0.303037, 0.520005,
                0.108155, 0.424214, 0.823031, -0.595168, 0.418338, -0.488094, -0.562944, -0.421132,
                -0.739436, 0.690178, -0.082967, 0.451987, 0.110805, 0.435434, 0.055336, 0.401684,
                0.356611, 0.624397, -0.056188, 0.503624, 0.661848, 0.093107, 0.503889, 0.501511,
                0.606863, -0.587745, 0.555991, 0.500660, -0.565841, -0.678327, -0.617068, 0.229146,
            ];

            check_sklearn_parity(
                2,
                &train_ref,
                &probe_ref,
                &oob_ref,
                0.064,
                0.2,
                Some(&sklearn_sample_weights()),
            );
        }
    }
}
