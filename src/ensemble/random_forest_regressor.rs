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

        // Due to bootstrapping, weighted value should be well above 9.
        for p in y_hat.iter() {
            assert!(p > &9.0f64, "expected value greater than 9, got {p}");
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
}
