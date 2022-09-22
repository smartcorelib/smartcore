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
//! use smartcore::linalg::dense::matrix::DenseMatrix;
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
//!         ]);
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

use rand::Rng;
use std::default::Default;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::{Failed, FailedError};
use crate::linalg::base::{Array1, Array2};
use crate::num::Number;
use crate::tree::decision_tree_regressor::{
    DecisionTreeRegressor, DecisionTreeRegressorParameters,
};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// Parameters of the Random Forest Regressor
/// Some parameters here are passed directly into base estimator.
pub struct RandomForestRegressorParameters {
    /// Tree max depth. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub max_depth: Option<u16>,
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_leaf: usize,
    /// The minimum number of samples required to split an internal node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_split: usize,
    /// The number of trees in the forest.
    pub n_trees: usize,
    /// Number of random sample of predictors to use as split candidates.
    pub m: Option<usize>,
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub keep_samples: bool,
    /// Seed used for bootstrap sampling and feature selection for each tree.
    pub seed: u64,
}

/// Random Forest Regressor
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct RandomForestRegressor<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
{
    parameters: RandomForestRegressorParameters,
    trees: Vec<DecisionTreeRegressor<TX, TY, X, Y>>,
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
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 10,
            m: Option::None,
            keep_samples: false,
            seed: 0,
        }
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for RandomForestRegressor<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        if self.trees.len() != other.trees.len() {
            false
        } else {
            self.trees
                .iter()
                .zip(other.trees.iter())
                .all(|(a, b)| a == b)
        }
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, RandomForestRegressorParameters>
    for RandomForestRegressor<TX, TY, X, Y>
{
    fn fit(x: &X, y: &Y, parameters: RandomForestRegressorParameters) -> Result<Self, Failed> {
        RandomForestRegressor::fit(x, y, parameters)
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> Predictor<X, Y>
    for RandomForestRegressor<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
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
        let (n_rows, num_attributes) = x.shape();

        let mtry = parameters
            .m
            .unwrap_or((num_attributes as f64).sqrt().floor() as usize);

        let mut trees: Vec<DecisionTreeRegressor<TX, TY, X, Y>> = Vec::new();

        for _ in 0..parameters.n_trees {
            let samples = RandomForestRegressor::<TX, TY, X, Y>::sample_with_replacement(
                n_rows, &mut rand::thread_rng());
            let params = DecisionTreeRegressorParameters {
                max_depth: parameters.max_depth,
                min_samples_leaf: parameters.min_samples_leaf,
                min_samples_split: parameters.min_samples_split,
            };
            let tree =
                DecisionTreeRegressor::fit_weak_learner(
                    x, y, samples, mtry, params)?;
            trees.push(tree);
        }

        Ok(RandomForestRegressor {
            parameters: parameters,
            trees,
        })
    }

    /// Predict class for `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let mut result = Y::zeros(x.shape().0);

        let (n, _) = x.shape();

        for i in 0..n {
            result.set(i, self.predict_for_row(x, i));
        }

        Ok(result)
    }

    fn predict_for_row(&self, x: &X, row: usize) -> TY {
        let n_trees = self.trees.len();

        let mut result = TY::zero();

        for tree in self.trees.iter() {
            result += tree.predict_for_row(x, row);
        }

        result / TY::from_usize(n_trees).unwrap()
    }

    /// Predict OOB classes for `x`. `x` is expected to be equal to the dataset used in training.
    // pub fn predict_oob(&self, x: &X) -> Result<Y, Failed> {
    //     let (n, _) = x.shape();
    //     if self.samples.is_none() {
    //         Err(Failed::because(
    //             FailedError::PredictFailed,
    //             "Need samples=true for OOB predictions.",
    //         ))
    //     } else if self.samples.as_ref().unwrap()[0].len() != n {
    //         Err(Failed::because(
    //             FailedError::PredictFailed,
    //             "Prediction matrix must match matrix used in training for OOB predictions.",
    //         ))
    //     } else {
    //         let mut result = X::zeros(1, n);

    //         for i in 0..n {
    //             // TODO: fix this call
    //             //result.set(0, i, self.predict_for_row_oob(x, i));
    //         }

    //         Ok(result.to_row_vector())
    //     }
    // }

    // fn predict_for_row_oob(&self, x: &X, row: usize) -> X {
    //     let mut n_trees = 0;
    //     let mut result = TX::zero();

    //     for (tree, samples) in self.trees.iter().zip(self.samples.as_ref().unwrap()) {
    //         if !samples[row] {
    //             result += tree.predict_for_row(x, row);
    //             n_trees += 1;
    //         }
    //     }

    //     // TODO: What to do if there are no oob trees?
    //     result / TX::from_i32(n_trees).unwrap()
    // }

    fn sample_with_replacement(nrows: usize, rng: &mut impl Rng) -> Vec<usize> {
        let mut samples = vec![0; nrows];
        for _ in 0..nrows {
            let xi = rng.gen_range(0..nrows);
            samples[xi] += 1;
        }
        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::dense::matrix::DenseMatrix;
    use crate::metrics::mean_absolute_error;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        ]);
        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let y_hat = RandomForestRegressor::fit(
            &x,
            &y,
            RandomForestRegressorParameters {
                max_depth: None,
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

    // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    // #[test]
    // fn fit_predict_longley_oob() {
    //     let x = DenseMatrix::from_2d_array(&[
    //         &[234.289, 235.6, 159., 107.608, 1947., 60.323],
    //         &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
    //         &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
    //         &[284.599, 335.1, 165., 110.929, 1950., 61.187],
    //         &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
    //         &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
    //         &[365.385, 187., 354.7, 115.094, 1953., 64.989],
    //         &[363.112, 357.8, 335., 116.219, 1954., 63.761],
    //         &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
    //         &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
    //         &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
    //         &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
    //         &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
    //         &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
    //         &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
    //         &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
    //     ]);
    //     let y = vec![
    //         83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
    //         114.2, 115.7, 116.9,
    //     ];

    //     let regressor = RandomForestRegressor::fit(
    //         &x,
    //         &y,
    //         RandomForestRegressorParameters {
    //             max_depth: None,
    //             min_samples_leaf: 1,
    //             min_samples_split: 2,
    //             n_trees: 1000,
    //             m: Option::None,
    //             keep_samples: true,
    //             seed: 87,
    //         },
    //     )
    //     .unwrap();

    //     let y_hat = regressor.predict(&x).unwrap();
    //     let y_hat_oob = regressor.predict_oob(&x).unwrap();

    //     assert!(mean_absolute_error(&y, &y_hat) < mean_absolute_error(&y, &y_hat_oob));
    // }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        ]);
        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let forest = RandomForestRegressor::fit(&x, &y, Default::default()).unwrap();

        let deserialized_forest: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            bincode::deserialize(&bincode::serialize(&forest).unwrap()).unwrap();

        assert_eq!(forest, deserialized_forest);
    }
}
