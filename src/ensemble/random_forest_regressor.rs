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
//! use smartcore::linalg::naive::dense_matrix::*;
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

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::default::Default;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::{Failed, FailedError};
use crate::linalg::{BaseMatrix, Matrix};
use crate::math::num::RealNumber;
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
pub struct RandomForestRegressor<T: RealNumber> {
    _parameters: RandomForestRegressorParameters,
    trees: Vec<DecisionTreeRegressor<T>>,
    samples: Option<Vec<Vec<bool>>>,
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

impl<T: RealNumber> PartialEq for RandomForestRegressor<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.trees.len() != other.trees.len() {
            false
        } else {
            for i in 0..self.trees.len() {
                if self.trees[i] != other.trees[i] {
                    return false;
                }
            }
            true
        }
    }
}

impl<T: RealNumber, M: Matrix<T>>
    SupervisedEstimator<M, M::RowVector, RandomForestRegressorParameters>
    for RandomForestRegressor<T>
where
    <M as BaseMatrix<T>>::RowVector: Sync + Send,
    M: std::marker::Sync,
{
    fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: RandomForestRegressorParameters,
    ) -> Result<Self, Failed> {
        RandomForestRegressor::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>> Predictor<M, M::RowVector> for RandomForestRegressor<T> {
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber> RandomForestRegressor<T> {
    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        parameters: RandomForestRegressorParameters,
    ) -> Result<RandomForestRegressor<T>, Failed>
    where
        <M as BaseMatrix<T>>::RowVector: Sync + Send,
        M: std::marker::Sync,
    {
        let (n_rows, num_attributes) = x.shape();

        let mtry = parameters
            .m
            .unwrap_or((num_attributes as f64).sqrt().floor() as usize);

        let tree_sample_pairs = RandomForestRegressor::<T>::collect_tree_sample_pairs(
            x,
            y,
            parameters.clone(),
            n_rows,
            mtry,
        );

        let (trees, samples) =
            RandomForestRegressor::<T>::parse_tree_sample_pairs(tree_sample_pairs);
        Ok(RandomForestRegressor {
            _parameters: parameters,
            trees,
            samples,
        })
    }

    fn collect_tree_sample_pairs<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        parameters: RandomForestRegressorParameters,
        n_rows: usize,
        mtry: usize,
    ) -> Vec<(DecisionTreeRegressor<T>, Option<Vec<bool>>)>
    where
        <M as BaseMatrix<T>>::RowVector: Sync + Send,
        M: std::marker::Sync,
    {
        (0..parameters.n_trees)
            .into_par_iter()
            .map(|tree_number| {
                let params = DecisionTreeRegressorParameters {
                    max_depth: parameters.max_depth,
                    min_samples_leaf: parameters.min_samples_leaf,
                    min_samples_split: parameters.min_samples_split,
                };

                let mut rng = StdRng::seed_from_u64(parameters.seed + tree_number as u64);
                let samples = RandomForestRegressor::<T>::sample_with_replacement(n_rows, &mut rng);
                let relevant_samples: Option<Vec<bool>> = match parameters.keep_samples {
                    true => Some(samples.iter().map(|x| *x != 0).collect()),
                    false => None,
                };

                (
                    DecisionTreeRegressor::fit_weak_learner(x, y, samples, mtry, params, &mut rng)
                        .unwrap(),
                    relevant_samples,
                )
            })
            .collect()
    }

    fn parse_tree_sample_pairs(
        tree_sample_pairs: Vec<(DecisionTreeRegressor<T>, Option<Vec<bool>>)>,
    ) -> (Vec<DecisionTreeRegressor<T>>, Option<Vec<Vec<bool>>>) {
        let mut trees = vec![];
        let mut samples = vec![];
        tree_sample_pairs
            .into_iter()
            .for_each(|(tree, samples_for_tree)| {
                trees.push(tree);
                if samples_for_tree.is_some() {
                    samples.push(samples_for_tree.unwrap());
                }
            });
        let samples = match samples.len() {
            0 => None,
            _ => Some(samples),
        };
        (trees, samples)
    }

    /// Predict class for `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict<M: Matrix<T>>(&self, x: &M) -> Result<M::RowVector, Failed> {
        let mut result = M::zeros(1, x.shape().0);

        let (n, _) = x.shape();

        for i in 0..n {
            result.set(0, i, self.predict_for_row(x, i));
        }

        Ok(result.to_row_vector())
    }

    fn predict_for_row<M: Matrix<T>>(&self, x: &M, row: usize) -> T {
        let n_trees = self.trees.len();

        let mut result = T::zero();

        for tree in self.trees.iter() {
            result += tree.predict_for_row(x, row);
        }

        result / T::from(n_trees).unwrap()
    }

    /// Predict OOB classes for `x`. `x` is expected to be equal to the dataset used in training.
    pub fn predict_oob<M: Matrix<T>>(&self, x: &M) -> Result<M::RowVector, Failed> {
        let (n, _) = x.shape();
        if self.samples.is_none() {
            Err(Failed::because(
                FailedError::PredictFailed,
                "Need samples=true for OOB predictions.",
            ))
        } else if self.samples.as_ref().unwrap()[0].len() != n {
            Err(Failed::because(
                FailedError::PredictFailed,
                "Prediction matrix must match matrix used in training for OOB predictions.",
            ))
        } else {
            let mut result = M::zeros(1, n);

            for i in 0..n {
                result.set(0, i, self.predict_for_row_oob(x, i));
            }

            Ok(result.to_row_vector())
        }
    }

    fn predict_for_row_oob<M: Matrix<T>>(&self, x: &M, row: usize) -> T {
        let mut n_trees = 0;
        let mut result = T::zero();

        for (tree, samples) in self.trees.iter().zip(self.samples.as_ref().unwrap()) {
            if !samples[row] {
                result += tree.predict_for_row(x, row);
                n_trees += 1;
            }
        }

        // TODO: What to do if there are no oob trees?
        result / T::from(n_trees).unwrap()
    }

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
    use crate::linalg::naive::dense_matrix::DenseMatrix;
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        ]);
        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let regressor = RandomForestRegressor::fit(
            &x,
            &y,
            RandomForestRegressorParameters {
                max_depth: None,
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

        assert!(mean_absolute_error(&y, &y_hat) < mean_absolute_error(&y, &y_hat_oob));
    }

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

        let deserialized_forest: RandomForestRegressor<f64> =
            bincode::deserialize(&bincode::serialize(&forest).unwrap()).unwrap();

        assert_eq!(forest, deserialized_forest);
    }
}
