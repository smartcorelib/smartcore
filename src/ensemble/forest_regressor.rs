use rand::Rng;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;

use crate::rand_custom::get_rng_impl;
use crate::tree::decision_tree_regressor::{
    DecisionTreeRegressor, DecisionTreeRegressorParameters,
};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// Parameters of the Forest Regressor
/// Some parameters here are passed directly into base estimator.
pub struct ForestRegressorParameters {
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
    #[cfg_attr(feature = "serde", serde(default))]
    pub bootstrap: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    pub use_random_splits: bool,
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for ForestRegressor<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        if self.trees.as_ref().unwrap().len() != other.trees.as_ref().unwrap().len() {
            false
        } else {
            self.trees
                .iter()
                .zip(other.trees.iter())
                .all(|(a, b)| a == b)
        }
    }
}

/// Forest Regressor
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct ForestRegressor<
    TX: Number + FloatNumber + PartialOrd,
    TY: Number,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    trees: Option<Vec<DecisionTreeRegressor<TX, TY, X, Y>>>,
    samples: Option<Vec<Vec<bool>>>,
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    ForestRegressor<TX, TY, X, Y>
{
    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: ForestRegressorParameters,
    ) -> Result<ForestRegressor<TX, TY, X, Y>, Failed> {
        let (n_rows, num_attributes) = x.shape();

        if n_rows != y.shape() {
            return Err(Failed::fit("Number of rows in X should = len(y)"));
        }

        let mtry = parameters
            .m
            .unwrap_or((num_attributes as f64).sqrt().floor() as usize);

        let mut rng = get_rng_impl(Some(parameters.seed));
        let mut trees: Vec<DecisionTreeRegressor<TX, TY, X, Y>> = Vec::new();

        let mut maybe_all_samples: Option<Vec<Vec<bool>>> = Option::None;
        if parameters.keep_samples {
            // TODO: use with_capacity here
            maybe_all_samples = Some(Vec::new());
        }

        let mut samples: Vec<usize> = (0..n_rows).map(|_| 1).collect();

        for _ in 0..parameters.n_trees {
            if parameters.bootstrap {
                samples =
                    ForestRegressor::<TX, TY, X, Y>::sample_with_replacement(n_rows, &mut rng);
            }

            // keep samples is flag is on
            if let Some(ref mut all_samples) = maybe_all_samples {
                all_samples.push(samples.iter().map(|x| *x != 0).collect())
            }

            let params = DecisionTreeRegressorParameters {
                max_depth: parameters.max_depth,
                min_samples_leaf: parameters.min_samples_leaf,
                min_samples_split: parameters.min_samples_split,
                seed: Some(parameters.seed),
            };
            let tree = DecisionTreeRegressor::fit_weak_learner(
                x,
                y,
                samples.clone(),
                mtry,
                params,
                parameters.use_random_splits,
            )?;
            trees.push(tree);
        }

        Ok(ForestRegressor {
            trees: Some(trees),
            samples: maybe_all_samples,
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
        let n_trees = self.trees.as_ref().unwrap().len();

        let mut result = TY::zero();

        for tree in self.trees.as_ref().unwrap().iter() {
            result += tree.predict_for_row(x, row);
        }

        result / TY::from_usize(n_trees).unwrap()
    }

    /// Predict OOB classes for `x`. `x` is expected to be equal to the dataset used in training.
    pub fn predict_oob(&self, x: &X) -> Result<Y, Failed> {
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
            let mut result = Y::zeros(n);

            for i in 0..n {
                result.set(i, self.predict_for_row_oob(x, i));
            }

            Ok(result)
        }
    }

    fn predict_for_row_oob(&self, x: &X, row: usize) -> TY {
        let mut n_trees = 0;
        let mut result = TY::zero();

        for (tree, samples) in self
            .trees
            .as_ref()
            .unwrap()
            .iter()
            .zip(self.samples.as_ref().unwrap())
        {
            if !samples[row] {
                result += tree.predict_for_row(x, row);
                n_trees += 1;
            }
        }

        // TODO: What to do if there are no oob trees?
        result / TY::from(n_trees).unwrap()
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
