use rand::RngExt;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::MutArrayView1;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;

use crate::rand_custom::get_rng_impl;
use crate::tree::base_tree_regressor::{BaseTreeRegressor, BaseTreeRegressorParameters, Splitter};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// Parameters of the Forest Regressor
/// Some parameters here are passed directly into base estimator.
#[must_use]
pub struct BaseForestRegressorParameters {
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
    pub splitter: Splitter,
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for BaseForestRegressor<TX, TY, X, Y>
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
pub struct BaseForestRegressor<
    TX: Number + FloatNumber + PartialOrd,
    TY: Number,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    trees: Option<Vec<BaseTreeRegressor<TX, TY, X, Y>>>,
    samples: Option<Vec<Vec<bool>>>,
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    BaseForestRegressor<TX, TY, X, Y>
{
    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    /// * `sample_weights`: optional sample_weights to use during fitting
    pub fn fit(
        x: &X,
        y: &Y,
        sample_weights: Option<&[f64]>,
        parameters: BaseForestRegressorParameters,
    ) -> Result<BaseForestRegressor<TX, TY, X, Y>, Failed> {
        let (n_rows, num_attributes) = x.shape();

        if n_rows != y.shape() {
            return Err(Failed::fit("Number of rows in X should = len(y)"));
        }
        if n_rows == 0 || num_attributes == 0 {
            return Err(Failed::because(
                FailedError::ParametersError,
                "Training data must contain at least one sample and one feature.",
            ));
        }

        let mtry = parameters
            .m
            .unwrap_or((num_attributes as f64).sqrt().floor() as usize);

        let mut rng = get_rng_impl(Some(parameters.seed));
        let n_trees = parameters.n_trees;
        let mut trees: Vec<BaseTreeRegressor<TX, TY, X, Y>> = Vec::with_capacity(n_trees);

        let mut maybe_all_samples: Option<Vec<Vec<bool>>> = Option::None;
        if parameters.keep_samples {
            maybe_all_samples = Some(Vec::with_capacity(n_trees));
        }

        let mut samples: Vec<usize> = (0..n_rows).map(|_| 1).collect();

        let dist = sample_weights
            .map(|weights| {
                rand::distr::weighted::WeightedIndex::new(weights)
                    .map_err(|e| Failed::fit(&e.to_string()))
            })
            .transpose()?;

        // Compute the order of each attribute once
        let mut order: Vec<Vec<usize>> = Vec::with_capacity(num_attributes);

        for i in 0..num_attributes {
            let mut col_i: Vec<TX> = x.get_col(i).iterator(0).copied().collect();
            order.push(col_i.argsort_mut());
        }

        for tree_idx in 0..parameters.n_trees {
            if parameters.bootstrap {
                samples = BaseForestRegressor::<TX, TY, X, Y>::sample_with_replacement(
                    n_rows,
                    &mut rng,
                    dist.as_ref(),
                );
            }

            // keep samples is flag is on
            if let Some(ref mut all_samples) = maybe_all_samples {
                all_samples.push(samples.iter().map(|x| *x != 0).collect())
            }

            let params = BaseTreeRegressorParameters {
                max_depth: parameters.max_depth,
                min_samples_leaf: parameters.min_samples_leaf,
                min_samples_split: parameters.min_samples_split,
                seed: Some(parameters.seed.wrapping_add(tree_idx as u64)), // give each tree its own fixed seed
                splitter: parameters.splitter,
            };
            // Only use sample weights on base tree if not already applied during bootstrapping
            let sample_weights_for_base_tree = if parameters.bootstrap {
                None
            } else {
                sample_weights
            };
            let tree = BaseTreeRegressor::fit_weak_learner(
                x,
                y,
                sample_weights_for_base_tree,
                samples.clone(),
                mtry,
                &order,
                params,
            )?;
            trees.push(tree);
        }

        Ok(BaseForestRegressor {
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

        let samples = match &self.samples {
            Some(s) => s,
            None => {
                return Err(Failed::because(
                    FailedError::PredictFailed,
                    "Need samples=true for OOB predictions.",
                ));
            }
        };

        if samples[0].len() != n {
            return Err(Failed::because(
                FailedError::PredictFailed,
                "Prediction matrix must match matrix used in training for OOB predictions.",
            ));
        }

        let mut result = Y::zeros(n);

        for i in 0..n {
            result.set(i, self.predict_for_row_oob(x, i));
        }

        Ok(result)
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

    fn sample_with_replacement(
        nrows: usize,
        rng: &mut impl rand::Rng,
        distribution: Option<&rand::distr::weighted::WeightedIndex<f64>>,
    ) -> Vec<usize> {
        let mut samples = vec![0; nrows];
        for _ in 0..nrows {
            let xi = match distribution {
                Some(dist) => rng.sample(dist),
                None => rng.random_range(0..nrows),
            };
            samples[xi] += 1;
        }

        samples
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn test_base_forest_regressor_keep_samples() {
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]).unwrap();
        let y = vec![1.0, 2.0, 3.0];
        let params = BaseForestRegressorParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 5,
            m: None,
            keep_samples: true,
            seed: 42,
            bootstrap: true,
            splitter: crate::tree::base_tree_regressor::Splitter::Best,
        };
        let regressor = BaseForestRegressor::fit(&x, &y, None, params).unwrap();
        assert_eq!(regressor.trees.unwrap().len(), 5);
        assert!(regressor.samples.is_some());
    }

    #[test]
    fn test_fit_on_empty_data_returns_error() {
        // 2 rows x 2 features — values are arbitrary; only the empty-row case is under test
        let full = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let empty = full.take(&[] as &[usize], 0);
        assert_eq!(empty.shape(), (0, 2));

        let y: Vec<f64> = vec![];
        let result = BaseForestRegressor::fit(
            &empty,
            &y,
            None,
            BaseForestRegressorParameters {
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 5,
                m: None,
                keep_samples: false,
                seed: 0,
                bootstrap: true,
                splitter: crate::tree::base_tree_regressor::Splitter::Best,
            },
        );
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error(), FailedError::ParametersError);
    }

    #[test]
    fn test_fit_on_zero_features_returns_error() {
        // 2 rows x 2 features — values are arbitrary; only the zero-feature case is under test
        let full = DenseMatrix::from_2d_vec(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let no_features = full.take(&[] as &[usize], 1);
        assert_eq!(no_features.shape(), (2, 0));

        let y: Vec<f64> = vec![1.0, 2.0];
        let result = BaseForestRegressor::fit(
            &no_features,
            &y,
            None,
            BaseForestRegressorParameters {
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 5,
                m: None,
                keep_samples: false,
                seed: 0,
                bootstrap: true,
                splitter: crate::tree::base_tree_regressor::Splitter::Best,
            },
        );
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error(), FailedError::ParametersError);
    }

    #[test]
    fn balance_property() {
        // Test the balance property on a random dataset
        // Fit random forest with bootstrapping = False.
        // The (weighted) average of predictions on the training data should equal the
        // weighted average of the actual targets of the training data
        let x: DenseMatrix<f64> = DenseMatrix::rand(1000, 10);
        let model_parameters = (0..=9).map(|x| x as f64).collect::<Vec<_>>();
        let y: Vec<f64> = model_parameters.xa(true, &x);

        let forest_parameters = BaseForestRegressorParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 5,
            m: None,
            keep_samples: true,
            seed: 42,
            bootstrap: false,
            splitter: crate::tree::base_tree_regressor::Splitter::Best,
        };

        let forest = BaseForestRegressor::fit(&x, &y, None, forest_parameters.clone())
            .expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");
        assert!((y_hat.iter().sum::<f64>() - y.iter().sum::<f64>()).abs() < 1e-9);

        // Seeded RNG: the test gives the same result on each run
        let mut rng = get_rng_impl(Some(42));

        // Positive weights in [0.5, 2.0)
        let sample_weights: Vec<f64> = (0..1000).map(|_| rng.random_range(0.5..2.0)).collect();
        let forest =
            BaseForestRegressor::fit(&x, &y, Some(&sample_weights), forest_parameters.clone())
                .expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");

        let s: f64 = sample_weights.iter().sum();
        let normalized_weights = sample_weights.iter().map(|w| *w / s).collect::<Vec<_>>();

        let expected = y
            .iter()
            .zip(normalized_weights.iter())
            .map(|(&yi, &w)| yi * w)
            .sum::<f64>();
        let actual = y_hat
            .iter()
            .zip(normalized_weights.iter())
            .map(|(&yi, &w)| yi * w)
            .sum::<f64>();

        assert!((expected - actual).abs() < 1e-9);
    }

    #[test]
    fn test_each_tree_gets_different_bootstrap_sample() {
        // actual data uses is irrelevant
        let n_rows = 100;
        let x: DenseMatrix<f64> =
            DenseMatrix::from_iterator((0..2 * n_rows).map(|k| k as f64), n_rows, 2, 0);
        let y: Vec<f64> = (0..n_rows).map(|i| i as f64).collect();
        let sample_weights: Vec<f64> = (0..n_rows).map(|i| 1.0 + (i % 4) as f64).collect();

        for weights in [None, Some(sample_weights.as_slice())] {
            let params = BaseForestRegressorParameters {
                max_depth: Some(1),
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 10,
                m: None,
                keep_samples: true, // keep samples used for each tree, so we can check that they are different
                seed: 42,
                bootstrap: true, // Use bootstrapping
                splitter: crate::tree::base_tree_regressor::Splitter::Best,
            };
            let regressor = BaseForestRegressor::fit(&x, &y, weights, params).unwrap();
            let samples = regressor.samples.unwrap();

            for (t, in_bag) in samples.iter().enumerate() {
                assert!(
                    in_bag.iter().any(|b| !b),
                    "tree {t} has no out-of-bag rows (weights: {weights:?})"
                );
                for (u, other) in samples.iter().enumerate().skip(t + 1) {
                    assert_ne!(
                        in_bag, other,
                        "trees {t} and {u} have the same bootstrap sample (weights: {weights:?})"
                    );
                }
            }
        }
    }

    #[test]
    fn each_tree_gets_different_feature_sample() {
        // Without bootstrap, all trees get the same rows. With m = 1, each node uses one
        // random feature, thus the trees must be different. If all trees use the same seed,
        // all trees are equal. The Random splitter also uses this rng for the thresholds.

        // Create some data
        let n_rows = 30;
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator(
            (0..4 * n_rows).map(|k| ((k * 7919) % 101) as f64),
            n_rows,
            4,
            0,
        );
        let y: Vec<f64> = (0..n_rows).map(|i| ((i * 31) % 17) as f64).collect();

        for splitter in [Splitter::Best, Splitter::Random] {
            let params = BaseForestRegressorParameters {
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 10,
                m: Some(1),
                keep_samples: false,
                seed: 42,
                bootstrap: false,
                splitter: splitter.clone(),
            };
            let forest = BaseForestRegressor::fit(&x, &y, None, params).unwrap();
            let trees = forest.trees.unwrap();
            assert!(
                trees.iter().any(|tree| tree != &trees[0]),
                "all trees are equal (splitter: {splitter:?})"
            );
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

        let parameters = BaseForestRegressorParameters {
            max_depth: Some(0),
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 10,
            m: None,
            keep_samples: true,
            seed: 42,
            bootstrap: false, // No bootstrapping
            splitter: crate::tree::base_tree_regressor::Splitter::Best,
        };

        let forest = BaseForestRegressor::fit(&x, &y, Some(&sample_weights), parameters.clone())
            .expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");

        // weighted mean is (10.0 * 0 + 90*10) / 100 = 9
        for p in y_hat.iter() {
            assert!(
                (p - 9.0f64).abs() < 1e-9,
                "expected value very close to 9, got {p}"
            );
        }

        // Without weights, the predicted value should be close to 5
        let forest = BaseForestRegressor::fit(&x, &y, None, parameters).expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");

        for p in y_hat.iter() {
            assert!(
                (p - 5.0).abs() < 1e-9,
                "expected value very close to 5, got {p}"
            );
        }

        // Use bootstrapping
        let parameters = BaseForestRegressorParameters {
            max_depth: Some(0), // Match sibling test on RandomForestRegressor
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 500, // Use more trees than before to smooth out randomness
            m: None,
            keep_samples: true, // keep samples used for each tree, so we can check that they are different
            seed: 42,
            bootstrap: true, // Use bootstrapping
            splitter: crate::tree::base_tree_regressor::Splitter::Best,
        };

        let forest = BaseForestRegressor::fit(&x, &y, Some(&sample_weights), parameters.clone())
            .expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");

        // with bootstrapping the weight should be close to 9, but not extremely close
        for p in y_hat.iter() {
            assert!(
                (p - 9.0).abs() < 0.1,
                "expected value reasonably close to 9, got {p}"
            );
        }

        // Without weights, the predicted value should be reasonably close to 5
        let forest = BaseForestRegressor::fit(&x, &y, None, parameters).expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");

        // The bootstrapping makes that we will not be very close to 5
        for p in y_hat.iter() {
            assert!(
                (p - 5.0).abs() < 0.1,
                "expected value reasonably close to 5, got {p}"
            );
        }
    }

    #[test]
    fn fit_twice_with_same_seed_gives_identical_forest() {
        // With bootstrap, m = 1 and the Random splitter, each random path is used:
        // bootstrap sample, feature selection and random thresholds.
        let n_rows = 30;
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator(
            (0..4 * n_rows).map(|k| ((k * 7919) % 101) as f64),
            n_rows,
            4,
            0,
        );
        let y: Vec<f64> = (0..n_rows).map(|i| ((i * 31) % 17) as f64).collect();
        let sample_weights: Vec<f64> = (0..n_rows).map(|i| 1.0 + (i % 4) as f64).collect();

        for splitter in [Splitter::Best, Splitter::Random] {
            for weights in [None, Some(sample_weights.as_slice())] {
                let params = BaseForestRegressorParameters {
                    max_depth: None,
                    min_samples_leaf: 1,
                    min_samples_split: 2,
                    n_trees: 10,
                    m: Some(1),
                    keep_samples: true,
                    seed: 42,
                    bootstrap: true,
                    splitter: splitter.clone(),
                };

                let forest_a = BaseForestRegressor::fit(&x, &y, weights, params.clone())
                    .expect("Fit should work");
                let forest_b = BaseForestRegressor::fit(&x, &y, weights, params.clone())
                    .expect("Fit should work");

                assert_eq!(
                    forest_a, forest_b,
                    "forests differ (splitter: {splitter:?}, weights: {weights:?})"
                );
                assert_eq!(
                    forest_a.samples, forest_b.samples,
                    "bootstrap samples differ (splitter: {splitter:?}, weights: {weights:?})"
                );
                assert_eq!(
                    forest_a.predict(&x).unwrap(),
                    forest_b.predict(&x).unwrap(),
                    "predictions differ (splitter: {splitter:?}, weights: {weights:?})"
                );

                // A different seed must give a different forest, else the check above is trivial
                let forest_c = BaseForestRegressor::fit(
                    &x,
                    &y,
                    weights,
                    BaseForestRegressorParameters { seed: 43, ..params },
                )
                .expect("Fit should work");
                assert_ne!(
                    forest_a, forest_c,
                    "seeds 42 and 43 give the same forest (splitter: {splitter:?}, weights: {weights:?})"
                );
            }
        }
    }
}
