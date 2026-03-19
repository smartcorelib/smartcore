//! # Random Forest Classifier
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
//! use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
//!
//! // Iris dataset
//! let x = DenseMatrix::from_2d_array(&[
//!              &[5.1, 3.5, 1.4, 0.2],
//!              &[4.9, 3.0, 1.4, 0.2],
//!              &[4.7, 3.2, 1.3, 0.2],
//!              &[4.6, 3.1, 1.5, 0.2],
//!              &[5.0, 3.6, 1.4, 0.2],
//!              &[5.4, 3.9, 1.7, 0.4],
//!              &[4.6, 3.4, 1.4, 0.3],
//!              &[5.0, 3.4, 1.5, 0.2],
//!              &[4.4, 2.9, 1.4, 0.2],
//!              &[4.9, 3.1, 1.5, 0.1],
//!              &[7.0, 3.2, 4.7, 1.4],
//!              &[6.4, 3.2, 4.5, 1.5],
//!              &[6.9, 3.1, 4.9, 1.5],
//!              &[5.5, 2.3, 4.0, 1.3],
//!              &[6.5, 2.8, 4.6, 1.5],
//!              &[5.7, 2.8, 4.5, 1.3],
//!              &[6.3, 3.3, 4.7, 1.6],
//!              &[4.9, 2.4, 3.3, 1.0],
//!              &[6.6, 2.9, 4.6, 1.3],
//!              &[5.2, 2.7, 3.9, 1.4],
//!         ]).unwrap();
//! let y = vec![
//!              0, 0, 0, 0, 0, 0, 0, 0,
//!              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//!         ];
//!
//! let classifier = RandomForestClassifier::fit(&x, &y, Default::default()).unwrap();
//! let y_hat = classifier.predict(&x).unwrap(); // use the same data for prediction
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
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;

use crate::rand_custom::get_rng_impl;
use crate::tree::decision_tree_classifier::{
    which_max, DecisionTreeClassifier, DecisionTreeClassifierParameters, SplitCriterion,
};

/// Parameters of the Random Forest algorithm.
/// Some parameters here are passed directly into base estimator.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct RandomForestClassifierParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Split criteria to use when building a tree. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub criterion: SplitCriterion,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Tree max depth. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub max_depth: Option<u16>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub min_samples_leaf: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to split an internal node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub min_samples_split: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The number of trees in the forest.
    pub n_trees: u16,
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

/// Random Forest Classifier
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct RandomForestClassifier<
    TX: Number + FloatNumber + PartialOrd,
    TY: Number + Ord,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    trees: Option<Vec<DecisionTreeClassifier<TX, TY, X, Y>>>,
    classes: Option<Vec<TY>>,
    samples: Option<Vec<Vec<bool>>>,
}

impl RandomForestClassifierParameters {
    /// Split criteria to use when building a tree. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }
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
    pub fn with_n_trees(mut self, n_trees: u16) -> Self {
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

impl<TX: Number + FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    PartialEq for RandomForestClassifier<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        if self.classes.as_ref().unwrap().len() != other.classes.as_ref().unwrap().len()
            || self.trees.as_ref().unwrap().len() != other.trees.as_ref().unwrap().len()
        {
            false
        } else {
            self.classes
                .iter()
                .zip(other.classes.iter())
                .all(|(a, b)| a == b)
                && self
                    .trees
                    .iter()
                    .zip(other.trees.iter())
                    .all(|(a, b)| a == b)
        }
    }
}

impl Default for RandomForestClassifierParameters {
    fn default() -> Self {
        RandomForestClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: Option::None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 100,
            m: Option::None,
            keep_samples: false,
            seed: 0,
        }
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, RandomForestClassifierParameters>
    for RandomForestClassifier<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            trees: Option::None,
            classes: Option::None,
            samples: Option::None,
        }
    }
    fn fit(x: &X, y: &Y, parameters: RandomForestClassifierParameters) -> Result<Self, Failed> {
        RandomForestClassifier::fit(x, y, parameters)
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    Predictor<X, Y> for RandomForestClassifier<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

/// RandomForestClassifier grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct RandomForestClassifierSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Split criteria to use when building a tree. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub criterion: Vec<SplitCriterion>,
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
    pub n_trees: Vec<u16>,
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

/// RandomForestClassifier grid search iterator
pub struct RandomForestClassifierSearchParametersIterator {
    random_forest_classifier_search_parameters: RandomForestClassifierSearchParameters,
    current_criterion: usize,
    current_max_depth: usize,
    current_min_samples_leaf: usize,
    current_min_samples_split: usize,
    current_n_trees: usize,
    current_m: usize,
    current_keep_samples: usize,
    current_seed: usize,
}

impl IntoIterator for RandomForestClassifierSearchParameters {
    type Item = RandomForestClassifierParameters;
    type IntoIter = RandomForestClassifierSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        RandomForestClassifierSearchParametersIterator {
            random_forest_classifier_search_parameters: self,
            current_criterion: 0,
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

impl Iterator for RandomForestClassifierSearchParametersIterator {
    type Item = RandomForestClassifierParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_criterion
            == self
                .random_forest_classifier_search_parameters
                .criterion
                .len()
            && self.current_max_depth
                == self
                    .random_forest_classifier_search_parameters
                    .max_depth
                    .len()
            && self.current_min_samples_leaf
                == self
                    .random_forest_classifier_search_parameters
                    .min_samples_leaf
                    .len()
            && self.current_min_samples_split
                == self
                    .random_forest_classifier_search_parameters
                    .min_samples_split
                    .len()
            && self.current_n_trees
                == self
                    .random_forest_classifier_search_parameters
                    .n_trees
                    .len()
            && self.current_m == self.random_forest_classifier_search_parameters.m.len()
            && self.current_keep_samples
                == self
                    .random_forest_classifier_search_parameters
                    .keep_samples
                    .len()
            && self.current_seed == self.random_forest_classifier_search_parameters.seed.len()
        {
            return None;
        }

        let next = RandomForestClassifierParameters {
            criterion: self.random_forest_classifier_search_parameters.criterion
                [self.current_criterion]
                .clone(),
            max_depth: self.random_forest_classifier_search_parameters.max_depth
                [self.current_max_depth],
            min_samples_leaf: self
                .random_forest_classifier_search_parameters
                .min_samples_leaf[self.current_min_samples_leaf],
            min_samples_split: self
                .random_forest_classifier_search_parameters
                .min_samples_split[self.current_min_samples_split],
            n_trees: self.random_forest_classifier_search_parameters.n_trees[self.current_n_trees],
            m: self.random_forest_classifier_search_parameters.m[self.current_m],
            keep_samples: self.random_forest_classifier_search_parameters.keep_samples
                [self.current_keep_samples],
            seed: self.random_forest_classifier_search_parameters.seed[self.current_seed],
        };

        if self.current_criterion + 1
            < self
                .random_forest_classifier_search_parameters
                .criterion
                .len()
        {
            self.current_criterion += 1;
        } else if self.current_max_depth + 1
            < self
                .random_forest_classifier_search_parameters
                .max_depth
                .len()
        {
            self.current_criterion = 0;
            self.current_max_depth += 1;
        } else if self.current_min_samples_leaf + 1
            < self
                .random_forest_classifier_search_parameters
                .min_samples_leaf
                .len()
        {
            self.current_criterion = 0;
            self.current_max_depth = 0;
            self.current_min_samples_leaf += 1;
        } else if self.current_min_samples_split + 1
            < self
                .random_forest_classifier_search_parameters
                .min_samples_split
                .len()
        {
            self.current_criterion = 0;
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split += 1;
        } else if self.current_n_trees + 1
            < self
                .random_forest_classifier_search_parameters
                .n_trees
                .len()
        {
            self.current_criterion = 0;
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_n_trees += 1;
        } else if self.current_m + 1 < self.random_forest_classifier_search_parameters.m.len() {
            self.current_criterion = 0;
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_n_trees = 0;
            self.current_m += 1;
        } else if self.current_keep_samples + 1
            < self
                .random_forest_classifier_search_parameters
                .keep_samples
                .len()
        {
            self.current_criterion = 0;
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_n_trees = 0;
            self.current_m = 0;
            self.current_keep_samples += 1;
        } else if self.current_seed + 1 < self.random_forest_classifier_search_parameters.seed.len()
        {
            self.current_criterion = 0;
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_n_trees = 0;
            self.current_m = 0;
            self.current_keep_samples = 0;
            self.current_seed += 1;
        } else {
            self.current_criterion += 1;
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

impl Default for RandomForestClassifierSearchParameters {
    fn default() -> Self {
        let default_params = RandomForestClassifierParameters::default();

        RandomForestClassifierSearchParameters {
            criterion: vec![default_params.criterion],
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

impl<TX: FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    RandomForestClassifier<TX, TY, X, Y>
{
    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: RandomForestClassifierParameters,
    ) -> Result<RandomForestClassifier<TX, TY, X, Y>, Failed> {
        let (x_nrows, num_attributes) = x.shape();
        let y_ncols = y.shape();
        if x_nrows != y_ncols {
            return Err(Failed::fit("Number of rows in X should = len(y)"));
        }

        let mut yi: Vec<usize> = vec![0; y_ncols];
        let classes = y.unique();

        for (i, yi_i) in yi.iter_mut().enumerate().take(y_ncols) {
            let yc = y.get(i);
            *yi_i = classes.iter().position(|c| yc == c).unwrap();
        }

        let mtry = parameters
            .m
            .unwrap_or_else(|| ((num_attributes as f64).sqrt().floor()) as usize);

        let mut rng = get_rng_impl(Some(parameters.seed));
        let classes = y.unique();
        let k = classes.len();
        // TODO: use with_capacity here
        let mut trees: Vec<DecisionTreeClassifier<TX, TY, X, Y>> = Vec::new();

        let mut maybe_all_samples: Option<Vec<Vec<bool>>> = Option::None;
        if parameters.keep_samples {
            // TODO: use with_capacity here
            maybe_all_samples = Some(Vec::new());
        }

        for _ in 0..parameters.n_trees {
            let samples: Vec<usize> =
                RandomForestClassifier::<TX, TY, X, Y>::sample_with_replacement(&yi, k, &mut rng);
            if let Some(ref mut all_samples) = maybe_all_samples {
                all_samples.push(samples.iter().map(|x| *x != 0).collect())
            }

            let params = DecisionTreeClassifierParameters {
                criterion: parameters.criterion.clone(),
                max_depth: parameters.max_depth,
                min_samples_leaf: parameters.min_samples_leaf,
                min_samples_split: parameters.min_samples_split,
                seed: Some(parameters.seed),
            };
            let tree = DecisionTreeClassifier::fit_weak_learner(x, y, samples, mtry, params)?;
            trees.push(tree);
        }

        Ok(RandomForestClassifier {
            trees: Some(trees),
            classes: Some(classes),
            samples: maybe_all_samples,
        })
    }

    /// Predict class for `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let mut result = Y::zeros(x.shape().0);

        let (n, _) = x.shape();

        for i in 0..n {
            result.set(
                i,
                self.classes.as_ref().unwrap()[self.predict_for_row(x, i)],
            );
        }

        Ok(result)
    }

    fn predict_for_row(&self, x: &X, row: usize) -> usize {
        let mut result = vec![0; self.classes.as_ref().unwrap().len()];

        for tree in self.trees.as_ref().unwrap().iter() {
            result[tree.predict_for_row(x, row)] += 1;
        }

        which_max(&result)
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
            result.set(
                i,
                self.classes.as_ref().unwrap()[self.predict_for_row_oob(x, i)],
            );
        }

        Ok(result)
    }

    fn predict_for_row_oob(&self, x: &X, row: usize) -> usize {
        let mut result = vec![0; self.classes.as_ref().unwrap().len()];

        for (tree, samples) in self
            .trees
            .as_ref()
            .unwrap()
            .iter()
            .zip(self.samples.as_ref().unwrap())
        {
            if !samples[row] {
                result[tree.predict_for_row(x, row)] += 1;
            }
        }

        which_max(&result)
    }

    fn sample_with_replacement(y: &[usize], num_classes: usize, rng: &mut impl Rng) -> Vec<usize> {
        let class_weight = vec![1.; num_classes];
        let nrows = y.len();
        let mut samples = vec![0; nrows];
        for (l, class_weight_l) in class_weight.iter().enumerate().take(num_classes) {
            let mut n_samples = 0;
            let mut index: Vec<usize> = Vec::new();
            for (i, y_i) in y.iter().enumerate().take(nrows) {
                if *y_i == l {
                    index.push(i);
                    n_samples += 1;
                }
            }

            let size = ((n_samples as f64) / *class_weight_l) as usize;
            for _ in 0..size {
                let xi: usize = rng.gen_range(0..n_samples);
                samples[index[xi]] += 1;
            }
        }
        samples
    }

    /// Predict class probabilities for a single input sample.
    ///
    /// This method averages the probability estimates from all trees in the forest.
    /// Each tree returns a probability distribution based on the class distribution
    /// in its leaf node (scikit-learn style), and these distributions are averaged
    /// across all trees to produce the final probability estimate.
    ///
    /// # Arguments
    ///
    /// * `x` - The input matrix containing all samples.
    /// * `row` - The index of the row in `x` for which to predict probabilities.
    ///
    /// # Returns
    ///
    /// A vector of probabilities, one for each class. The sum of probabilities equals 1.0.
    /// Each probability represents the average fraction of training samples of that class
    /// across all trees that reached the same leaf node for this input.
    fn predict_proba_for_row(&self, x: &X, row: usize) -> Vec<f64> {
        // improvement: unwrap делаем один раз
        let trees = self.trees.as_ref().unwrap();

        // improvement: unwrap classes тоже один раз
        let k = self.classes.as_ref().unwrap().len();

        let mut probs = vec![0.0; k];

        for tree in trees {
            let tree_probs = tree.predict_proba_for_row_real(x, row);

            // improvement: убран range loop
            // improvement: нет индексирования
            // improvement: zip гарантирует покомпонентное сложение
            for (p, tp) in probs.iter_mut().zip(tree_probs.iter()) {
                *p += *tp; // важно разыменование
            }
        }

        // improvement: unwrap уже не нужен
        let n_trees = trees.len() as f64;

        // improvement: убран needless_range_loop
        for p in &mut probs {
            *p /= n_trees;
        }

        probs
    }

    /// Predict class probabilities for the input samples.
    ///
    /// This method returns probability estimates for each sample in the input matrix.
    /// For each sample, probabilities are computed by averaging the predictions from
    /// all trees in the forest. Each tree contributes a probability distribution based
    /// on the class distribution in its leaf node.
    ///
    /// This is the scikit-learn style `predict_proba` behavior, providing calibrated
    /// probability estimates rather than just class predictions.
    ///
    /// # Arguments
    ///
    /// * `x` - The input samples as a matrix where each row is a sample and each column
    ///   is a feature.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<Vec<f64>>` where each inner vector corresponds to
    /// a sample and contains probabilities for each class. The sum of probabilities
    /// for each sample equals 1.0.
    ///
    /// # Note
    ///
    /// Return type is `Vec<Vec<f64>>` for minimal API changes. The tree classifier
    /// returns `DenseMatrix<f64>` for the same method.
    ///
    /// # Errors
    ///
    /// Returns an error if the forest has not been fitted (trees are None).
    pub fn predict_proba(&self, x: &X) -> Result<Vec<Vec<f64>>, Failed> {
        let (n, _) = x.shape();

        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            result.push(self.predict_proba_for_row(x, i));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::metrics::*;

    #[test]
    fn search_parameters() {
        let parameters = RandomForestClassifierSearchParameters {
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
    fn fit_predict() {
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ])
        .unwrap();
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let classifier = RandomForestClassifier::fit(
            &x,
            &y,
            RandomForestClassifierParameters {
                criterion: SplitCriterion::Gini,
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 100,
                m: Option::None,
                keep_samples: false,
                seed: 87,
            },
        )
        .unwrap();

        assert!(accuracy(&y, &classifier.predict(&x).unwrap()) >= 0.95);
    }

    #[test]
    fn test_random_matrix_with_wrong_rownum() {
        let x_rand: DenseMatrix<f64> = DenseMatrix::<f64>::rand(21, 200);

        let y: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let fail = RandomForestClassifier::fit(
            &x_rand,
            &y,
            RandomForestClassifierParameters {
                criterion: SplitCriterion::Gini,
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 100,
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
    fn fit_predict_iris_oob() {
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ])
        .unwrap();
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let classifier = RandomForestClassifier::fit(
            &x,
            &y,
            RandomForestClassifierParameters {
                criterion: SplitCriterion::Gini,
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 100,
                m: Option::None,
                keep_samples: true,
                seed: 87,
            },
        )
        .unwrap();

        assert!(
            accuracy(&y, &classifier.predict_oob(&x).unwrap())
                < accuracy(&y, &classifier.predict(&x).unwrap())
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ])
        .unwrap();
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let forest = RandomForestClassifier::fit(&x, &y, Default::default()).unwrap();

        let deserialized_forest: RandomForestClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>> =
            bincode::deserialize(&bincode::serialize(&forest).unwrap()).unwrap();

        assert_eq!(forest, deserialized_forest);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_predict_proba_iris() {
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ])
        .unwrap();
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let classifier = RandomForestClassifier::fit(
            &x,
            &y,
            RandomForestClassifierParameters {
                criterion: SplitCriterion::Gini,
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 10,
                m: Option::None,
                keep_samples: false,
                seed: 87,
            },
        )
        .unwrap();

        let probabilities = classifier.predict_proba(&x).unwrap();

        // Check 1: dimensions
        assert_eq!(probabilities.len(), 20);
        assert_eq!(probabilities[0].len(), 2);

        // Check 2: probabilities sum to 1.0 for all rows
        for row in 0..20 {
            let row_sum: f64 = probabilities[row].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {} probabilities should sum to 1, got {}",
                row,
                row_sum
            );
        }

        // Check 3: first 8 samples → higher prob for class 0
        for i in 0..8 {
            assert!(
                probabilities[i][0] > probabilities[i][1],
                "Sample {} should have higher prob for class 0",
                i
            );
        }

        // Check 4: last 12 samples → higher prob for class 1
        for i in 8..20 {
            assert!(
                probabilities[i][1] > probabilities[i][0],
                "Sample {} should have higher prob for class 1",
                i
            );
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_predict_proba_iris_mixed_leaves() {
        // Dataset with mixed leaves
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[5.1, 3.5, 1.4, 0.2], // Same features
            &[5.1, 3.5, 1.4, 0.2], // Same features
            &[7.0, 3.2, 4.7, 1.4],
            &[7.0, 3.2, 4.7, 1.4], // Same features
        ])
        .unwrap();
        let y = vec![0, 0, 1, 1, 1]; // Mixed classes in same feature region

        let classifier = RandomForestClassifier::fit(
            &x,
            &y,
            RandomForestClassifierParameters {
                n_trees: 5,
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap();

        let probabilities = classifier.predict_proba(&x).unwrap();

        // Check 1: All probabilities should be valid
        for row in 0..5 {
            let sum: f64 = probabilities[row].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Probabilities for row {} should sum to 1.0",
                row
            );
            for &p in &probabilities[row] {
                assert!(p >= 0.0 && p <= 1.0, "Probability out of range");
            }
        }

        // Check 2: First 3 samples must have non-zero prob for both classes, since they are mixed
        for i in 0..3 {
            assert!(
                probabilities[i][0] > 0.0,
                "Sample {} should have non-zero prob for class 0",
                i
            );
            assert!(
                probabilities[i][1] > 0.0,
                "Sample {} should have non-zero prob for class 1",
                i
            );
        }
    }
}
