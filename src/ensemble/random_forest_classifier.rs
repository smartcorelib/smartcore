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

use std::default::Default;
use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::ensemble::base_forest::{
    BaseForest, BaseForestParameters, predict_mode, predict_mode_oob,
};
use crate::ensemble::max_features::MaxFeatures;
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;

use crate::tree::base_tree::{Splitter, validate_sample_weights};
use crate::tree::criterion::{
    ClassificationErrorCriterion, ClassificationOutput, EntropyCriterion, GiniCriterion,
};
use crate::tree::decision_tree_classifier::SplitCriterion;

/// Parameters of the Random Forest algorithm.
/// Some parameters here are passed directly into base estimator.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
#[must_use]
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
    /// Number of features to test at each split. See [`MaxFeatures`].
    pub m: MaxFeatures,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub keep_samples: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Seed used for bootstrap sampling and feature selection for each tree. With None, a seed is chosen at fit time (from OS entropy with std_rand).
    pub seed: Option<u64>,
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
    forest_classifier: Option<BaseForest<TX, usize, X, Vec<usize>, ClassificationOutput>>,
    classes: Option<Vec<TY>>,
    _phantom: PhantomData<Y>,
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
    /// Number of features to test at each split. See [`MaxFeatures`].
    pub fn with_m(mut self, m: impl Into<MaxFeatures>) -> Self {
        self.m = m.into();
        self
    }

    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub fn with_keep_samples(mut self, keep_samples: bool) -> Self {
        self.keep_samples = keep_samples;
        self
    }

    /// Seed used for bootstrap sampling and feature selection for each tree. With None, a seed is chosen at fit time (from OS entropy with std_rand).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    PartialEq for RandomForestClassifier<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        self.forest_classifier == other.forest_classifier && self.classes == other.classes
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
            m: MaxFeatures::Sqrt,
            keep_samples: false,
            seed: Option::None,
        }
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, RandomForestClassifierParameters>
    for RandomForestClassifier<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            forest_classifier: None,
            classes: None,
            _phantom: PhantomData,
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
#[must_use]
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
    pub m: Vec<MaxFeatures>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub keep_samples: Vec<bool>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Seed used for bootstrap sampling and feature selection for each tree. With None, a seed is chosen at fit time (from OS entropy with std_rand).
    pub seed: Vec<Option<u64>>,
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
        Self::fit_inner(x, y, None, parameters)
    }

    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    /// * `sample_weights`: sample_weights to use during fitting
    ///
    /// The weights must have one value for each row of `x`, be finite and non-negative,
    /// and have a positive sum. Otherwise this function returns a `Failed::fit` error.
    /// A row with weight 0 has no effect on the fit.
    ///
    /// Each tree gets a bootstrap sample where the rows are drawn with a probability
    /// proportional to their weights. The trees are then fitted on that sample without
    /// weights, so the weights are not applied twice. `RandomForestRegressor` does the same.
    ///
    /// ```
    /// use smartcore::ensemble::random_forest_classifier::*;
    /// use smartcore::linalg::basic::matrix::DenseMatrix;
    ///
    /// let rows: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
    /// let x = DenseMatrix::from_2d_vec(&rows).unwrap();
    /// let y: Vec<u32> = (0..20).map(|i| if i < 15 { 0 } else { 1 }).collect();
    /// // The rows of class 1 hold 75% of the total weight.
    /// let sample_weights: Vec<f64> = (0..20).map(|i| if i < 15 { 1.0 } else { 9.0 }).collect();
    /// let parameters = RandomForestClassifierParameters::default()
    ///     .with_max_depth(0)
    ///     .with_n_trees(50)
    ///     .with_seed(42);
    /// let forest =
    ///     RandomForestClassifier::fit_with_weights(&x, &y, &sample_weights, parameters).unwrap();
    /// assert_eq!(forest.predict(&x).unwrap(), vec![1; 20]);
    /// ```
    pub fn fit_with_weights(
        x: &X,
        y: &Y,
        sample_weights: &[f64],
        parameters: RandomForestClassifierParameters,
    ) -> Result<RandomForestClassifier<TX, TY, X, Y>, Failed> {
        validate_sample_weights(sample_weights, x.shape().0)?;
        Self::fit_inner(x, y, Some(sample_weights), parameters)
    }

    fn fit_inner(
        x: &X,
        y: &Y,
        sample_weights: Option<&[f64]>,
        parameters: RandomForestClassifierParameters,
    ) -> Result<RandomForestClassifier<TX, TY, X, Y>, Failed> {
        let classifier_params = BaseForestParameters {
            max_depth: parameters.max_depth,
            min_samples_leaf: parameters.min_samples_leaf,
            min_samples_split: parameters.min_samples_split,
            n_trees: parameters.n_trees as usize,
            m: parameters.m,
            keep_samples: parameters.keep_samples,
            seed: parameters.seed,
            bootstrap: true,
            splitter: Splitter::Best,
        };

        let classes = y.unique();
        let mut yi: Vec<usize> = vec![0; y.shape()];

        for (i, yi_i) in yi.iter_mut().enumerate().take(y.shape()) {
            let yc = y.get(i);
            *yi_i = classes.iter().position(|c| yc == c).unwrap();
        }

        let k = classes.len();
        let forest_classifier = match parameters.criterion {
            SplitCriterion::Gini => BaseForest::fit(
                x,
                &yi,
                sample_weights,
                &GiniCriterion::new(k),
                classifier_params,
            ),
            SplitCriterion::Entropy => BaseForest::fit(
                x,
                &yi,
                sample_weights,
                &EntropyCriterion::new(k),
                classifier_params,
            ),
            SplitCriterion::ClassificationError => BaseForest::fit(
                x,
                &yi,
                sample_weights,
                &ClassificationErrorCriterion::new(k),
                classifier_params,
            ),
        }?;

        Ok(RandomForestClassifier {
            forest_classifier: Some(forest_classifier),
            classes: Some(classes),
            _phantom: PhantomData,
        })
    }

    /// Predict class for `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        match (&self.forest_classifier, &self.classes) {
            (Some(forest), Some(classes)) => {
                let preds_as_usize = predict_mode(forest, x, classes.len())?;
                let mut result = Y::zeros(x.shape().0);
                for (i, class_index) in preds_as_usize.iter().enumerate() {
                    result.set(i, classes[*class_index]);
                }
                Ok(result)
            }
            _ => Err(Failed::predict(
                "'fit' should be called before calling 'predict'",
            )),
        }
    }

    /// Predict OOB classes for `x`. `x` is expected to be equal to the dataset used in training.
    pub fn predict_oob(&self, x: &X) -> Result<Y, Failed> {
        match (&self.forest_classifier, &self.classes) {
            (Some(forest), Some(classes)) => {
                let preds_as_usize = predict_mode_oob(forest, x, classes.len())?;
                let mut result = Y::zeros(x.shape().0);
                for (i, class_index) in preds_as_usize.iter().enumerate() {
                    result.set(i, classes[*class_index]);
                }
                Ok(result)
            }
            _ => Err(Failed::predict(
                "'fit' should be called before calling 'predict'",
            )),
        }
    }

    /// Compute the impurity-based feature importances for this forest.
    /// * returns: a vector where index `j` gives the importance of feature `j`.
    ///   The importances are normalized: they add up to 1 when the trees contain at least one split.
    ///   If no tree contains a split (for example, with `max_depth = 0`), all the importances are 0.
    ///
    /// The importance of a feature is the total impurity decrease of the splits on that feature.
    /// It is normalized for each tree, averaged over the trees and normalized again.
    /// If the model is not fitted, this function returns a `Failed::predict` error,
    /// as `RandomForestRegressor` does.
    ///
    /// ```
    /// use smartcore::ensemble::random_forest_classifier::*;
    /// use smartcore::linalg::basic::matrix::DenseMatrix;
    ///
    /// // Column 0 sets the class. Column 1 is a permutation of the row index, so it is noise.
    /// let n = 50;
    /// let rows: Vec<Vec<f64>> = (0..n)
    ///     .map(|i| vec![i as f64, ((i * 37) % n) as f64])
    ///     .collect();
    /// let x = DenseMatrix::from_2d_vec(&rows).unwrap();
    /// let y: Vec<u32> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
    /// let parameters = RandomForestClassifierParameters::default()
    ///     .with_n_trees(20)
    ///     .with_m(2)
    ///     .with_seed(42);
    /// let forest = RandomForestClassifier::fit(&x, &y, parameters).unwrap();
    ///
    /// let importances = forest.compute_feature_importances().unwrap();
    /// assert!((importances.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    /// assert!(importances[0] > importances[1]);
    /// ```
    pub fn compute_feature_importances(&self) -> Result<Vec<f64>, Failed> {
        match &self.forest_classifier {
            Some(forest) => forest.compute_feature_importances(),
            None => Err(Failed::predict(
                "'fit' should be called before calling 'compute_feature_importances'",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FailedError;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::metrics::*;

    #[test]
    fn search_parameters() {
        let parameters = RandomForestClassifierSearchParameters {
            n_trees: vec![10, 100],
            m: vec![MaxFeatures::Sqrt, MaxFeatures::Count(1)],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 10);
        assert_eq!(next.m, MaxFeatures::Sqrt);
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 100);
        assert_eq!(next.m, MaxFeatures::Sqrt);
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 10);
        assert_eq!(next.m, MaxFeatures::Count(1));
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 100);
        assert_eq!(next.m, MaxFeatures::Count(1));
        assert!(iter.next().is_none());
    }

    #[test]
    fn search_parameters_with_seed() {
        let parameters = RandomForestClassifierSearchParameters {
            n_trees: vec![10, 100],
            seed: vec![None, Some(42)],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 10);
        assert_eq!(next.seed, None);
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 100);
        assert_eq!(next.seed, None);
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 10);
        assert_eq!(next.seed, Some(42));
        let next = iter.next().unwrap();
        assert_eq!(next.n_trees, 100);
        assert_eq!(next.seed, Some(42));
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
                m: MaxFeatures::Sqrt,
                keep_samples: false,
                seed: Some(87),
            },
        )
        .unwrap();

        assert!(accuracy(&y, &classifier.predict(&x).unwrap()) >= 0.95);
    }

    // Regression test for #322: `cross_validate` with a `RandomForestClassifier`
    // and the `precision` metric must accept the same ordered integer labels
    // (`Number + Ord`) that `fit` accepts.
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cross_validate_with_precision_and_integer_labels() {
        use crate::model_selection::{KFold, cross_validate};

        // Feature matrix is `f32`, mirroring the report in #322.
        let x: DenseMatrix<f32> = DenseMatrix::from_2d_array(&[
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
        let y: Vec<u16> = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let results = cross_validate(
            RandomForestClassifier::new(),
            &x,
            &y,
            RandomForestClassifierParameters {
                criterion: SplitCriterion::Gini,
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 10,
                m: MaxFeatures::Sqrt,
                keep_samples: false,
                seed: Some(87),
            },
            &KFold::default().with_n_splits(5),
            &precision,
        )
        .unwrap();

        assert_eq!(results.test_score.len(), 5);
        assert!(results.test_score.iter().all(|s| *s >= 0.0 && *s <= 1.0));
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
                m: MaxFeatures::Sqrt,
                keep_samples: false,
                seed: Some(87),
            },
        );

        assert!(fail.is_err());
    }

    #[test]
    fn test_fit_on_empty_data_returns_error() {
        // 2 rows x 2 features — values are arbitrary; only the empty-row case is under test
        let full = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64, 1.0], vec![0.0, 1.0]]).unwrap();
        let empty = full.take(&[] as &[usize], 0);
        assert_eq!(empty.shape(), (0, 2));

        let y: Vec<u32> = vec![];
        let result = RandomForestClassifier::fit(
            &empty,
            &y,
            RandomForestClassifierParameters {
                criterion: SplitCriterion::Gini,
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 10,
                m: MaxFeatures::Sqrt,
                keep_samples: false,
                seed: Some(0),
            },
        );
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error(), FailedError::ParametersError);
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
                m: MaxFeatures::Sqrt,
                keep_samples: true,
                seed: Some(87),
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
    fn predict_without_fit_should_not_panic() {
        let tree: RandomForestClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            RandomForestClassifier::new();
        let x = DenseMatrix::from_2d_array(&[&[1.0f64]]).expect("Construction of x should work");
        let yhat = tree.predict(&x);
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
        let tree: RandomForestClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            RandomForestClassifier::new();
        let x = DenseMatrix::from_2d_array(&[&[1.0f64]]).expect("Construction of x should work");
        let yhat = tree.predict_oob(&x);
        assert!(yhat.is_err());
        let msg = "'fit' should be called before calling 'predict'";
        assert_eq!(yhat.err(), Some(Failed::predict(msg)));
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

        let forest = RandomForestClassifier::fit(
            &x,
            &y,
            RandomForestClassifierParameters::default().with_seed(42),
        )
        .unwrap();

        let deserialized_forest: RandomForestClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>> =
            postcard::from_bytes(&postcard::to_allocvec(&forest).unwrap()).unwrap();

        assert_eq!(forest, deserialized_forest);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(feature = "serde")]
    fn serde_seed_parameter() {
        for seed in [Some(42u64), None] {
            let params = RandomForestClassifierParameters {
                seed,
                ..Default::default()
            };

            let json = serde_json::to_string(&params).unwrap();
            let from_json: RandomForestClassifierParameters = serde_json::from_str(&json).unwrap();
            assert_eq!(from_json.seed, seed);

            let bytes = postcard::to_allocvec(&params).unwrap();
            let from_postcard: RandomForestClassifierParameters =
                postcard::from_bytes(&bytes).unwrap();
            assert_eq!(from_postcard.seed, seed);
        }

        let old: RandomForestClassifierParameters =
            serde_json::from_str(r#"{"seed": 42}"#).unwrap();
        assert_eq!(old.seed, Some(42));

        let null: RandomForestClassifierParameters =
            serde_json::from_str(r#"{"seed": null}"#).unwrap();
        assert_eq!(null.seed, None);

        let missing: RandomForestClassifierParameters = serde_json::from_str("{}").unwrap();
        assert_eq!(
            missing.seed,
            RandomForestClassifierParameters::default().seed
        );
        assert_eq!(missing.seed, None);

        let search = RandomForestClassifierSearchParameters {
            seed: vec![None, Some(42)],
            ..Default::default()
        };
        let json = serde_json::to_string(&search).unwrap();
        let from_json: RandomForestClassifierSearchParameters =
            serde_json::from_str(&json).unwrap();
        assert_eq!(from_json.seed, vec![None, Some(42)]);

        let old_search: RandomForestClassifierSearchParameters =
            serde_json::from_str(r#"{"seed": [1, 2]}"#).unwrap();
        assert_eq!(old_search.seed, vec![Some(1), Some(2)]);
    }

    #[test]
    fn each_tree_gets_different_feature_sample() {
        // The classifier always uses bootstrapping, thus the trees are different also when
        // they all use the same feature sample. For this reason, we look at the features
        // directly. If all trees use the same seed, all trees split on the
        // same feature.

        // Create some data: x[i][j] = i * (j + 1)
        let n_rows = 30;
        let n_cols = 4;
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator(
            (0..n_cols * n_rows).map(|k| ((k / n_cols) * (k % n_cols + 1)) as f64),
            n_rows,
            n_cols,
            0,
        );
        let y: Vec<u32> = (0..n_rows)
            .map(|i| if i < n_rows / 2 { 0 } else { 1 })
            .collect();

        let params = RandomForestClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: Some(2),
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 10,
            m: MaxFeatures::Count(1),
            keep_samples: false,
            seed: Some(42),
        };
        let forest = RandomForestClassifier::fit(&x, &y, params).unwrap();

        // Each tree has one split, thus one feature has a positive importance
        let mut importances = vec![0.0; n_cols];
        let root_features: Vec<usize> = forest
            .forest_classifier
            .unwrap()
            .trees()
            .iter()
            .map(|tree| {
                tree.compute_feature_importances(false, &mut importances);

                importances
                    .iter()
                    .position(|importance| *importance > 0.0)
                    .unwrap_or(usize::MAX)
            })
            .collect();

        assert!(
            root_features.iter().any(|f| *f != root_features[0]),
            "all trees split on the same feature: {root_features:?}"
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_predict_labels_outside_class_index_range() {
        // The labels are not 0..k-1: one is negative, one is equal to k = 3 and one is large.
        // The forest must use the position of the label in `classes`, not the label itself.
        let x = DenseMatrix::from_2d_array(&[
            &[0.0],
            &[1.0],
            &[2.0],
            &[10.0],
            &[11.0],
            &[12.0],
            &[20.0],
            &[21.0],
            &[22.0],
        ])
        .unwrap();
        let y: Vec<i32> = vec![-7, -7, -7, 3, 3, 3, 100, 100, 100];

        let params = RandomForestClassifierParameters {
            n_trees: 20,
            keep_samples: true,
            seed: Some(42),
            ..Default::default()
        };
        let forest = RandomForestClassifier::fit(&x, &y, params).unwrap();
        assert_eq!(forest.predict(&x).unwrap(), y);

        // Each OOB prediction must be one of the original labels
        let oob = forest.predict_oob(&x).unwrap();
        assert!(oob.iter().all(|label| [-7, 3, 100].contains(label)));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_with_single_class_gives_trivial_classifier() {
        // As in scikit-learn, one class is not an error: each tree is one leaf.
        let x = DenseMatrix::from_2d_array(&[
            &[0.0, 5.0],
            &[1.0, 4.0],
            &[2.0, 3.0],
            &[3.0, 2.0],
            &[4.0, 1.0],
            &[5.0, 0.0],
        ])
        .unwrap();
        let y: Vec<i32> = vec![5; 6];

        let params = RandomForestClassifierParameters {
            n_trees: 10,
            keep_samples: true,
            seed: Some(42),
            ..Default::default()
        };
        let forest = RandomForestClassifier::fit(&x, &y, params).unwrap();

        let trees = forest.forest_classifier.as_ref().unwrap().trees();
        assert_eq!(trees.len(), 10);
        assert!(trees.iter().all(|tree| tree.depth() == 0));

        // The forest predicts the single class, also for data that is not in the training set
        let x_new =
            DenseMatrix::from_2d_array(&[&[-10.0, 10.0], &[2.5, 2.5], &[100.0, -1.0]]).unwrap();
        assert_eq!(forest.predict(&x).unwrap(), y);
        assert_eq!(forest.predict(&x_new).unwrap(), vec![5; 3]);
        assert_eq!(forest.predict_oob(&x).unwrap(), y);
    }

    #[test]
    fn fit_with_weights() {
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator((0..20).map(|i| i as f64), 20, 1, 0);
        let y: Vec<u32> = (0..20).map(|i| if i < 15 { 0 } else { 1 }).collect();
        // Rows with y = 1 have weight 9, so they hold 75% of the total weight.
        let sample_weights: Vec<f64> = (0..20).map(|i| if i < 15 { 1.0 } else { 9.0 }).collect();
        let parameters = RandomForestClassifierParameters::default()
            .with_max_depth(0)
            .with_n_trees(50)
            .with_seed(42);

        let forest = RandomForestClassifier::fit(&x, &y, parameters.clone()).unwrap();
        assert_eq!(forest.predict(&x).unwrap(), vec![0; 20]);
        let forest =
            RandomForestClassifier::fit_with_weights(&x, &y, &sample_weights, parameters.clone())
                .unwrap();
        assert_eq!(forest.predict(&x).unwrap(), vec![1; 20]);

        let wrong_length = "Number of sample weights must equal number of rows in x";
        let not_finite_or_negative = "Sample weights must be finite and non-negative";
        let mut negative = sample_weights.clone();
        negative[3] = -1.0;
        let mut nan = sample_weights.clone();
        nan[3] = f64::NAN;
        let mut inf = sample_weights.clone();
        inf[3] = f64::INFINITY;
        let cases: Vec<(Vec<f64>, &str)> = vec![
            (vec![1.0; 3], wrong_length),
            (negative, not_finite_or_negative),
            (nan, not_finite_or_negative),
            (inf, not_finite_or_negative),
            (vec![0.0; 20], "Sum of sample weights must be positive"),
        ];
        for (weights, msg) in cases {
            let result =
                RandomForestClassifier::fit_with_weights(&x, &y, &weights, parameters.clone());
            assert_eq!(result.err(), Some(Failed::fit(msg)), "weights: {weights:?}");
        }
    }

    #[test]
    fn fit_with_zero_weights_ignores_rows() {
        // The bootstrap sample is drawn proportionally to the weights,
        // so no tree uses a row with weight 0.
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator((0..20).map(|i| i as f64), 20, 1, 0);
        let y: Vec<u32> = (0..20).map(|i| (i % 3) as u32).collect();
        let sample_weights: Vec<f64> = (0..20)
            .map(|i| if i % 3 == 2 { 0.0 } else { 1.0 })
            .collect();
        for criterion in [
            SplitCriterion::Gini,
            SplitCriterion::Entropy,
            SplitCriterion::ClassificationError,
        ] {
            let parameters = RandomForestClassifierParameters::default()
                .with_criterion(criterion.clone())
                .with_n_trees(20)
                .with_keep_samples(true)
                .with_seed(42);
            let forest =
                RandomForestClassifier::fit_with_weights(&x, &y, &sample_weights, parameters)
                    .unwrap();
            let samples = forest
                .forest_classifier
                .as_ref()
                .unwrap()
                .samples()
                .unwrap();
            for tree_samples in samples {
                for (i, used) in tree_samples.iter().enumerate() {
                    assert!(
                        !used || sample_weights[i] > 0.0,
                        "row {i} has weight 0 but is used (criterion: {criterion:?})"
                    );
                }
            }
            // Class 2 has weight 0, so it is never predicted.
            assert!(
                forest.predict(&x).unwrap().iter().all(|&c| c != 2),
                "criterion: {criterion:?}"
            );
        }
    }

    #[test]
    fn compute_feature_importances() {
        let forest: RandomForestClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            RandomForestClassifier::new();
        let msg = "'fit' should be called before calling 'compute_feature_importances'";
        assert_eq!(
            forest.compute_feature_importances().err(),
            Some(Failed::predict(msg))
        );

        // Column 0 sets the class. Column 1 is a permutation of the row index, so it is noise.
        let n = 50;
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![i as f64, ((i * 37) % n) as f64])
            .collect();
        let x = DenseMatrix::from_2d_vec(&rows).unwrap();
        let y: Vec<u32> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        let parameters = RandomForestClassifierParameters::default()
            .with_n_trees(20)
            .with_m(2)
            .with_seed(42);
        let forest = RandomForestClassifier::fit(&x, &y, parameters).unwrap();

        let importances = forest.compute_feature_importances().unwrap();
        assert!((importances.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(importances[0] > importances[1]);
    }
}
