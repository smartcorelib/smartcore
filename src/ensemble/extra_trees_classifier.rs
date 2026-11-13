//! # Extra Trees Classifier
//! An Extra-Trees (Extremely Randomized Trees) classifier is an ensemble estimator that fits multiple
//! randomized [decision trees](../../tree/index.html) on the dataset. The predicted class is the class
//! that gets the most votes from the trees.
//!
//! It is similar to a [random forest](../random_forest_classifier/index.html), but it introduces more
//! randomness in the way the trees are built. This can reduce the variance of the model and often makes
//! the training faster.
//!
//! The two key differences from a random forest are:
//! 1. By default, each tree uses the whole original dataset instead of a bootstrap sample.
//!    Set `bootstrap` to `true` to use bootstrap samples. OOB prediction needs bootstrap samples.
//! 2. When a node is split, each candidate feature gets one random split point, drawn uniformly between
//!    the minimum and the maximum of the feature in the node. The best of these random splits is used.
//!    A random forest uses the best split point of each candidate feature.
//!
//! At each node, a random sample of _m_ predictors is used as split candidates.
//! By default, _m_ is \\(\sqrt{p}\\), where _p_ is the number of predictors.
//!
//! By default, no bootstrap samples are used, so there are no out-of-bag samples. To use
//! `predict_oob`, set both `bootstrap` and `keep_samples` to `true`.
//!
//! See [ensemble models](../index.html) for more details.
//!
//! ## References:
//!
//! * ["Extremely randomized trees", Geurts P., Ernst D., Wehenkel L., Machine Learning 63(1), 3-42, 2006](https://doi.org/10.1007/s10994-006-6226-1)
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 8.2 Bagging, Random Forests, Boosting](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::ensemble::extra_trees_classifier::*;
//!
//! // Iris dataset
//! let x = DenseMatrix::from_2d_array(&[
//!     &[5.1, 3.5, 1.4, 0.2],
//!     &[4.9, 3.0, 1.4, 0.2],
//!     &[4.7, 3.2, 1.3, 0.2],
//!     &[4.6, 3.1, 1.5, 0.2],
//!     &[5.0, 3.6, 1.4, 0.2],
//!     &[5.4, 3.9, 1.7, 0.4],
//!     &[4.6, 3.4, 1.4, 0.3],
//!     &[5.0, 3.4, 1.5, 0.2],
//!     &[4.4, 2.9, 1.4, 0.2],
//!     &[4.9, 3.1, 1.5, 0.1],
//!     &[7.0, 3.2, 4.7, 1.4],
//!     &[6.4, 3.2, 4.5, 1.5],
//!     &[6.9, 3.1, 4.9, 1.5],
//!     &[5.5, 2.3, 4.0, 1.3],
//!     &[6.5, 2.8, 4.6, 1.5],
//!     &[5.7, 2.8, 4.5, 1.3],
//!     &[6.3, 3.3, 4.7, 1.6],
//!     &[4.9, 2.4, 3.3, 1.0],
//!     &[6.6, 2.9, 4.6, 1.3],
//!     &[5.2, 2.7, 3.9, 1.4],
//! ]).unwrap();
//! let y = vec![
//!     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//!     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//! ];
//!
//! let parameters = ExtraTreesClassifierParameters::default().with_seed(42);
//! let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
//! let y_hat = classifier.predict(&x).unwrap(); // use the same data for prediction
//! assert_eq!(y_hat, y);
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
use crate::tree::criterion::{ClassificationErrorCriterion, EntropyCriterion, GiniCriterion};
use crate::tree::decision_tree_classifier::SplitCriterion;

/// Parameters of the Extra Trees Classifier.
/// Some parameters here are passed directly into base estimator.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
#[must_use]
pub struct ExtraTreesClassifierParameters {
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
    pub n_trees: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of features to test at each split. See [`MaxFeatures`].
    pub m: MaxFeatures,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub keep_samples: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Seed used for feature selection and for the random split points of each tree. With `None`, a seed is chosen at fit time (from OS entropy with `std_rand`).
    pub seed: Option<u64>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Whether to use bootstrap samples to build trees. If `false`, each tree uses the whole dataset.
    /// This is required for OOB prediction.
    pub bootstrap: bool,
}

/// Extra Trees Classifier
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct ExtraTreesClassifier<
    TX: Number + FloatNumber + PartialOrd,
    TY: Number + Ord,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    forest_classifier: Option<BaseForest<TX, usize, X, Vec<usize>, usize>>,
    classes: Option<Vec<TY>>,
    #[cfg_attr(feature = "serde", serde(default))]
    bootstrap: bool,
    _phantom: PhantomData<Y>,
}

impl ExtraTreesClassifierParameters {
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
    pub fn with_n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }
    /// Number of features to test at each split. See [`MaxFeatures`].
    pub fn with_m(mut self, m: impl Into<MaxFeatures>) -> Self {
        self.m = m.into();
        self
    }
    /// Seed used for feature selection and for the random split points of each tree.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub fn with_keep_samples(mut self, keep_samples: bool) -> Self {
        self.keep_samples = keep_samples;
        self
    }
    /// Whether to use bootstrap samples to build trees. If `false`, each tree uses the whole dataset.
    /// This is required for OOB prediction.
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    PartialEq for ExtraTreesClassifier<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        self.forest_classifier == other.forest_classifier
            && self.classes == other.classes
            && self.bootstrap == other.bootstrap
    }
}

impl Default for ExtraTreesClassifierParameters {
    fn default() -> Self {
        ExtraTreesClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: Option::None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 10,
            m: MaxFeatures::Sqrt,
            keep_samples: false,
            seed: Option::None,
            bootstrap: false,
        }
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, ExtraTreesClassifierParameters>
    for ExtraTreesClassifier<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            forest_classifier: None,
            classes: None,
            bootstrap: false,
            _phantom: PhantomData,
        }
    }
    fn fit(x: &X, y: &Y, parameters: ExtraTreesClassifierParameters) -> Result<Self, Failed> {
        ExtraTreesClassifier::fit(x, y, parameters)
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    Predictor<X, Y> for ExtraTreesClassifier<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: FloatNumber + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    ExtraTreesClassifier<TX, TY, X, Y>
{
    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: ExtraTreesClassifierParameters,
    ) -> Result<ExtraTreesClassifier<TX, TY, X, Y>, Failed> {
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
    /// By default, no bootstrap samples are used, so each tree is fitted on all the rows with
    /// their weights. With `bootstrap = true`, each tree gets a bootstrap sample where the rows
    /// are drawn with a probability proportional to their weights, as `RandomForestClassifier` does.
    pub fn fit_with_weights(
        x: &X,
        y: &Y,
        sample_weights: &[f64],
        parameters: ExtraTreesClassifierParameters,
    ) -> Result<ExtraTreesClassifier<TX, TY, X, Y>, Failed> {
        validate_sample_weights(sample_weights, x.shape().0)?;
        Self::fit_inner(x, y, Some(sample_weights), parameters)
    }

    fn fit_inner(
        x: &X,
        y: &Y,
        sample_weights: Option<&[f64]>,
        parameters: ExtraTreesClassifierParameters,
    ) -> Result<ExtraTreesClassifier<TX, TY, X, Y>, Failed> {
        let classifier_params = BaseForestParameters {
            max_depth: parameters.max_depth,
            min_samples_leaf: parameters.min_samples_leaf,
            min_samples_split: parameters.min_samples_split,
            n_trees: parameters.n_trees,
            m: parameters.m,
            keep_samples: parameters.keep_samples,
            seed: parameters.seed,
            bootstrap: parameters.bootstrap,
            splitter: Splitter::Random,
        };

        let classes = y.unique();
        let mut yi: Vec<usize> = vec![0; y.shape()];

        for (i, yi_i) in yi.iter_mut().enumerate() {
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

        Ok(ExtraTreesClassifier {
            forest_classifier: Some(forest_classifier),
            classes: Some(classes),
            bootstrap: parameters.bootstrap,
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
    /// The forest must be fitted with `bootstrap` and `keep_samples` set to `true`.
    ///
    /// ```
    /// use smartcore::ensemble::extra_trees_classifier::*;
    /// use smartcore::linalg::basic::matrix::DenseMatrix;
    ///
    /// let rows: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64]).collect();
    /// let x = DenseMatrix::from_2d_vec(&rows).unwrap();
    /// let y: Vec<u32> = (0..40).map(|i| if i < 20 { 0 } else { 1 }).collect();
    /// let parameters = ExtraTreesClassifierParameters::default()
    ///     .with_n_trees(100)
    ///     .with_bootstrap(true)
    ///     .with_keep_samples(true)
    ///     .with_seed(42);
    /// let forest = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
    /// assert_eq!(forest.predict_oob(&x).unwrap().len(), y.len());
    /// ```
    pub fn predict_oob(&self, x: &X) -> Result<Y, Failed> {
        match (&self.forest_classifier, &self.classes) {
            // Without bootstrap, all rows are in-bag for all trees.
            (Some(_), Some(_)) if !self.bootstrap => {
                Err(Failed::predict("OOB prediction needs bootstrap = true"))
            }
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
    /// If the model is not fitted, this function returns a `Failed::predict` error.
    ///
    /// ```
    /// use smartcore::ensemble::extra_trees_classifier::*;
    /// use smartcore::linalg::basic::matrix::DenseMatrix;
    ///
    /// // Column 0 sets the class. Column 1 is constant, so it cannot split the data.
    /// let rows: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, 7.0]).collect();
    /// let x = DenseMatrix::from_2d_vec(&rows).unwrap();
    /// let y: Vec<u32> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();
    /// let parameters = ExtraTreesClassifierParameters::default()
    ///     .with_n_trees(10)
    ///     .with_seed(42);
    /// let forest = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
    ///
    /// // All splits use column 0, so it gets all the importance.
    /// assert_eq!(forest.compute_feature_importances().unwrap(), vec![1.0, 0.0]);
    /// ```
    pub fn compute_feature_importances(&self) -> Result<Vec<f64>, Failed> {
        match (&self.forest_classifier, &self.classes) {
            (Some(forest), Some(_)) => forest.compute_feature_importances(),
            _ => Err(Failed::predict(
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

    /// Part of the Iris dataset: 10 rows of class 0 and 10 rows of class 1.
    fn iris() -> (DenseMatrix<f64>, Vec<u32>) {
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
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        (x, y)
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_predict() {
        let (x, y) = iris();
        for criterion in [
            SplitCriterion::Gini,
            SplitCriterion::Entropy,
            SplitCriterion::ClassificationError,
        ] {
            let parameters = ExtraTreesClassifierParameters::default()
                .with_criterion(criterion.clone())
                .with_n_trees(50)
                .with_seed(87);
            let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
            let y_hat = classifier.predict(&x).unwrap();
            assert_eq!(accuracy(&y, &y_hat), 1.0, "criterion: {criterion:?}");
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_is_reproducible_for_a_seed() {
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(10)
            .with_seed(42);
        let first = ExtraTreesClassifier::fit(&x, &y, parameters.clone()).unwrap();
        let second = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        assert!(first == second);

        // A different seed gives different random split points, thus a different forest.
        let other = ExtraTreesClassifier::fit(
            &x,
            &y,
            ExtraTreesClassifierParameters::default()
                .with_n_trees(10)
                .with_seed(43),
        )
        .unwrap();
        assert!(first != other);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn trees_differ_without_bootstrap() {
        // All trees see the same rows. They must still differ, because each tree gets
        // its own seed for the random split points.
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(10)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        let trees = classifier.forest_classifier.as_ref().unwrap().trees();
        assert!(
            trees.iter().skip(1).any(|tree| *tree != trees[0]),
            "all trees are equal"
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn trees_differ_with_bootstrap() {
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(10)
            .with_bootstrap(true)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        let trees = classifier.forest_classifier.as_ref().unwrap().trees();
        assert!(
            trees.iter().skip(1).any(|tree| *tree != trees[0]),
            "all trees are equal"
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_oob_without_fit_should_not_panic() {
        let classifier: ExtraTreesClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            ExtraTreesClassifier::new();
        let x = DenseMatrix::from_2d_array(&[&[1.0f64]]).unwrap();
        let yhat = classifier.predict_oob(&x);
        let msg = "'fit' should be called before calling 'predict'";
        assert_eq!(yhat.err(), Some(Failed::predict(msg)));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_oob_without_bootstrap_should_fail() {
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(10)
            .with_keep_samples(true)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        let msg = "OOB prediction needs bootstrap = true";
        assert_eq!(classifier.predict_oob(&x).err(), Some(Failed::predict(msg)));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_oob_without_keep_samples_should_fail() {
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(10)
            .with_bootstrap(true)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        let result = classifier.predict_oob(&x);
        assert_eq!(
            result.err().map(|e| e.error()),
            Some(FailedError::PredictFailed)
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_oob_with_bootstrap_gives_good_predictions() {
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(100)
            .with_bootstrap(true)
            .with_keep_samples(true)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        let y_oob = classifier.predict_oob(&x).unwrap();
        assert_eq!(y_oob.len(), y.len());
        assert!(accuracy(&y, &y_oob) >= 0.9, "{y_oob:?}");
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_oob_labels_outside_class_index_range() {
        let (x, y) = iris();
        let y: Vec<i32> = y.iter().map(|c| if *c == 0 { -7 } else { 12 }).collect();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(100)
            .with_bootstrap(true)
            .with_keep_samples(true)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        let y_oob = classifier.predict_oob(&x).unwrap();
        assert!(y_oob.iter().all(|label| [-7, 12].contains(label)));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_on_empty_data_returns_error() {
        let empty: DenseMatrix<f64> = DenseMatrix::new(0, 2, vec![], false).unwrap();
        assert_eq!(empty.shape(), (0, 2));

        let y: Vec<u32> = vec![];
        let result = ExtraTreesClassifier::fit(
            &empty,
            &y,
            ExtraTreesClassifierParameters::default().with_seed(0),
        );
        assert_eq!(
            result.err().map(|e| e.error()),
            Some(FailedError::ParametersError)
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_and_feature_importances_without_fit_should_not_panic() {
        let classifier: ExtraTreesClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            ExtraTreesClassifier::new();
        let x = DenseMatrix::from_2d_array(&[&[1.0f64]]).unwrap();
        assert!(classifier.predict(&x).is_err());
        assert!(classifier.compute_feature_importances().is_err());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_predict_labels_outside_class_index_range() {
        // The labels are not 0..k, so they must be mapped to class indices and back.
        let (x, y) = iris();
        let y: Vec<i32> = y.iter().map(|c| if *c == 0 { -7 } else { 12 }).collect();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(20)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        assert_eq!(classifier.predict(&x).unwrap(), y);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_with_single_class_gives_trivial_classifier() {
        let (x, _) = iris();
        let y = vec![3u32; 20];
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(5)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        assert_eq!(classifier.predict(&x).unwrap(), y);
        assert_eq!(
            classifier.compute_feature_importances().unwrap(),
            vec![0.0; 4]
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_with_weights_validates_weights() {
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(5)
            .with_seed(42);
        let mut negative = vec![1.0; 20];
        negative[3] = -1.0;
        let mut not_finite = vec![1.0; 20];
        not_finite[3] = f64::NAN;
        for weights in [vec![1.0; 19], negative, not_finite, vec![0.0; 20]] {
            let result =
                ExtraTreesClassifier::fit_with_weights(&x, &y, &weights, parameters.clone());
            assert!(result.is_err(), "weights: {weights:?}");
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_with_weights_uses_the_weights_in_each_tree() {
        // With max_depth 0, each tree is one leaf. Without bootstrap samples, the leaf holds
        // all the rows, so it predicts the class with the largest total weight.
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_max_depth(0)
            .with_n_trees(5)
            .with_seed(42);

        let mut weights = vec![1.0; 20];
        weights[0] = 11.0; // class 0 now has weight 20 against 10 for class 1
        let classifier =
            ExtraTreesClassifier::fit_with_weights(&x, &y, &weights, parameters.clone()).unwrap();
        assert_eq!(classifier.predict(&x).unwrap(), vec![0; 20]);

        weights[0] = 1.0;
        weights[19] = 11.0; // class 1 now has weight 20 against 10 for class 0
        let classifier =
            ExtraTreesClassifier::fit_with_weights(&x, &y, &weights, parameters).unwrap();
        assert_eq!(classifier.predict(&x).unwrap(), vec![1; 20]);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_with_zero_weights_ignores_rows() {
        // Give the rows of class 1 the labels of class 0, but weight 0.
        // The rows have no effect, so the forest must still predict class 1 for them.
        let (x, y) = iris();
        let mut y_wrong = y.clone();
        let mut weights = vec![1.0; 20];
        for i in [10, 11, 12] {
            y_wrong[i] = 0;
            weights[i] = 0.0;
        }
        for bootstrap in [false, true] {
            let parameters = ExtraTreesClassifierParameters::default()
                .with_n_trees(50)
                .with_bootstrap(bootstrap)
                .with_seed(42);
            let classifier =
                ExtraTreesClassifier::fit_with_weights(&x, &y_wrong, &weights, parameters).unwrap();
            let y_hat = classifier.predict(&x).unwrap();
            assert_eq!(y_hat, y, "bootstrap: {bootstrap}");
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_with_weights_and_bootstrap_draws_rows_by_weight() {
        let (x, y) = iris();
        let n_trees = 10;
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(n_trees)
            .with_bootstrap(true)
            .with_keep_samples(true)
            .with_seed(42);
        let weights: Vec<f64> = (0..20).map(|i| if i < 10 { 100.0 } else { 1.0 }).collect();
        let classifier =
            ExtraTreesClassifier::fit_with_weights(&x, &y, &weights, parameters).unwrap();

        let samples = classifier
            .forest_classifier
            .as_ref()
            .unwrap()
            .samples()
            .unwrap();
        assert_eq!(samples.len(), n_trees);
        let light_rows_in_bag = samples
            .iter()
            .map(|in_bag| in_bag[10..].iter().filter(|b| **b).count())
            .sum::<usize>();
        // Expected value is about 2 of 100 pairs (tree, light row).
        assert!(
            light_rows_in_bag <= 10,
            "light rows in bag: {light_rows_in_bag} of {}",
            10 * n_trees
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn compute_feature_importances() {
        // Column 0 sets the class. Columns 1 and 2 are constant, column 3 is noise.
        let n = 40;
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![i as f64, 1.0, -2.0, ((i * 17) % n) as f64])
            .collect();
        let x = DenseMatrix::from_2d_vec(&rows).unwrap();
        let y: Vec<u32> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(50)
            .with_m(4)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();

        let importances = classifier.compute_feature_importances().unwrap();
        assert_eq!(importances.len(), 4);
        assert!((importances.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(importances[1], 0.0);
        assert_eq!(importances[2], 0.0);
        assert!(importances[0] > importances[3]);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn feature_importances_are_zero_without_splits() {
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_max_depth(0)
            .with_n_trees(5)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();
        assert_eq!(
            classifier.compute_feature_importances().unwrap(),
            vec![0.0; 4]
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let (x, y) = iris();
        let parameters = ExtraTreesClassifierParameters::default()
            .with_n_trees(10)
            .with_seed(42);
        let classifier = ExtraTreesClassifier::fit(&x, &y, parameters).unwrap();

        let deserialized: ExtraTreesClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            postcard::from_bytes(&postcard::to_allocvec(&classifier).unwrap()).unwrap();
        assert!(classifier == deserialized);
        assert_eq!(deserialized.predict(&x).unwrap(), y);

        // Parameters: a missing seed is None. Note: `serde(default)` on a field uses the
        // default of the field type, not `ExtraTreesClassifierParameters::default()`.
        let params: ExtraTreesClassifierParameters = serde_json::from_str("{}").unwrap();
        assert_eq!(params.seed, None);
        let params: ExtraTreesClassifierParameters =
            serde_json::from_str(r#"{"seed": 42}"#).unwrap();
        assert_eq!(params.seed, Some(42));
        assert!(!params.bootstrap);
        assert!(!params.keep_samples);
    }
}
