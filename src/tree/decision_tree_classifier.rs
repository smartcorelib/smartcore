//! # Decision Tree Classifier
//!
//! The process of building a classification tree is similar to the task of building a [regression tree](../decision_tree_regressor/index.html).
//! However, in the classification setting one of these criteriums is used for making the binary splits:
//!
//! * Classification error rate, \\(E = 1 - \max_k(p_{mk})\\)
//!
//! * Gini index, \\(G = \sum_{k=1}^K p_{mk}(1 - p_{mk})\\)
//!
//! * Entropy, \\(D = -\sum_{k=1}^K p_{mk}\log p_{mk}\\)
//!
//! where \\(p_{mk}\\) represents the proportion of training observations in the *m*th region that are from the *k*th class.
//!
//! The classification error rate is simply the fraction of the training observations in that region that do not belong to the most common class.
//! Classification error is not sufficiently sensitive for tree-growing, and in practice Gini index or Entropy are preferable.
//!
//! The Gini index is referred to as a measure of node purity. A small value indicates that a node contains predominantly observations from a single class.
//!
//! The Entropy, like Gini index will take on a small value if the *m*th node is pure.
//!
//! Example:
//!
//! ```
//! use rand::Rng;
//!
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::tree::decision_tree_classifier::*;
//!
//! // Iris dataset
//! let x = DenseMatrix::from_2d_array(&[
//!            &[5.1, 3.5, 1.4, 0.2],
//!            &[4.9, 3.0, 1.4, 0.2],
//!            &[4.7, 3.2, 1.3, 0.2],
//!            &[4.6, 3.1, 1.5, 0.2],
//!            &[5.0, 3.6, 1.4, 0.2],
//!            &[5.4, 3.9, 1.7, 0.4],
//!            &[4.6, 3.4, 1.4, 0.3],
//!            &[5.0, 3.4, 1.5, 0.2],
//!            &[4.4, 2.9, 1.4, 0.2],
//!            &[4.9, 3.1, 1.5, 0.1],
//!            &[7.0, 3.2, 4.7, 1.4],
//!            &[6.4, 3.2, 4.5, 1.5],
//!            &[6.9, 3.1, 4.9, 1.5],
//!            &[5.5, 2.3, 4.0, 1.3],
//!            &[6.5, 2.8, 4.6, 1.5],
//!            &[5.7, 2.8, 4.5, 1.3],
//!            &[6.3, 3.3, 4.7, 1.6],
//!            &[4.9, 2.4, 3.3, 1.0],
//!            &[6.6, 2.9, 4.6, 1.3],
//!            &[5.2, 2.7, 3.9, 1.4],
//!         ]).unwrap();
//! let y = vec![ 0, 0, 0, 0, 0, 0, 0, 0,
//!            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
//!
//! let tree = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
//!
//! let y_hat = tree.predict(&x).unwrap(); // use the same data for prediction
//! ```
//!
//!
//! ## References:
//! * ["Classification and regression trees", Breiman, L, Friedman, J H, Olshen, R A, and Stone, C J, 1984](https://www.sciencebase.gov/catalog/item/545d07dfe4b0ba8303f728c1)
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., Chapter 8](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::default::Default;
use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::MutArray;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::linalg::basic::matrix::DenseMatrix;
use crate::numbers::basenum::Number;
use crate::tree::base_tree::{BaseTree, BaseTreeParameters, Splitter};
use crate::tree::criterion::{ClassificationErrorCriterion, EntropyCriterion, GiniCriterion};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// Parameters of Decision Tree
#[must_use]
pub struct DecisionTreeClassifierParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Split criteria to use when building a tree.
    pub criterion: SplitCriterion,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The maximum depth of the tree.
    pub max_depth: Option<u16>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to be at a leaf node.
    pub min_samples_leaf: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Controls the randomness of the estimator
    pub seed: Option<u64>,
}

/// Decision Tree
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct DecisionTreeClassifier<
    TX: Number + PartialOrd,
    TY: Number + Ord,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    tree_classifier: Option<BaseTree<TX, usize, X, Vec<usize>, usize>>,
    num_features: Option<usize>,
    classes: Option<Vec<TY>>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    DecisionTreeClassifier<TX, TY, X, Y>
{
    /// get classes vector, return a shared reference
    fn classes(&self) -> &Vec<TY> {
        self.classes.as_ref().unwrap()
    }
    /// Get depth of tree
    #[must_use]
    pub fn depth(&self) -> u16 {
        self.tree_classifier.as_ref().map_or(0, |tree| tree.depth())
    }
}

/// The function to measure the quality of a split.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub enum SplitCriterion {
    /// [Gini index](../decision_tree_classifier/index.html)
    #[default]
    Gini,
    /// [Entropy](../decision_tree_classifier/index.html)
    Entropy,
    /// [Classification error](../decision_tree_classifier/index.html)
    ClassificationError,
}

impl<TX: Number + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for DecisionTreeClassifier<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        self.tree_classifier == other.tree_classifier && self.classes == other.classes
    }
}

impl DecisionTreeClassifierParameters {
    /// Split criteria to use when building a tree.
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        self.criterion = criterion;
        self
    }
    /// The maximum depth of the tree.
    pub fn with_max_depth(mut self, max_depth: u16) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
    /// The minimum number of samples required to be at a leaf node.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }
    /// The minimum number of samples required to split an internal node.
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }
}

impl Default for DecisionTreeClassifierParameters {
    fn default() -> Self {
        DecisionTreeClassifierParameters {
            criterion: SplitCriterion::default(),
            max_depth: Option::None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: Option::None,
        }
    }
}

/// DecisionTreeClassifier grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
#[must_use]
pub struct DecisionTreeClassifierSearchParameters {
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
    /// Controls the randomness of the estimator
    pub seed: Vec<Option<u64>>,
}

/// DecisionTreeClassifier grid search iterator
pub struct DecisionTreeClassifierSearchParametersIterator {
    decision_tree_classifier_search_parameters: DecisionTreeClassifierSearchParameters,
    current_criterion: usize,
    current_max_depth: usize,
    current_min_samples_leaf: usize,
    current_min_samples_split: usize,
    current_seed: usize,
}

impl IntoIterator for DecisionTreeClassifierSearchParameters {
    type Item = DecisionTreeClassifierParameters;
    type IntoIter = DecisionTreeClassifierSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        DecisionTreeClassifierSearchParametersIterator {
            decision_tree_classifier_search_parameters: self,
            current_criterion: 0,
            current_max_depth: 0,
            current_min_samples_leaf: 0,
            current_min_samples_split: 0,
            current_seed: 0,
        }
    }
}

impl Iterator for DecisionTreeClassifierSearchParametersIterator {
    type Item = DecisionTreeClassifierParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_criterion
            == self
                .decision_tree_classifier_search_parameters
                .criterion
                .len()
            && self.current_max_depth
                == self
                    .decision_tree_classifier_search_parameters
                    .max_depth
                    .len()
            && self.current_min_samples_leaf
                == self
                    .decision_tree_classifier_search_parameters
                    .min_samples_leaf
                    .len()
            && self.current_min_samples_split
                == self
                    .decision_tree_classifier_search_parameters
                    .min_samples_split
                    .len()
            && self.current_seed == self.decision_tree_classifier_search_parameters.seed.len()
        {
            return None;
        }

        let next = DecisionTreeClassifierParameters {
            criterion: self.decision_tree_classifier_search_parameters.criterion
                [self.current_criterion]
                .clone(),
            max_depth: self.decision_tree_classifier_search_parameters.max_depth
                [self.current_max_depth],
            min_samples_leaf: self
                .decision_tree_classifier_search_parameters
                .min_samples_leaf[self.current_min_samples_leaf],
            min_samples_split: self
                .decision_tree_classifier_search_parameters
                .min_samples_split[self.current_min_samples_split],
            seed: self.decision_tree_classifier_search_parameters.seed[self.current_seed],
        };

        if self.current_criterion + 1
            < self
                .decision_tree_classifier_search_parameters
                .criterion
                .len()
        {
            self.current_criterion += 1;
        } else if self.current_max_depth + 1
            < self
                .decision_tree_classifier_search_parameters
                .max_depth
                .len()
        {
            self.current_criterion = 0;
            self.current_max_depth += 1;
        } else if self.current_min_samples_leaf + 1
            < self
                .decision_tree_classifier_search_parameters
                .min_samples_leaf
                .len()
        {
            self.current_criterion = 0;
            self.current_max_depth = 0;
            self.current_min_samples_leaf += 1;
        } else if self.current_min_samples_split + 1
            < self
                .decision_tree_classifier_search_parameters
                .min_samples_split
                .len()
        {
            self.current_criterion = 0;
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split += 1;
        } else if self.current_seed + 1 < self.decision_tree_classifier_search_parameters.seed.len()
        {
            self.current_criterion = 0;
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_seed += 1;
        } else {
            self.current_criterion += 1;
            self.current_max_depth += 1;
            self.current_min_samples_leaf += 1;
            self.current_min_samples_split += 1;
            self.current_seed += 1;
        }

        Some(next)
    }
}

impl Default for DecisionTreeClassifierSearchParameters {
    fn default() -> Self {
        let default_params = DecisionTreeClassifierParameters::default();

        DecisionTreeClassifierSearchParameters {
            criterion: vec![default_params.criterion],
            max_depth: vec![default_params.max_depth],
            min_samples_leaf: vec![default_params.min_samples_leaf],
            min_samples_split: vec![default_params.min_samples_split],
            seed: vec![default_params.seed],
        }
    }
}

pub(crate) fn which_max(x: &[usize]) -> usize {
    let mut m = x[0];
    let mut which = 0;

    for (i, x_i) in x.iter().enumerate().skip(1) {
        if *x_i > m {
            m = *x_i;
            which = i;
        }
    }

    which
}

impl<TX: Number + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, DecisionTreeClassifierParameters>
    for DecisionTreeClassifier<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            tree_classifier: None,
            num_features: None,
            classes: None,
            _phantom_y: PhantomData,
        }
    }

    fn fit(x: &X, y: &Y, parameters: DecisionTreeClassifierParameters) -> Result<Self, Failed> {
        DecisionTreeClassifier::fit(x, y, parameters)
    }
}

impl<TX: Number + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> Predictor<X, Y>
    for DecisionTreeClassifier<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    DecisionTreeClassifier<TX, TY, X, Y>
{
    /// Build a decision tree classifier from the training data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: DecisionTreeClassifierParameters,
    ) -> Result<DecisionTreeClassifier<TX, TY, X, Y>, Failed> {
        Self::fit_inner(x, y, None, parameters)
    }

    fn fit_inner(
        x: &X,
        y: &Y,
        sample_weights: Option<&[f64]>,
        parameters: DecisionTreeClassifierParameters,
    ) -> Result<DecisionTreeClassifier<TX, TY, X, Y>, Failed> {
        let tree_parameters = BaseTreeParameters {
            max_depth: parameters.max_depth,
            min_samples_leaf: parameters.min_samples_leaf,
            min_samples_split: parameters.min_samples_split,
            seed: parameters.seed,
            splitter: Splitter::Best,
        };

        let classes = y.unique();

        let mut yi: Vec<usize> = vec![0; y.shape()];

        for (i, yi_i) in yi.iter_mut().enumerate().take(y.shape()) {
            let yc = y.get(i);
            *yi_i = classes.iter().position(|c| yc == c).unwrap();
        }

        let k = classes.len();
        let tree = match parameters.criterion {
            SplitCriterion::Gini => BaseTree::fit_inner(
                x,
                &yi,
                sample_weights,
                &GiniCriterion::new(k),
                tree_parameters,
            ),
            SplitCriterion::Entropy => BaseTree::fit_inner(
                x,
                &yi,
                sample_weights,
                &EntropyCriterion::new(k),
                tree_parameters,
            ),
            SplitCriterion::ClassificationError => BaseTree::fit_inner(
                x,
                &yi,
                sample_weights,
                &ClassificationErrorCriterion::new(k),
                tree_parameters,
            ),
        }?;

        Ok(Self {
            tree_classifier: Some(tree),
            num_features: Some(x.shape().1),
            classes: Some(classes),
            _phantom_y: PhantomData,
        })
    }

    /// Predict class value for `x`.
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        match &self.tree_classifier {
            Some(tree) => {
                let mut result = Y::zeros(x.shape().0);

                let (n, _) = x.shape();

                for i in 0..n {
                    result.set(i, self.classes()[*tree.leaf_output_for_row(x, i)]);
                }

                Ok(result)
            }
            None => Err(Failed::predict(
                "'fit' should be called before calling 'predict'",
            )),
        }
    }

    /// Compute feature importances for the fitted tree.
    pub fn compute_feature_importances(&self, normalize: bool) -> Result<Vec<f64>, Failed> {
        match (&self.tree_classifier, self.num_features) {
            (Some(tree), Some(num_features)) => {
                let mut importances = vec![0.0f64; num_features];
                tree.compute_feature_importances(normalize, &mut importances);
                Ok(importances)
            }
            _ => Err(Failed::predict(
                "'fit' should be called before calling 'compute_feature_importances'",
            )),
        }
    }

    /// Predict class probabilities for the input samples.
    ///
    /// # Arguments
    ///
    /// * `x` - The input samples as a matrix where each row is a sample and each column is a feature.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `DenseMatrix<f64>` where each row corresponds to a sample and each column
    /// corresponds to a class. The values represent the probability of the sample belonging to each class.
    ///
    /// # Errors
    ///
    /// Returns an error if at least one row prediction process fails.
    pub fn predict_proba(&self, x: &X) -> Result<DenseMatrix<f64>, Failed> {
        match (&self.tree_classifier, &self.classes) {
            (Some(tree), Some(classes)) => {
                let (n, _) = x.shape();
                let mut result = DenseMatrix::<f64>::zeros(n, classes.len());

                for i in 0..n {
                    result.set((i, *tree.leaf_output_for_row(x, i)), 1.0);
                }

                Ok(result)
            }
            _ => Err(Failed::predict(
                "'fit' should be called before calling 'predict_proba'",
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
    use crate::metrics::accuracy;

    #[test]
    fn search_parameters() {
        let parameters = DecisionTreeClassifierSearchParameters {
            max_depth: vec![Some(10), Some(100)],
            min_samples_split: vec![1, 2],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.max_depth, Some(10));
        assert_eq!(next.min_samples_split, 1);
        let next = iter.next().unwrap();
        assert_eq!(next.max_depth, Some(100));
        assert_eq!(next.min_samples_split, 1);
        let next = iter.next().unwrap();
        assert_eq!(next.max_depth, Some(10));
        assert_eq!(next.min_samples_split, 2);
        let next = iter.next().unwrap();
        assert_eq!(next.max_depth, Some(100));
        assert_eq!(next.min_samples_split, 2);
        assert!(iter.next().is_none());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_predict_proba() {
        let x: DenseMatrix<f64> = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
        ])
        .unwrap();
        let y: Vec<usize> = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let tree = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
        let probabilities = tree.predict_proba(&x).unwrap();

        assert_eq!(probabilities.shape(), (10, 2));

        for row in 0..10 {
            let row_sum: f64 = probabilities.get_row(row).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row probabilities should sum to 1"
            );
        }

        // Check if the first 5 samples have higher probability for class 0
        for i in 0..5 {
            assert!(probabilities.get((i, 0)) > probabilities.get((i, 1)));
        }

        // Check if the last 5 samples have higher probability for class 1
        for i in 5..10 {
            assert!(probabilities.get((i, 1)) > probabilities.get((i, 0)));
        }
    }

    #[test]
    fn full_depth() {
        let x = DenseMatrix::from_2d_vec(&vec![
            vec![1.0_f64],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ])
        .unwrap();
        let y = vec![0, 1, 2, 2, 3, 4];

        let parameters = DecisionTreeClassifierParameters {
            max_depth: Some(3),
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: None,
            criterion: SplitCriterion::Gini,
        };

        let tree = DecisionTreeClassifier::fit(&x, &y, parameters).expect("Fit should work");
        let y_hat = tree.predict(&x).expect("Predict should work");
        assert_eq!(tree.tree_classifier.as_ref().unwrap().num_nodes(), 7);
        assert_eq!(tree.depth(), 3);

        // Tree should have 5 out of 6 examples correct
        assert!((accuracy(&y, &y_hat) - 5.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn min_samples_split_boundary() {
        // A node that holds exactly `min_samples_split` samples must still split.
        let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0usize, 0, 1];

        let parameters = DecisionTreeClassifierParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 3,
            seed: None,
            criterion: SplitCriterion::Gini,
        };

        let tree = DecisionTreeClassifier::fit(&x, &y, parameters).expect("Fit should work");
        assert_eq!(tree.tree_classifier.as_ref().unwrap().num_nodes(), 3);
        assert_eq!(tree.depth(), 2);

        // A node with fewer than `min_samples_split` samples must stay a leaf.
        let parameters = DecisionTreeClassifierParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 4,
            seed: None,
            criterion: SplitCriterion::Gini,
        };

        let tree = DecisionTreeClassifier::fit(&x, &y, parameters).expect("Fit should work");
        assert_eq!(tree.tree_classifier.as_ref().unwrap().num_nodes(), 1);
        assert_eq!(tree.depth(), 0);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(feature = "datasets")]
    fn fit_predict_iris() {
        let x: DenseMatrix<f64> = DenseMatrix::from_2d_array(&[
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
        let y: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        assert_eq!(
            y,
            DecisionTreeClassifier::fit(&x, &y, Default::default())
                .and_then(|t| t.predict(&x))
                .unwrap()
        );

        println!(
            "{:?}",
            //3,
            DecisionTreeClassifier::fit(
                &x,
                &y,
                DecisionTreeClassifierParameters {
                    criterion: SplitCriterion::Entropy,
                    max_depth: Some(3),
                    min_samples_leaf: 1,
                    min_samples_split: 2,
                    seed: Option::None
                }
            )
            .unwrap()
            .depth()
        );
    }

    #[test]
    fn test_random_matrix_with_wrong_rownum() {
        let x_rand: DenseMatrix<f64> = DenseMatrix::<f64>::rand(21, 200);

        let y: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let fail = DecisionTreeClassifier::fit(&x_rand, &y, Default::default());

        assert!(fail.is_err());
    }

    #[test]
    fn test_fit_on_empty_data_returns_error() {
        // 2 rows x 2 features — values are arbitrary; only the empty-row case is under test
        let full = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64, 1.0], vec![0.0, 1.0]]).unwrap();
        let empty = full.take(&[] as &[usize], 0);
        assert_eq!(empty.shape(), (0, 2));

        let y: Vec<u32> = vec![];
        let result = DecisionTreeClassifier::fit(&empty, &y, Default::default());
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error(), FailedError::ParametersError);
    }

    #[test]
    fn test_fit_on_zero_features_returns_error() {
        // 2 rows x 2 features — values are arbitrary; only the zero-feature case is under test
        let full = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64, 1.0], vec![0.0, 1.0]]).unwrap();
        let no_features = full.take(&[] as &[usize], 1);
        assert_eq!(no_features.shape(), (2, 0));

        let y: Vec<u32> = vec![0, 1];
        let result = DecisionTreeClassifier::fit(&no_features, &y, Default::default());
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error(), FailedError::ParametersError);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_predict_baloons() {
        let x: DenseMatrix<f64> = DenseMatrix::from_2d_array(&[
            &[1., 1., 1., 0.],
            &[1., 1., 1., 0.],
            &[1., 1., 1., 1.],
            &[1., 1., 0., 0.],
            &[1., 1., 0., 1.],
            &[1., 0., 1., 0.],
            &[1., 0., 1., 0.],
            &[1., 0., 1., 1.],
            &[1., 0., 0., 0.],
            &[1., 0., 0., 1.],
            &[0., 1., 1., 0.],
            &[0., 1., 1., 0.],
            &[0., 1., 1., 1.],
            &[0., 1., 0., 0.],
            &[0., 1., 0., 1.],
            &[0., 0., 1., 0.],
            &[0., 0., 1., 0.],
            &[0., 0., 1., 1.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 1.],
        ])
        .unwrap();
        let y: Vec<u32> = vec![1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0];

        assert_eq!(
            y,
            DecisionTreeClassifier::fit(&x, &y, Default::default())
                .and_then(|t| t.predict(&x))
                .unwrap()
        );
    }

    #[test]
    fn test_compute_feature_importances() {
        let x: DenseMatrix<f64> = DenseMatrix::from_2d_array(&[
            &[1., 1., 1., 0.],
            &[1., 1., 1., 0.],
            &[1., 1., 1., 1.],
            &[1., 1., 0., 0.],
            &[1., 1., 0., 1.],
            &[1., 0., 1., 0.],
            &[1., 0., 1., 0.],
            &[1., 0., 1., 1.],
            &[1., 0., 0., 0.],
            &[1., 0., 0., 1.],
            &[0., 1., 1., 0.],
            &[0., 1., 1., 0.],
            &[0., 1., 1., 1.],
            &[0., 1., 0., 0.],
            &[0., 1., 0., 1.],
            &[0., 0., 1., 0.],
            &[0., 0., 1., 0.],
            &[0., 0., 1., 1.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 1.],
        ])
        .unwrap();
        let y: Vec<u32> = vec![1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0];
        let tree = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();

        // y = 1 if and only if x2 = 1 and x3 = 0. Thus a split on x2 and a split on x3
        // give the same gain at the root. The random feature order breaks the tie,
        // so the root feature gets 0.2667 and the other feature gets 0.2133.
        let raw = tree
            .compute_feature_importances(false)
            .expect("Feature importances must work");
        assert_eq!(raw[0], 0.);
        assert_eq!(raw[1], 0.);
        let (raw_low, raw_high) = (raw[2].min(raw[3]), raw[2].max(raw[3]));
        assert!((raw_low - 0.21333333333333332).abs() < 1e-12);
        assert!((raw_high - 0.26666666666666666).abs() < 1e-12);

        let normalized = tree
            .compute_feature_importances(true)
            .expect("Feature importances must work");
        assert_eq!(normalized[0], 0.);
        assert_eq!(normalized[1], 0.);
        let (norm_low, norm_high) = (
            normalized[2].min(normalized[3]),
            normalized[2].max(normalized[3]),
        );
        assert!((norm_low - 0.4444444444444444).abs() < 1e-12);
        assert!((norm_high - 0.5555555555555556).abs() < 1e-12);

        // Normalization must keep the order of the features.
        assert_eq!(raw[2] > raw[3], normalized[2] > normalized[3]);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_without_fit_should_not_panic() {
        let tree: DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            DecisionTreeClassifier::new();
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
    fn predict_proba_without_fit_should_not_panic() {
        let tree: DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            DecisionTreeClassifier::new();
        let x = DenseMatrix::from_2d_array(&[&[1.0f64]]).expect("Construction of x should work");
        let yhat = tree.predict_proba(&x);
        assert!(yhat.is_err());
        let msg = "'fit' should be called before calling 'predict_proba'";
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
            &[1., 1., 1., 0.],
            &[1., 1., 1., 0.],
            &[1., 1., 1., 1.],
            &[1., 1., 0., 0.],
            &[1., 1., 0., 1.],
            &[1., 0., 1., 0.],
            &[1., 0., 1., 0.],
            &[1., 0., 1., 1.],
            &[1., 0., 0., 0.],
            &[1., 0., 0., 1.],
            &[0., 1., 1., 0.],
            &[0., 1., 1., 0.],
            &[0., 1., 1., 1.],
            &[0., 1., 0., 0.],
            &[0., 1., 0., 1.],
            &[0., 0., 1., 0.],
            &[0., 0., 1., 0.],
            &[0., 0., 1., 1.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 1.],
        ])
        .unwrap();
        let y = vec![1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0];

        let tree = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();

        let deserialized_tree: DecisionTreeClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>> =
            postcard::from_bytes(&postcard::to_allocvec(&tree).unwrap()).unwrap();

        assert_eq!(tree, deserialized_tree);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_predict_labels_outside_class_index_range() {
        // The labels are not 0..k-1: one is negative, one is equal to k = 3 and one is large.
        // The tree must use the position of the label in `classes`, not the label itself.
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

        let tree = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
        assert_eq!(tree.predict(&x).unwrap(), y);

        // The columns of predict_proba are in the order of the sorted classes: -7, 3, 100
        let probabilities = tree.predict_proba(&x).unwrap();
        assert_eq!(probabilities.shape(), (9, 3));
        for (row, expected_column) in [0, 0, 0, 1, 1, 1, 2, 2, 2].iter().enumerate() {
            for column in 0..3 {
                let expected = if column == *expected_column { 1.0 } else { 0.0 };
                assert_eq!(*probabilities.get((row, column)), expected);
            }
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_with_single_class_gives_trivial_classifier() {
        // As in scikit-learn, one class is not an error: the tree is one leaf.
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

        let tree = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
        assert_eq!(tree.depth(), 0);
        assert_eq!(tree.tree_classifier.as_ref().unwrap().num_nodes(), 1);

        // The tree predicts the single class, also for data that is not in the training set
        let x_new =
            DenseMatrix::from_2d_array(&[&[-10.0, 10.0], &[2.5, 2.5], &[100.0, -1.0]]).unwrap();
        assert_eq!(tree.predict(&x).unwrap(), y);
        assert_eq!(tree.predict(&x_new).unwrap(), vec![5; 3]);

        // One column with probability 1
        let probabilities = tree.predict_proba(&x_new).unwrap();
        assert_eq!(probabilities.shape(), (3, 1));
        for row in 0..3 {
            assert_eq!(*probabilities.get((row, 0)), 1.0);
        }

        // Without a split, no feature is important
        assert_eq!(
            tree.compute_feature_importances(false).unwrap(),
            vec![0.0; 2]
        );
        assert_eq!(
            tree.compute_feature_importances(true).unwrap(),
            vec![0.0; 2]
        );
    }
}
