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
use std::collections::LinkedList;
use std::default::Default;
use std::fmt::Debug;
use std::marker::PhantomData;

use rand::seq::SliceRandom;
use rand::Rng;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2, MutArrayView1};
use crate::numbers::basenum::Number;
use crate::rand_custom::get_rng_impl;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// Parameters of Decision Tree
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
    nodes: Vec<Node>,
    parameters: Option<DecisionTreeClassifierParameters>,
    num_classes: usize,
    classes: Vec<TY>,
    depth: u16,
    num_features: usize,
    _phantom_tx: PhantomData<TX>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    DecisionTreeClassifier<TX, TY, X, Y>
{
    /// Get nodes, return a shared reference
    fn nodes(&self) -> &Vec<Node> {
        self.nodes.as_ref()
    }
    /// Get parameters, return a shared reference
    fn parameters(&self) -> &DecisionTreeClassifierParameters {
        self.parameters.as_ref().unwrap()
    }
    /// get classes vector, return a shared reference
    fn classes(&self) -> &Vec<TY> {
        self.classes.as_ref()
    }
    /// Get depth of tree
    pub fn depth(&self) -> u16 {
        self.depth
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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
struct Node {
    output: usize,
    n_node_samples: usize,
    split_feature: usize,
    split_value: Option<f64>,
    split_score: Option<f64>,
    true_child: Option<usize>,
    false_child: Option<usize>,
    impurity: Option<f64>,
}

impl<TX: Number + PartialOrd, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for DecisionTreeClassifier<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        if self.depth != other.depth
            || self.num_classes != other.num_classes
            || self.nodes().len() != other.nodes().len()
        {
            false
        } else {
            self.classes()
                .iter()
                .zip(other.classes().iter())
                .all(|(a, b)| a == b)
                && self
                    .nodes()
                    .iter()
                    .zip(other.nodes().iter())
                    .all(|(a, b)| a == b)
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.output == other.output
            && self.split_feature == other.split_feature
            && match (self.split_value, other.split_value) {
                (Some(a), Some(b)) => (a - b).abs() < std::f64::EPSILON,
                (None, None) => true,
                _ => false,
            }
            && match (self.split_score, other.split_score) {
                (Some(a), Some(b)) => (a - b).abs() < std::f64::EPSILON,
                (None, None) => true,
                _ => false,
            }
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

impl Node {
    fn new(output: usize, n_node_samples: usize) -> Self {
        Node {
            output,
            n_node_samples,
            split_feature: 0,
            split_value: Option::None,
            split_score: Option::None,
            true_child: Option::None,
            false_child: Option::None,
            impurity: Option::None,
        }
    }
}

struct NodeVisitor<'a, TX: Number + PartialOrd, X: Array2<TX>> {
    x: &'a X,
    y: &'a [usize],
    node: usize,
    samples: Vec<usize>,
    order: &'a [Vec<usize>],
    true_child_output: usize,
    false_child_output: usize,
    level: u16,
    phantom: PhantomData<&'a TX>,
}

fn impurity(criterion: &SplitCriterion, count: &[usize], n: usize) -> f64 {
    let mut impurity = 0f64;

    match criterion {
        SplitCriterion::Gini => {
            impurity = 1f64;
            for count_i in count.iter() {
                if *count_i > 0 {
                    let p = *count_i as f64 / n as f64;
                    impurity -= p * p;
                }
            }
        }

        SplitCriterion::Entropy => {
            for count_i in count.iter() {
                if *count_i > 0 {
                    let p = *count_i as f64 / n as f64;
                    impurity -= p * p.log2();
                }
            }
        }
        SplitCriterion::ClassificationError => {
            for count_i in count.iter() {
                if *count_i > 0 {
                    impurity = impurity.max(*count_i as f64 / n as f64);
                }
            }
            impurity = (1f64 - impurity).abs();
        }
    }

    impurity
}

impl<'a, TX: Number + PartialOrd, X: Array2<TX>> NodeVisitor<'a, TX, X> {
    fn new(
        node_id: usize,
        samples: Vec<usize>,
        order: &'a [Vec<usize>],
        x: &'a X,
        y: &'a [usize],
        level: u16,
    ) -> Self {
        NodeVisitor {
            x,
            y,
            node: node_id,
            samples,
            order,
            true_child_output: 0,
            false_child_output: 0,
            level,
            phantom: PhantomData,
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
            nodes: vec![],
            parameters: Option::None,
            num_classes: 0usize,
            classes: vec![],
            depth: 0u16,
            num_features: 0usize,
            _phantom_tx: PhantomData,
            _phantom_x: PhantomData,
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
        let (x_nrows, num_attributes) = x.shape();
        if x_nrows != y.shape() {
            return Err(Failed::fit("Size of x should equal size of y"));
        }

        let samples = vec![1; x_nrows];
        DecisionTreeClassifier::fit_weak_learner(x, y, samples, num_attributes, parameters)
    }

    pub(crate) fn fit_weak_learner(
        x: &X,
        y: &Y,
        samples: Vec<usize>,
        mtry: usize,
        parameters: DecisionTreeClassifierParameters,
    ) -> Result<DecisionTreeClassifier<TX, TY, X, Y>, Failed> {
        let y_ncols = y.shape();
        let (_, num_attributes) = x.shape();
        let classes = y.unique();
        let k = classes.len();
        if k < 2 {
            return Err(Failed::fit(&format!(
                "Incorrect number of classes: {k}. Should be >= 2."
            )));
        }

        let mut rng = get_rng_impl(parameters.seed);
        let mut yi: Vec<usize> = vec![0; y_ncols];

        for (i, yi_i) in yi.iter_mut().enumerate().take(y_ncols) {
            let yc = y.get(i);
            *yi_i = classes.iter().position(|c| yc == c).unwrap();
        }

        let mut change_nodes: Vec<Node> = Vec::new();

        let mut count = vec![0; k];
        for i in 0..y_ncols {
            count[yi[i]] += samples[i];
        }

        let root = Node::new(which_max(&count), y_ncols);
        change_nodes.push(root);
        let mut order: Vec<Vec<usize>> = Vec::new();

        for i in 0..num_attributes {
            let mut col_i: Vec<TX> = x.get_col(i).iterator(0).copied().collect();
            order.push(col_i.argsort_mut());
        }

        let mut tree = DecisionTreeClassifier {
            nodes: change_nodes,
            parameters: Some(parameters),
            num_classes: k,
            classes,
            depth: 0u16,
            num_features: num_attributes,
            _phantom_tx: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        };

        let mut visitor = NodeVisitor::<TX, X>::new(0, samples, &order, x, &yi, 1);

        let mut visitor_queue: LinkedList<NodeVisitor<'_, TX, X>> = LinkedList::new();

        if tree.find_best_cutoff(&mut visitor, mtry, &mut rng) {
            visitor_queue.push_back(visitor);
        }

        while tree.depth() < tree.parameters().max_depth.unwrap_or(std::u16::MAX) {
            match visitor_queue.pop_front() {
                Some(node) => tree.split(node, mtry, &mut visitor_queue, &mut rng),
                None => break,
            };
        }

        Ok(tree)
    }

    /// Predict class value for `x`.
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let mut result = Y::zeros(x.shape().0);

        let (n, _) = x.shape();

        for i in 0..n {
            result.set(i, self.classes()[self.predict_for_row(x, i)]);
        }

        Ok(result)
    }

    pub(crate) fn predict_for_row(&self, x: &X, row: usize) -> usize {
        let mut result = 0;
        let mut queue: LinkedList<usize> = LinkedList::new();

        queue.push_back(0);

        while !queue.is_empty() {
            match queue.pop_front() {
                Some(node_id) => {
                    let node = &self.nodes()[node_id];
                    if node.true_child.is_none() && node.false_child.is_none() {
                        result = node.output;
                    } else if x.get((row, node.split_feature)).to_f64().unwrap()
                        <= node.split_value.unwrap_or(std::f64::NAN)
                    {
                        queue.push_back(node.true_child.unwrap());
                    } else {
                        queue.push_back(node.false_child.unwrap());
                    }
                }
                None => break,
            };
        }

        result
    }

    fn find_best_cutoff(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, X>,
        mtry: usize,
        rng: &mut impl Rng,
    ) -> bool {
        let (n_rows, n_attr) = visitor.x.shape();

        let mut label = Option::None;
        let mut is_pure = true;
        for i in 0..n_rows {
            if visitor.samples[i] > 0 {
                if label.is_none() {
                    label = Option::Some(visitor.y[i]);
                } else if visitor.y[i] != label.unwrap() {
                    is_pure = false;
                    break;
                }
            }
        }

        let n = visitor.samples.iter().sum();
        let mut count = vec![0; self.num_classes];
        let mut false_count = vec![0; self.num_classes];
        for i in 0..n_rows {
            if visitor.samples[i] > 0 {
                count[visitor.y[i]] += visitor.samples[i];
            }
        }

        self.nodes[visitor.node].impurity = Some(impurity(&self.parameters().criterion, &count, n));

        if is_pure {
            return false;
        }

        if n <= self.parameters().min_samples_split {
            return false;
        }

        let mut variables = (0..n_attr).collect::<Vec<_>>();

        if mtry < n_attr {
            variables.shuffle(rng);
        }

        for variable in variables.iter().take(mtry) {
            self.find_best_split(visitor, n, &count, &mut false_count, *variable);
        }

        self.nodes()[visitor.node].split_score.is_some()
    }

    fn find_best_split(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, X>,
        n: usize,
        count: &[usize],
        false_count: &mut [usize],
        j: usize,
    ) {
        let mut true_count = vec![0; self.num_classes];
        let mut prevx = Option::None;
        let mut prevy = 0;

        for i in visitor.order[j].iter() {
            if visitor.samples[*i] > 0 {
                let x_ij = *visitor.x.get((*i, j));

                if prevx.is_none() || x_ij == prevx.unwrap() || visitor.y[*i] == prevy {
                    prevx = Some(x_ij);
                    prevy = visitor.y[*i];
                    true_count[visitor.y[*i]] += visitor.samples[*i];
                    continue;
                }

                let tc = true_count.iter().sum();
                let fc = n - tc;

                if tc < self.parameters().min_samples_leaf
                    || fc < self.parameters().min_samples_leaf
                {
                    prevx = Some(x_ij);
                    prevy = visitor.y[*i];
                    true_count[visitor.y[*i]] += visitor.samples[*i];
                    continue;
                }

                for l in 0..self.num_classes {
                    false_count[l] = count[l] - true_count[l];
                }

                let true_label = which_max(&true_count);
                let false_label = which_max(false_count);
                let parent_impurity = self.nodes()[visitor.node].impurity.unwrap();
                let gain = parent_impurity
                    - tc as f64 / n as f64
                        * impurity(&self.parameters().criterion, &true_count, tc)
                    - fc as f64 / n as f64
                        * impurity(&self.parameters().criterion, false_count, fc);

                if self.nodes()[visitor.node].split_score.is_none()
                    || gain > self.nodes()[visitor.node].split_score.unwrap()
                {
                    self.nodes[visitor.node].split_feature = j;
                    self.nodes[visitor.node].split_value =
                        Option::Some((x_ij + prevx.unwrap()).to_f64().unwrap() / 2f64);
                    self.nodes[visitor.node].split_score = Option::Some(gain);

                    visitor.true_child_output = true_label;
                    visitor.false_child_output = false_label;
                }

                prevx = Some(x_ij);
                prevy = visitor.y[*i];
                true_count[visitor.y[*i]] += visitor.samples[*i];
            }
        }
    }

    fn split<'a>(
        &mut self,
        mut visitor: NodeVisitor<'a, TX, X>,
        mtry: usize,
        visitor_queue: &mut LinkedList<NodeVisitor<'a, TX, X>>,
        rng: &mut impl Rng,
    ) -> bool {
        let (n, _) = visitor.x.shape();
        let mut tc = 0;
        let mut fc = 0;
        let mut true_samples: Vec<usize> = vec![0; n];

        for (i, true_sample) in true_samples.iter_mut().enumerate().take(n) {
            if visitor.samples[i] > 0 {
                if visitor
                    .x
                    .get((i, self.nodes()[visitor.node].split_feature))
                    .to_f64()
                    .unwrap()
                    <= self.nodes()[visitor.node]
                        .split_value
                        .unwrap_or(std::f64::NAN)
                {
                    *true_sample = visitor.samples[i];
                    tc += *true_sample;
                    visitor.samples[i] = 0;
                } else {
                    fc += visitor.samples[i];
                }
            }
        }

        if tc < self.parameters().min_samples_leaf || fc < self.parameters().min_samples_leaf {
            self.nodes[visitor.node].split_feature = 0;
            self.nodes[visitor.node].split_value = Option::None;
            self.nodes[visitor.node].split_score = Option::None;

            return false;
        }

        let true_child_idx = self.nodes().len();

        self.nodes.push(Node::new(visitor.true_child_output, tc));
        let false_child_idx = self.nodes().len();
        self.nodes.push(Node::new(visitor.false_child_output, fc));
        self.nodes[visitor.node].true_child = Some(true_child_idx);
        self.nodes[visitor.node].false_child = Some(false_child_idx);

        self.depth = u16::max(self.depth, visitor.level + 1);

        let mut true_visitor = NodeVisitor::<TX, X>::new(
            true_child_idx,
            true_samples,
            visitor.order,
            visitor.x,
            visitor.y,
            visitor.level + 1,
        );

        if self.find_best_cutoff(&mut true_visitor, mtry, rng) {
            visitor_queue.push_back(true_visitor);
        }

        let mut false_visitor = NodeVisitor::<TX, X>::new(
            false_child_idx,
            visitor.samples,
            visitor.order,
            visitor.x,
            visitor.y,
            visitor.level + 1,
        );

        if self.find_best_cutoff(&mut false_visitor, mtry, rng) {
            visitor_queue.push_back(false_visitor);
        }

        true
    }

    /// Compute feature importances for the fitted tree.
    pub fn compute_feature_importances(&self, normalize: bool) -> Vec<f64> {
        let mut importances = vec![0f64; self.num_features];

        for node in self.nodes().iter() {
            if node.true_child.is_none() && node.false_child.is_none() {
                continue;
            }
            let left = &self.nodes()[node.true_child.unwrap()];
            let right = &self.nodes()[node.false_child.unwrap()];

            importances[node.split_feature] += node.n_node_samples as f64 * node.impurity.unwrap()
                - left.n_node_samples as f64 * left.impurity.unwrap()
                - right.n_node_samples as f64 * right.impurity.unwrap();
        }
        for item in importances.iter_mut() {
            *item /= self.nodes()[0].n_node_samples as f64;
        }
        if normalize {
            let sum = importances.iter().sum::<f64>();
            for importance in importances.iter_mut() {
                *importance /= sum;
            }
        }
        importances
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

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
    fn gini_impurity() {
        assert!((impurity(&SplitCriterion::Gini, &[7, 3], 10) - 0.42).abs() < std::f64::EPSILON);
        assert!(
            (impurity(&SplitCriterion::Entropy, &[7, 3], 10) - 0.8812908992306927).abs()
                < std::f64::EPSILON
        );
        assert!(
            (impurity(&SplitCriterion::ClassificationError, &[7, 3], 10) - 0.3).abs()
                < std::f64::EPSILON
        );
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
            .depth
        );
    }

    #[test]
    fn test_random_matrix_with_wrong_rownum() {
        let x_rand: DenseMatrix<f64> = DenseMatrix::<f64>::rand(21, 200);

        let y: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let fail = DecisionTreeClassifier::fit(&x_rand, &y, Default::default());

        assert!(fail.is_err());
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
        ]).unwrap();
        let y: Vec<u32> = vec![1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0];
        let tree = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
        assert_eq!(
            tree.compute_feature_importances(false),
            vec![0., 0., 0.21333333333333332, 0.26666666666666666]
        );
        assert_eq!(
            tree.compute_feature_importances(true),
            vec![0., 0., 0.4444444444444444, 0.5555555555555556]
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
            bincode::deserialize(&bincode::serialize(&tree).unwrap()).unwrap();

        assert_eq!(tree, deserialized_tree);
    }
}
