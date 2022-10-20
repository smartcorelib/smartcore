//! # Decision Tree Regressor
//!
//! The process of building a decision tree can be simplified to these two steps:
//!
//! 1. Divide the predictor space \\(X\\) into K distinct and non-overlapping regions, \\(R_1, R_2, ..., R_K\\).
//! 1. For every observation that falls into the region \\(R_k\\), we make the same prediction, which is simply the mean of the response values for the training observations in \\(R_k\\).
//!
//! Regions \\(R_1, R_2, ..., R_K\\) are build in such a way that minimizes the residual sum of squares (RSS) given by
//!
//! \\[RSS = \sum_{k=1}^K\sum_{i \in R_k} (y_i - \hat{y}_{Rk})^2\\]
//!
//! where \\(\hat{y}_{Rk}\\) is the mean response for the training observations withing region _k_.
//!
//! SmartCore uses recursive binary splitting approach to build \\(R_1, R_2, ..., R_K\\) regions. The approach begins at the top of the tree and then successively splits the predictor space
//! one predictor at a time. At each step of the tree-building process, the best split is made at that particular step, rather than looking ahead and picking a split that will lead to a better
//! tree in some future step.
//!
//! Example:
//!
//! ```
//! use rand::thread_rng;
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::tree::decision_tree_regressor::*;
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
//!        ]);
//! let y: Vec<f64> = vec![
//!             83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0,
//!             101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9,
//!        ];
//!
//! let tree = DecisionTreeRegressor::fit(&x, &y, Default::default()).unwrap();
//!
//! let y_hat = tree.predict(&x).unwrap(); // use the same data for prediction
//! ```
//!
//! ## References:
//!
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
/// Parameters of Regression Tree
pub struct DecisionTreeRegressorParameters {
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

/// Regression Tree
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct DecisionTreeRegressor<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
{
    nodes: Vec<Node>,
    parameters: Option<DecisionTreeRegressorParameters>,
    depth: u16,
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> DecisionTreeRegressor<TX, TY, X, Y> {
    /// Get nodes, return a shared reference
    fn nodes(&self) -> &Vec<Node> {
        self.nodes.as_ref()
    }
    /// Get parameters, return a shared reference
    fn parameters(&self) -> &DecisionTreeRegressorParameters {
        self.parameters.as_ref().unwrap()
    }
    /// Get estimate of intercept, return value
    fn depth(&self) -> u16 {
        self.depth
    }

}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
struct Node {
    index: usize,
    output: f64,
    split_feature: usize,
    split_value: Option<f64>,
    split_score: Option<f64>,
    true_child: Option<usize>,
    false_child: Option<usize>,
}

impl DecisionTreeRegressorParameters {
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

impl Default for DecisionTreeRegressorParameters {
    fn default() -> Self {
        DecisionTreeRegressorParameters {
            max_depth: Option::None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: Option::None,
        }
    }
}

/// DecisionTreeRegressor grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressorSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Tree max depth. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub max_depth: Vec<Option<u16>>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_leaf: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to split an internal node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_split: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Controls the randomness of the estimator
    pub seed: Vec<Option<u64>>,
}

/// DecisionTreeRegressor grid search iterator
pub struct DecisionTreeRegressorSearchParametersIterator {
    decision_tree_regressor_search_parameters: DecisionTreeRegressorSearchParameters,
    current_max_depth: usize,
    current_min_samples_leaf: usize,
    current_min_samples_split: usize,
    current_seed: usize,
}

impl IntoIterator for DecisionTreeRegressorSearchParameters {
    type Item = DecisionTreeRegressorParameters;
    type IntoIter = DecisionTreeRegressorSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        DecisionTreeRegressorSearchParametersIterator {
            decision_tree_regressor_search_parameters: self,
            current_max_depth: 0,
            current_min_samples_leaf: 0,
            current_min_samples_split: 0,
            current_seed: 0,
        }
    }
}

impl Iterator for DecisionTreeRegressorSearchParametersIterator {
    type Item = DecisionTreeRegressorParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_max_depth
            == self
                .decision_tree_regressor_search_parameters
                .max_depth
                .len()
            && self.current_min_samples_leaf
                == self
                    .decision_tree_regressor_search_parameters
                    .min_samples_leaf
                    .len()
            && self.current_min_samples_split
                == self
                    .decision_tree_regressor_search_parameters
                    .min_samples_split
                    .len()
            && self.current_seed == self.decision_tree_regressor_search_parameters.seed.len()
        {
            return None;
        }

        let next = DecisionTreeRegressorParameters {
            max_depth: self.decision_tree_regressor_search_parameters.max_depth
                [self.current_max_depth],
            min_samples_leaf: self
                .decision_tree_regressor_search_parameters
                .min_samples_leaf[self.current_min_samples_leaf],
            min_samples_split: self
                .decision_tree_regressor_search_parameters
                .min_samples_split[self.current_min_samples_split],
            seed: self.decision_tree_regressor_search_parameters.seed[self.current_seed],
        };

        if self.current_max_depth + 1
            < self
                .decision_tree_regressor_search_parameters
                .max_depth
                .len()
        {
            self.current_max_depth += 1;
        } else if self.current_min_samples_leaf + 1
            < self
                .decision_tree_regressor_search_parameters
                .min_samples_leaf
                .len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf += 1;
        } else if self.current_min_samples_split + 1
            < self
                .decision_tree_regressor_search_parameters
                .min_samples_split
                .len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split += 1;
        } else if self.current_seed + 1 < self.decision_tree_regressor_search_parameters.seed.len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_seed += 1;
        } else {
            self.current_max_depth += 1;
            self.current_min_samples_leaf += 1;
            self.current_min_samples_split += 1;
            self.current_seed += 1;
        }

        Some(next)
    }
}

impl Default for DecisionTreeRegressorSearchParameters {
    fn default() -> Self {
        let default_params = DecisionTreeRegressorParameters::default();

        DecisionTreeRegressorSearchParameters {
            max_depth: vec![default_params.max_depth],
            min_samples_leaf: vec![default_params.min_samples_leaf],
            min_samples_split: vec![default_params.min_samples_split],
            seed: vec![default_params.seed],
        }
    }
}

impl Node {
    fn new(index: usize, output: f64) -> Self {
        Node {
            index,
            output,
            split_feature: 0,
            split_value: Option::None,
            split_score: Option::None,
            true_child: Option::None,
            false_child: Option::None,
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        (self.output - other.output).abs() < std::f64::EPSILON
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

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for DecisionTreeRegressor<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        if self.depth != other.depth || self.nodes().len() != other.nodes().len() {
            false
        } else {
            self.nodes()
                .iter()
                .zip(other.nodes().iter())
                .all(|(a, b)| a == b)
        }
    }
}

struct NodeVisitor<'a, TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    x: &'a X,
    y: &'a Y,
    node: usize,
    samples: Vec<usize>,
    order: &'a [Vec<usize>],
    true_child_output: f64,
    false_child_output: f64,
    level: u16,
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
}

impl<'a, TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    NodeVisitor<'a, TX, TY, X, Y>
{
    fn new(
        node_id: usize,
        samples: Vec<usize>,
        order: &'a [Vec<usize>],
        x: &'a X,
        y: &'a Y,
        level: u16,
    ) -> Self {
        NodeVisitor {
            x,
            y,
            node: node_id,
            samples,
            order,
            true_child_output: 0f64,
            false_child_output: 0f64,
            level,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
        }
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, DecisionTreeRegressorParameters>
    for DecisionTreeRegressor<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            nodes: vec![],
            parameters: Option::None,
            depth: 0u16,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        }
    }

    fn fit(x: &X, y: &Y, parameters: DecisionTreeRegressorParameters) -> Result<Self, Failed> {
        DecisionTreeRegressor::fit(x, y, parameters)
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> Predictor<X, Y>
    for DecisionTreeRegressor<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    DecisionTreeRegressor<TX, TY, X, Y>
{
    /// Build a decision tree regressor from the training data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: DecisionTreeRegressorParameters,
    ) -> Result<DecisionTreeRegressor<TX, TY, X, Y>, Failed> {
        let (x_nrows, num_attributes) = x.shape();
        let samples = vec![1; x_nrows];
        DecisionTreeRegressor::fit_weak_learner(x, y, samples, num_attributes, parameters)
    }

    pub(crate) fn fit_weak_learner(
        x: &X,
        y: &Y,
        samples: Vec<usize>,
        mtry: usize,
        parameters: DecisionTreeRegressorParameters,
    ) -> Result<DecisionTreeRegressor<TX, TY, X, Y>, Failed> {
        let y_m = y.clone();

        let y_ncols = y_m.shape();
        let (_, num_attributes) = x.shape();

        let mut nodes: Vec<Node> = Vec::new();
        let mut rng = get_rng_impl(parameters.seed);

        let mut n = 0;
        let mut sum = 0f64;
        for (i, sample_i) in samples.iter().enumerate().take(y_ncols) {
            n += *sample_i;
            sum += *sample_i as f64 * y_m.get(i).to_f64().unwrap();
        }

        let root = Node::new(0, sum / (n as f64));
        nodes.push(root);
        let mut order: Vec<Vec<usize>> = Vec::new();

        for i in 0..num_attributes {
            let mut col_i: Vec<TX> = x.get_col(i).iterator(0).copied().collect();
            order.push(col_i.argsort_mut());
        }

        let mut tree = DecisionTreeRegressor {
            nodes,
            parameters: Some(parameters),
            depth: 0u16,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        };

        let mut visitor = NodeVisitor::<TX, TY, X, Y>::new(0, samples, &order, x, &y_m, 1);

        let mut visitor_queue: LinkedList<NodeVisitor<'_, TX, TY, X, Y>> = LinkedList::new();

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

    /// Predict regression value for `x`.
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let mut result = Y::zeros(x.shape().0);

        let (n, _) = x.shape();

        for i in 0..n {
            result.set(i, self.predict_for_row(x, i));
        }

        Ok(result)
    }

    pub(crate) fn predict_for_row(&self, x: &X, row: usize) -> TY {
        let mut result = 0f64;
        let mut queue: LinkedList<usize> = LinkedList::new();

        queue.push_back(0);

        while !queue.is_empty() {
            match queue.pop_front() {
                Some(node_id) => {
                    let node = &self.nodes()[node_id];
                    if node.true_child == None && node.false_child == None {
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

        TY::from_f64(result).unwrap()
    }

    fn find_best_cutoff(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, TY, X, Y>,
        mtry: usize,
        rng: &mut impl Rng,
    ) -> bool {
        let (_, n_attr) = visitor.x.shape();

        let n: usize = visitor.samples.iter().sum();

        if n < self.parameters().min_samples_split {
            return false;
        }

        let sum = self.nodes()[visitor.node].output * n as f64;

        let mut variables = (0..n_attr).collect::<Vec<_>>();

        if mtry < n_attr {
            variables.shuffle(rng);
        }

        let parent_gain =
            n as f64 * self.nodes()[visitor.node].output * self.nodes()[visitor.node].output;

        for variable in variables.iter().take(mtry) {
            self.find_best_split(visitor, n, sum, parent_gain, *variable);
        }

        self.nodes()[visitor.node].split_score != Option::None
    }

    fn find_best_split(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, TY, X, Y>,
        n: usize,
        sum: f64,
        parent_gain: f64,
        j: usize,
    ) {
        let mut true_sum = 0f64;
        let mut true_count = 0;
        let mut prevx = Option::None;

        for i in visitor.order[j].iter() {
            if visitor.samples[*i] > 0 {
                let x_ij = *visitor.x.get((*i, j));

                if prevx.is_none() || x_ij == prevx.unwrap() {
                    prevx = Some(x_ij);
                    true_count += visitor.samples[*i];
                    true_sum += visitor.samples[*i] as f64 * visitor.y.get(*i).to_f64().unwrap();
                    continue;
                }

                let false_count = n - true_count;

                if true_count < self.parameters().min_samples_leaf
                    || false_count < self.parameters().min_samples_leaf
                {
                    prevx = Some(x_ij);
                    true_count += visitor.samples[*i];
                    true_sum += visitor.samples[*i] as f64 * visitor.y.get(*i).to_f64().unwrap();
                    continue;
                }

                let true_mean = true_sum / true_count as f64;
                let false_mean = (sum - true_sum) / false_count as f64;

                let gain = (true_count as f64 * true_mean * true_mean
                    + false_count as f64 * false_mean * false_mean)
                    - parent_gain;

                if self.nodes()[visitor.node].split_score.is_none()
                    || gain > self.nodes()[visitor.node].split_score.unwrap()
                {

                    self.nodes[visitor.node].split_feature = j;
                    self.nodes[visitor.node].split_value =
                        Option::Some((x_ij + prevx.unwrap()).to_f64().unwrap() / 2f64);
                    self.nodes[visitor.node].split_score = Option::Some(gain);

                    visitor.true_child_output = true_mean;
                    visitor.false_child_output = false_mean;
                }

                prevx = Some(x_ij);
                true_sum += visitor.samples[*i] as f64 * visitor.y.get(*i).to_f64().unwrap();
                true_count += visitor.samples[*i];
            }
        }
    }

    fn split<'a>(
        &mut self,
        mut visitor: NodeVisitor<'a, TX, TY, X, Y>,
        mtry: usize,
        visitor_queue: &mut LinkedList<NodeVisitor<'a, TX, TY, X, Y>>,
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

        self.nodes
            .push(Node::new(true_child_idx, visitor.true_child_output));
        let false_child_idx = self.nodes().len();
        self.nodes
            .push(Node::new(false_child_idx, visitor.false_child_output));

        self.nodes[visitor.node].true_child = Some(true_child_idx);
        self.nodes[visitor.node].false_child = Some(false_child_idx);

        self.depth = u16::max(self.depth, visitor.level + 1);

        let mut true_visitor = NodeVisitor::<TX, TY, X, Y>::new(
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

        let mut false_visitor = NodeVisitor::<TX, TY, X, Y>::new(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn search_parameters() {
        let parameters = DecisionTreeRegressorSearchParameters {
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
        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let y_hat = DecisionTreeRegressor::fit(&x, &y, Default::default())
            .and_then(|t| t.predict(&x))
            .unwrap();

        for i in 0..y_hat.len() {
            assert!((y_hat[i] - y[i]).abs() < 0.1);
        }

        let expected_y = vec![
            87.3, 87.3, 87.3, 87.3, 98.9, 98.9, 98.9, 98.9, 98.9, 107.9, 107.9, 107.9, 114.85,
            114.85, 114.85, 114.85,
        ];
        let y_hat = DecisionTreeRegressor::fit(
            &x,
            &y,
            DecisionTreeRegressorParameters {
                max_depth: Option::None,
                min_samples_leaf: 2,
                min_samples_split: 6,
                seed: Option::None,
            },
        )
        .and_then(|t| t.predict(&x))
        .unwrap();

        for i in 0..y_hat.len() {
            assert!((y_hat[i] - expected_y[i]).abs() < 0.1);
        }

        let expected_y = vec![
            83.0, 88.35, 88.35, 89.5, 97.15, 97.15, 99.5, 99.5, 101.2, 104.6, 109.6, 109.6, 113.4,
            113.4, 116.30, 116.30,
        ];
        let y_hat = DecisionTreeRegressor::fit(
            &x,
            &y,
            DecisionTreeRegressorParameters {
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 3,
                seed: Option::None,
            },
        )
        .and_then(|t| t.predict(&x))
        .unwrap();

        for i in 0..y_hat.len() {
            assert!((y_hat[i] - expected_y[i]).abs() < 0.1);
        }
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
        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let tree = DecisionTreeRegressor::fit(&x, &y, Default::default()).unwrap();

        let deserialized_tree: DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            bincode::deserialize(&bincode::serialize(&tree).unwrap()).unwrap();

        assert_eq!(tree, deserialized_tree);
    }
}
