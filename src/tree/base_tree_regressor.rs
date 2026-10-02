use std::collections::VecDeque;
use std::default::Default;
use std::fmt::Debug;
use std::marker::PhantomData;

use rand::RngExt;
use rand::seq::SliceRandom;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, Array2, MutArrayView1};
use crate::numbers::basenum::Number;
use crate::rand_custom::get_rng_impl;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub enum Splitter {
    Random,
    #[default]
    Best,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// Parameters of Regression base_tree
#[must_use]
pub struct BaseTreeRegressorParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// The maximum depth of the base_tree.
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
    #[cfg_attr(feature = "serde", serde(default))]
    /// Determines the strategy used to choose the split at each node.
    pub splitter: Splitter,
}

/// Regression base_tree
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct BaseTreeRegressor<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    nodes: Vec<Node>,
    parameters: Option<BaseTreeRegressorParameters>,
    depth: u16,
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    BaseTreeRegressor<TX, TY, X, Y>
{
    /// Get nodes, return a shared reference
    fn nodes(&self) -> &Vec<Node> {
        self.nodes.as_ref()
    }
    /// Get parameters, return a shared reference
    fn parameters(&self) -> &BaseTreeRegressorParameters {
        self.parameters.as_ref().unwrap()
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
struct Node {
    output: f64,
    split_feature: usize,
    split_value: Option<f64>,
    split_score: Option<f64>,
    true_child: Option<usize>,
    false_child: Option<usize>,
}

impl Node {
    fn new(output: f64) -> Self {
        Node {
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
        (self.output - other.output).abs() < f64::EPSILON
            && self.split_feature == other.split_feature
            && match (self.split_value, other.split_value) {
                (Some(a), Some(b)) => (a - b).abs() < f64::EPSILON,
                (None, None) => true,
                _ => false,
            }
            && match (self.split_score, other.split_score) {
                (Some(a), Some(b)) => (a - b).abs() < f64::EPSILON,
                (None, None) => true,
                _ => false,
            }
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for BaseTreeRegressor<TX, TY, X, Y>
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
    start_idx: usize, // start index for the elements in `sorted_by_feature` of SplitWorkspace
    end_idx: usize,   // end index (exclusive) in the same vector(s)
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
        start_idx: usize,
        end_idx: usize,
        x: &'a X,
        y: &'a Y,
        level: u16,
    ) -> Self {
        NodeVisitor {
            x,
            y,
            node: node_id,
            start_idx,
            end_idx,
            true_child_output: 0f64,
            false_child_output: 0f64,
            level,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
        }
    }
}

/// Weighted count of sample `i`. The weight is 1.0 if no weights are given.
fn mass_of(i: usize, samples: &[usize], sample_weights: Option<&[f64]>) -> f64 {
    match sample_weights {
        Some(weights) => samples[i] as f64 * weights[i],
        None => samples[i] as f64,
    }
}

// Struct representing an element that belongs logically to a Node, as stored in SplitWorkspace
#[derive(Copy, Clone, Default)]
struct NodeElement {
    // the row index in the dataset
    pub row_idx: u32,
    // the number of times this row is present, should always be > 0
    pub count: u32,
    // total mass of this element, equals count * mass of individual element.
    // equals count when no sample weights were used
    pub mass: f64,
}

impl NodeElement {
    // return the row_idx as usize
    #[inline(always)]
    fn row(&self) -> usize {
        self.row_idx as usize
    }
}

// Checks whether the example indicated by node_element is a "true child" for this node
fn is_true_sample<TX, X>(node_element: &NodeElement, x: &X, node: &Node) -> bool
where
    TX: Number + PartialOrd,
    X: Array2<TX>,
{
    x.get((node_element.row(), node.split_feature))
        .to_f64()
        .unwrap()
        <= node.split_value.unwrap_or(f64::NAN)
}

// slice: slice that will be partitioned
// scratch: temp buffer
// is_true: is_true[idx] checks whether element with row idx equal to idx belongs to the true branch
// returns: index of first element of false branch
fn stable_partition(
    slice: &mut [NodeElement],
    scratch: &mut [NodeElement],
    is_true: &[bool],
) -> usize {
    // Note: this is intentionally written without an if/else branch in the main loop
    let n = slice.len();
    let scratch = &mut scratch[..n];
    let (mut w, mut f) = (0usize, 0usize);
    for i in 0..n {
        let e = slice[i];
        let t = is_true[e.row_idx as usize];
        slice[w] = e; // w <= i, so this never clobbers an unread element
        scratch[f] = e;
        // advance only one of the pointers
        w += t as usize;
        f += (!t) as usize;
    }
    slice[w..].copy_from_slice(&scratch[..f]);
    w
}

struct SplitWorkspace {
    in_true_branch: Vec<bool>, // indexed by row index in X
    partition_buffer: Vec<NodeElement>,
    sorted_by_feature: Vec<Vec<NodeElement>>,
}

impl SplitWorkspace {
    fn new(
        in_true_branch: Vec<bool>,
        partition_buffer: Vec<NodeElement>,
        sorted_by_feature: Vec<Vec<NodeElement>>,
    ) -> Self {
        Self {
            in_true_branch,
            partition_buffer,
            sorted_by_feature,
        }
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    BaseTreeRegressor<TX, TY, X, Y>
{
    pub(crate) fn fit_inner(
        x: &X,
        y: &Y,
        sample_weights: Option<&[f64]>,
        parameters: BaseTreeRegressorParameters,
    ) -> Result<BaseTreeRegressor<TX, TY, X, Y>, Failed> {
        let (x_nrows, num_attributes) = x.shape();
        if x_nrows != y.shape() {
            return Err(Failed::fit("Size of x should equal size of y"));
        }
        if x_nrows == 0 || num_attributes == 0 {
            return Err(Failed::because(
                FailedError::ParametersError,
                "Training data must contain at least one sample and one feature.",
            ));
        }

        // Compute the order of each attribute once
        let mut order: Vec<Vec<usize>> = Vec::new();

        for i in 0..num_attributes {
            let mut col_i: Vec<TX> = x.get_col(i).iterator(0).copied().collect();
            order.push(col_i.argsort_mut());
        }

        let samples = vec![1; x_nrows];
        BaseTreeRegressor::fit_weak_learner(
            x,
            y,
            sample_weights,
            samples,
            num_attributes,
            &order,
            parameters,
        )
    }

    pub(crate) fn fit_weak_learner(
        x: &X,
        y: &Y,
        sample_weights: Option<&[f64]>,
        samples: Vec<usize>,
        mtry: usize,
        order: &[Vec<usize>],
        parameters: BaseTreeRegressorParameters,
    ) -> Result<BaseTreeRegressor<TX, TY, X, Y>, Failed> {
        let n_rows = y.shape();

        let mut nodes: Vec<Node> = Vec::new();
        let mut rng = get_rng_impl(parameters.seed);

        let mut sum = 0f64;
        let mut mass = 0f64;

        for i in 0..n_rows {
            let mass_i = mass_of(i, &samples, sample_weights);
            mass += mass_i;
            sum += mass_i * y.get(i).to_f64().unwrap();
        }

        let root = Node::new(sum / mass);
        nodes.push(root);

        let sorted_by_feature: Vec<Vec<NodeElement>> = order
            .iter()
            .map(|col_order| {
                col_order
                    .iter()
                    .filter(|&&i| samples[i] > 0)
                    .map(|&i| NodeElement {
                        row_idx: i as u32,
                        count: samples[i] as u32,
                        mass: mass_of(i, &samples, sample_weights),
                    })
                    .collect()
            })
            .collect();
        let end_idx = sorted_by_feature[0].len();

        let mut workspace = SplitWorkspace::new(
            vec![false; x.shape().0],
            vec![NodeElement::default(); end_idx],
            sorted_by_feature,
        );

        let mut base_tree = BaseTreeRegressor {
            nodes,
            parameters: Some(parameters),
            depth: 0u16,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        };

        let mut visitor = NodeVisitor::<TX, TY, X, Y>::new(0, 0, end_idx, x, y, 1);

        let mut visitor_queue: VecDeque<NodeVisitor<'_, TX, TY, X, Y>> = VecDeque::new();

        if base_tree.find_best_cutoff(&mut visitor, mtry, mass, &mut rng, &workspace) {
            visitor_queue.push_back(visitor);
        }

        let max_depth = base_tree.parameters().max_depth.unwrap_or(u16::MAX);
        while let Some(node) = visitor_queue.pop_front() {
            if node.level < max_depth {
                base_tree.split(node, mtry, &mut visitor_queue, &mut rng, &mut workspace);
            }
        }

        Ok(base_tree)
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
        let mut node_id = 0;
        loop {
            let node = &self.nodes()[node_id];
            let Some(true_child) = node.true_child else {
                return TY::from_f64(node.output).unwrap();
            };
            let false_child = node.false_child.unwrap();
            node_id = if x.get((row, node.split_feature)).to_f64().unwrap()
                <= node.split_value.unwrap_or(f64::NAN)
            {
                true_child
            } else {
                false_child
            };
        }
    }

    fn find_best_cutoff(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, TY, X, Y>,
        mtry: usize,
        mass: f64,
        rng: &mut impl rand::Rng,
        workspace: &SplitWorkspace,
    ) -> bool {
        let (_, n_attr) = visitor.x.shape();

        let n: usize = workspace.sorted_by_feature[0][visitor.start_idx..visitor.end_idx]
            .iter()
            .map(|elem| elem.count as usize)
            .sum();

        if n < self.parameters().min_samples_split {
            return false;
        }

        let sum = self.nodes()[visitor.node].output * mass;

        // TODO later: get rid of this allocation in every iteration
        let mut variables = (0..n_attr).collect::<Vec<_>>();

        if mtry < n_attr {
            variables.shuffle(rng);
        }

        let parent_gain =
            mass * self.nodes()[visitor.node].output * self.nodes()[visitor.node].output;

        let splitter = self.parameters().splitter.clone();

        for variable in variables.iter().take(mtry) {
            match splitter {
                Splitter::Random => {
                    self.find_random_split(
                        visitor,
                        n,
                        mass,
                        sum,
                        parent_gain,
                        *variable,
                        rng,
                        workspace,
                    );
                }
                Splitter::Best => {
                    self.find_best_split(visitor, n, mass, sum, parent_gain, *variable, workspace);
                }
            }
        }

        self.nodes()[visitor.node].split_score.is_some()
    }

    fn find_random_split(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, TY, X, Y>,
        n: usize,
        mass: f64,
        sum: f64,
        parent_gain: f64,
        j: usize,
        rng: &mut impl rand::Rng,
        workspace: &SplitWorkspace,
    ) {
        if visitor.start_idx == visitor.end_idx {
            return;
        }
        let first_elem = workspace.sorted_by_feature[j][visitor.start_idx];
        let min_val = visitor.x.get((first_elem.row(), j));
        let last_elem = workspace.sorted_by_feature[j][visitor.end_idx - 1];
        let max_val = visitor.x.get((last_elem.row(), j));

        if min_val >= max_val {
            return;
        }

        let split_value = rng.random_range(min_val.to_f64().unwrap()..max_val.to_f64().unwrap());

        let mut true_sum = 0f64;
        let mut true_mass = 0f64;
        let mut true_count = 0;
        for elem in &workspace.sorted_by_feature[j][visitor.start_idx..visitor.end_idx] {
            if visitor.x.get((elem.row(), j)).to_f64().unwrap() <= split_value {
                true_sum += elem.mass * visitor.y.get(elem.row()).to_f64().unwrap();
                true_count += elem.count;
                true_mass += elem.mass;
            }
        }

        let false_count = n - (true_count as usize);

        if (true_count as usize) < self.parameters().min_samples_leaf
            || false_count < self.parameters().min_samples_leaf
        {
            return;
        }

        let true_mean = if true_mass > 0f64 {
            true_sum / true_mass
        } else {
            0.0
        };
        let false_mass = mass - true_mass;
        let false_mean = if false_mass > 0f64 {
            (sum - true_sum) / false_mass
        } else {
            0.0
        };
        let gain = (true_mass * true_mean * true_mean + false_mass * false_mean * false_mean)
            - parent_gain;

        if self.nodes[visitor.node].split_score.is_none()
            || gain > self.nodes[visitor.node].split_score.unwrap()
        {
            self.nodes[visitor.node].split_feature = j;
            self.nodes[visitor.node].split_value = Some(split_value);
            self.nodes[visitor.node].split_score = Some(gain);
            visitor.true_child_output = true_mean;
            visitor.false_child_output = false_mean;
        }
    }

    fn find_best_split(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, TY, X, Y>,
        n: usize,
        mass: f64,
        sum: f64,
        parent_gain: f64,
        j: usize,
        workspace: &SplitWorkspace,
    ) {
        let mut true_sum = 0f64;
        let mut true_count = 0;
        let mut true_mass = 0f64;
        let mut prevx = Option::None;

        for elem in &workspace.sorted_by_feature[j][visitor.start_idx..visitor.end_idx] {
            let x_ij = *visitor.x.get((elem.row(), j));

            if prevx.is_none() || x_ij == prevx.unwrap() {
                prevx = Some(x_ij);
                true_count += elem.count;
                true_mass += elem.mass;
                true_sum += elem.mass * visitor.y.get(elem.row()).to_f64().unwrap();
                continue;
            }

            let false_count = n - (true_count as usize);

            if (true_count as usize) < self.parameters().min_samples_leaf
                || false_count < self.parameters().min_samples_leaf
            {
                prevx = Some(x_ij);
                true_count += elem.count;
                true_mass += elem.mass;
                true_sum += elem.mass * visitor.y.get(elem.row()).to_f64().unwrap();
                continue;
            }

            let true_mean = if true_mass > 0.0 {
                true_sum / true_mass
            } else {
                0.0
            };
            let false_mass = mass - true_mass;
            let false_mean = if false_mass > 0.0 {
                (sum - true_sum) / false_mass
            } else {
                0.0
            };

            let gain = (true_mass * true_mean * true_mean + false_mass * false_mean * false_mean)
                - parent_gain;

            if self.nodes()[visitor.node].split_score.is_none()
                || gain > self.nodes()[visitor.node].split_score.unwrap()
            {
                self.nodes[visitor.node].split_feature = j;
                self.nodes[visitor.node].split_value = Option::Some(
                    (x_ij.to_f64().unwrap() + prevx.unwrap().to_f64().unwrap()) / 2f64,
                );
                self.nodes[visitor.node].split_score = Option::Some(gain);

                visitor.true_child_output = true_mean;
                visitor.false_child_output = false_mean;
            }

            prevx = Some(x_ij);
            true_sum += elem.mass * visitor.y.get(elem.row()).to_f64().unwrap();
            true_count += elem.count;
            true_mass += elem.mass;
        }
    }

    fn split<'a>(
        &mut self,
        visitor: NodeVisitor<'a, TX, TY, X, Y>,
        mtry: usize,
        visitor_queue: &mut VecDeque<NodeVisitor<'a, TX, TY, X, Y>>,
        rng: &mut impl rand::Rng,
        workspace: &mut SplitWorkspace,
    ) -> bool {
        let this_node = &self.nodes()[visitor.node];

        let mut tc = 0usize;
        let mut true_mass = 0f64;
        let mut fc = 0usize;
        let mut false_mass = 0f64;
        // for each row_index, does it belong in the true branch or not?
        let in_true_branch = &mut workspace.in_true_branch;
        let mut n_true = 0usize;
        for e in &workspace.sorted_by_feature[0][visitor.start_idx..visitor.end_idx] {
            let t = is_true_sample(e, visitor.x, this_node);
            in_true_branch[e.row()] = t;
            n_true += t as usize;
            // Fill in tc, etc while we are at it
            if t {
                tc += e.count as usize;
                true_mass += e.mass;
            } else {
                fc += e.count as usize;
                false_mass += e.mass;
            }
        }

        // Stop early if it is clear that there will be too few examples in the leaf
        if tc < self.parameters().min_samples_leaf || fc < self.parameters().min_samples_leaf {
            self.nodes[visitor.node].split_feature = 0;
            self.nodes[visitor.node].split_value = Option::None;
            self.nodes[visitor.node].split_score = Option::None;

            return false;
        }

        // Add the child nodes to the tree
        let split_idx = visitor.start_idx + n_true;

        let true_child_idx = self.nodes().len();

        self.nodes.push(Node::new(visitor.true_child_output));
        let false_child_idx = self.nodes().len();
        self.nodes.push(Node::new(visitor.false_child_output));

        self.nodes[visitor.node].true_child = Some(true_child_idx);
        self.nodes[visitor.node].false_child = Some(false_child_idx);

        self.depth = u16::max(self.depth, visitor.level + 1);

        // If the child nodes can not be split any further, there is no point in partitioning the ranges
        let max_depth = self.parameters().max_depth.unwrap_or(u16::MAX);
        let child_level = visitor.level + 1;
        let min_split = self.parameters().min_samples_split;
        let true_can_split = child_level < max_depth && tc >= min_split;
        let false_can_split = child_level < max_depth && fc >= min_split;
        if !true_can_split && !false_can_split {
            return true; // both children are leaves: no partition
        }

        for j in 0..visitor.x.shape().1 {
            stable_partition(
                &mut workspace.sorted_by_feature[j][visitor.start_idx..visitor.end_idx],
                &mut workspace.partition_buffer,
                in_true_branch,
            );
        }

        let mut true_visitor = NodeVisitor::<TX, TY, X, Y>::new(
            true_child_idx,
            visitor.start_idx,
            split_idx,
            visitor.x,
            visitor.y,
            visitor.level + 1,
        );

        if self.find_best_cutoff(&mut true_visitor, mtry, true_mass, rng, workspace) {
            visitor_queue.push_back(true_visitor);
        }

        let mut false_visitor = NodeVisitor::<TX, TY, X, Y>::new(
            false_child_idx,
            split_idx,
            visitor.end_idx,
            visitor.x,
            visitor.y,
            visitor.level + 1,
        );

        if self.find_best_cutoff(&mut false_visitor, mtry, false_mass, rng, workspace) {
            visitor_queue.push_back(false_visitor);
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::metrics::mean_absolute_error;

    #[test]
    fn test_fit_on_empty_data_returns_error() {
        // 2 rows x 2 features — values are arbitrary; only the empty-row case is under test
        let full = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]).unwrap();
        let empty = full.take(&[] as &[usize], 0);
        assert_eq!(empty.shape(), (0, 2));

        let y: Vec<f64> = vec![];
        let result = BaseTreeRegressor::fit_inner(
            &empty,
            &y,
            None,
            BaseTreeRegressorParameters {
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                seed: None,
                splitter: Splitter::Best,
            },
        );
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error(), FailedError::ParametersError);
    }

    #[test]
    fn test_fit_on_zero_features_returns_error() {
        // 2 rows x 2 features — values are arbitrary; only the zero-feature case is under test
        let full = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]).unwrap();
        let no_features = full.take(&[] as &[usize], 1);
        assert_eq!(no_features.shape(), (2, 0));

        let y = vec![1.0_f64, 2.0];
        let result = BaseTreeRegressor::fit_inner(
            &no_features,
            &y,
            None,
            BaseTreeRegressorParameters {
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                seed: None,
                splitter: Splitter::Best,
            },
        );
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().error(), FailedError::ParametersError);
    }

    #[test]
    fn root_prediction_is_weighted_mean() {
        // Create a tree with no splits. Assert that the prediction is the weighted mean of the targets.

        // X-values are arbitrary. 3 examples
        let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])
            .unwrap();
        let y = vec![1.0, 2.0, 3.0];
        let sample_weights = vec![5.0, 6.0, 7.0];

        let result = BaseTreeRegressor::fit_inner(
            &x,
            &y,
            Some(&sample_weights),
            BaseTreeRegressorParameters {
                max_depth: Some(0),
                min_samples_leaf: 1,
                min_samples_split: 2,
                seed: None,
                splitter: Splitter::Best,
            },
        );
        assert!(result.is_ok());
        let tree = result.unwrap();
        let expected = y
            .iter()
            .zip(sample_weights.iter())
            .map(|(yi, wi)| yi * wi)
            .sum::<f64>()
            / sample_weights.iter().sum::<f64>();

        assert!((tree.predict_for_row(&x, 0) - expected).abs() < 1e-9);
    }

    #[test]
    fn uniform_weights_match_unweighted() {
        // Test that using no weights is equivalent to using uniform weights
        let x_rand: DenseMatrix<f64> = DenseMatrix::<f64>::rand(17, 5);
        let y_rand: Vec<f64> = (0..17).collect::<Vec<_>>().map(|y| *y as f64);
        let parameters = BaseTreeRegressorParameters {
            max_depth: Some(5),
            min_samples_leaf: 1,
            min_samples_split: 1,
            seed: Some(42),
            splitter: Splitter::Best,
        };

        let tree_no_weights =
            BaseTreeRegressor::fit_inner(&x_rand, &y_rand, None, parameters.clone())
                .expect("Fit should work");
        let uniform_weights = vec![1.0f64; 17];
        let tree_with_weights =
            BaseTreeRegressor::fit_inner(&x_rand, &y_rand, Some(&uniform_weights), parameters)
                .expect("Fit should work");

        let y_pred_no_weights = tree_no_weights
            .predict(&x_rand)
            .expect("Predict should work");
        let y_pred_with_weights = tree_with_weights
            .predict(&x_rand)
            .expect("Predict should work");
        assert!(mean_absolute_error(&y_pred_no_weights, &y_pred_with_weights) < 1e-9);
    }

    #[test]
    fn integer_weights_equivalent_to_repeating_sample() {
        // Test that setting weight to "2" (or "3") is equivalent to having the same sample twice (or trice)
        let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])
            .unwrap();
        let y = vec![4.0, 5.0, 6.0];
        let sample_weights = vec![1.0, 2.0, 3.0];

        let x_repeated = DenseMatrix::from_2d_vec(&vec![
            vec![1.0_f64, 2.0],
            vec![3.0, 4.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![5.0, 6.0],
            vec![5.0, 6.0],
        ])
        .unwrap();
        let y_repeated = vec![4.0, 5.0, 5.0, 6.0, 6.0, 6.0];

        let weighted_parameters = BaseTreeRegressorParameters {
            max_depth: Some(1), // tree should not be able to fully separate all the examples
            min_samples_leaf: 1,
            min_samples_split: 1,
            seed: Some(42),
            splitter: Splitter::Best,
        };

        let repeated_parameters = BaseTreeRegressorParameters {
            max_depth: Some(1),
            min_samples_leaf: 1,
            min_samples_split: 1,
            seed: Some(42),
            splitter: Splitter::Best,
        };

        let tree_weighted =
            BaseTreeRegressor::fit_inner(&x, &y, Some(&sample_weights), weighted_parameters)
                .expect("Fit should work");
        let tree_repeated =
            BaseTreeRegressor::fit_inner(&x_repeated, &y_repeated, None, repeated_parameters)
                .expect("Fit should work");
        // Predict on the same data
        let y_pred_weighted = tree_weighted.predict(&x).expect("Predict should work");
        let y_pred_repeated = tree_repeated.predict(&x).expect("Predict should work");

        assert!(mean_absolute_error(&y_pred_weighted, &y_pred_repeated) < 1e-9);
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
        let y = vec![1.0f64, 2.0, 6.0, 7.0, 11., 12.];

        let parameters = BaseTreeRegressorParameters {
            max_depth: Some(3),
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: None,
            splitter: Splitter::Best,
        };

        let tree = BaseTreeRegressor::fit_inner(&x, &y, None, parameters).expect("Fit should work");
        let y_expected = vec![1.0, 2.0, 6.5, 6.5, 11.50, 11.50];
        let y_hat = tree.predict(&x).expect("Predict should work");
        assert_eq!(tree.nodes().len(), 7);
        assert_eq!(tree.depth, 3);
        assert!(mean_absolute_error(&y_expected, &y_hat) < 1e-9);
    }

    #[test]
    fn full_depth_with_weights() {
        let x = DenseMatrix::from_2d_vec(&vec![
            vec![1.0_f64],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ])
        .unwrap();
        let y = vec![1.0f64, 2.0, 6.0, 7.0, 11., 12.];
        let parameters = BaseTreeRegressorParameters {
            max_depth: Some(3),
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: None,
            splitter: Splitter::Best,
        };

        let sample_weights = [1.0f64, 1.0, 1.0, 1.0, 10.0, 10.0];
        let tree = BaseTreeRegressor::fit_inner(&x, &y, Some(&sample_weights), parameters)
            .expect("Fit should work");
        let x_test: DenseMatrix<f64> =
            DenseMatrix::from_iterator((1..=12).map(|i| i as f64), 12, 1, 0);
        let y_expected = vec![
            1.50, 1.50, 6.50, 6.50, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
        ];
        let y_hat = tree.predict(&x_test).expect("Predict should work");
        assert_eq!(tree.nodes().len(), 7);
        assert_eq!(tree.depth, 3);
        assert!(mean_absolute_error(&y_expected, &y_hat) < 1e-9);
    }

    #[test]
    fn min_samples_split_boundary() {
        // A node that holds exactly `min_samples_split` samples must still split.
        let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![1.0f64, 2.0, 6.0];

        let parameters = BaseTreeRegressorParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 3,
            seed: None,
            splitter: Splitter::Best,
        };

        let tree = BaseTreeRegressor::fit_inner(&x, &y, None, parameters).expect("Fit should work");
        assert_eq!(tree.nodes().len(), 3);
        assert_eq!(tree.depth, 2);

        // A node with fewer than `min_samples_split` samples must stay a leaf.
        let parameters = BaseTreeRegressorParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 4,
            seed: None,
            splitter: Splitter::Best,
        };

        let tree = BaseTreeRegressor::fit_inner(&x, &y, None, parameters).expect("Fit should work");
        assert_eq!(tree.nodes().len(), 1);
        assert_eq!(tree.depth, 0);
    }

    #[test]
    fn zero_weight_on_true_side_gives_finite_predictions() {
        // The first candidate split has only zero-weight samples on its true side,
        // so true_mass is 0 in find_best_split. This must not produce NaN.
        let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![1.0f64, 2.0, 3.0];
        let sample_weights = [0.0f64, 1.0, 1.0];

        let parameters = BaseTreeRegressorParameters {
            max_depth: Some(2),
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: None,
            splitter: Splitter::Best,
        };

        let tree = BaseTreeRegressor::fit_inner(&x, &y, Some(&sample_weights), parameters)
            .expect("Fit should work");

        assert!(tree.nodes().iter().all(|node| node.output.is_finite()));
        assert!(
            tree.nodes()
                .iter()
                .all(|node| node.split_score.is_none_or(f64::is_finite))
        );

        let y_hat = tree.predict(&x).expect("Predict should work");
        assert!(y_hat.iter().all(|v| v.is_finite()));

        // The best split separates x=3 from the other rows.
        let y_expected = vec![2.0, 2.0, 3.0];
        assert!(mean_absolute_error(&y_expected, &y_hat) < 1e-9);
    }

    #[test]
    fn zero_weight_on_false_side_gives_finite_predictions() {
        // The two first rows share x=1, so the only candidate split is between x=1 and x=2.
        // The false side of that split holds only a zero-weight sample,
        // so false_mass is 0 in find_best_split. This must not produce NaN.
        let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64], vec![1.0], vec![2.0]]).unwrap();
        let y = vec![1.0f64, 2.0, 3.0];
        let sample_weights = [1.0f64, 1.0, 0.0];

        let parameters = BaseTreeRegressorParameters {
            max_depth: Some(2),
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: None,
            splitter: Splitter::Best,
        };

        let tree = BaseTreeRegressor::fit_inner(&x, &y, Some(&sample_weights), parameters)
            .expect("Fit should work");

        assert!(tree.nodes().iter().all(|node| node.output.is_finite()));
        assert!(
            tree.nodes()
                .iter()
                .all(|node| node.split_score.is_none_or(f64::is_finite))
        );

        let y_hat = tree.predict(&x).expect("Predict should work");
        assert!(y_hat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn weights_on_tied_feature_values() {
        // Rows with the same feature value but different weights. The split must keep each
        // tie group together, and each leaf must give the weighted mean of its group.
        let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64], vec![1.0], vec![2.0], vec![2.0]])
            .unwrap();
        let y = vec![0.0f64, 10.0, 20.0, 30.0];
        let sample_weights = [3.0f64, 1.0, 1.0, 3.0];

        let parameters = BaseTreeRegressorParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: None,
            splitter: Splitter::Best,
        };

        let tree = BaseTreeRegressor::fit_inner(&x, &y, Some(&sample_weights), parameters.clone())
            .expect("Fit should work");

        assert_eq!(tree.nodes().len(), 3);
        assert_eq!(tree.depth, 2);
        assert!((tree.nodes()[0].split_value.unwrap() - 1.5).abs() < 1e-9); // Split should be at 1.5

        let y_hat = tree.predict(&x).expect("Predict should work");
        let y_expected = vec![2.5, 2.5, 27.5, 27.5]; // Expected values are weighted means
        assert!(mean_absolute_error(&y_expected, &y_hat) < 1e-9);

        // Integer weights must give the same tree as repeated rows.
        let x_repeated = DenseMatrix::from_2d_vec(&vec![
            vec![1.0_f64],
            vec![1.0],
            vec![1.0],
            vec![1.0],
            vec![2.0],
            vec![2.0],
            vec![2.0],
            vec![2.0],
        ])
        .unwrap();
        let y_repeated = vec![0.0f64, 0.0, 0.0, 10.0, 20.0, 30.0, 30.0, 30.0];
        let tree_repeated =
            BaseTreeRegressor::fit_inner(&x_repeated, &y_repeated, None, parameters)
                .expect("Fit should work");
        let y_hat_repeated = tree_repeated.predict(&x).expect("Predict should work");
        assert!(mean_absolute_error(&y_hat, &y_hat_repeated) < 1e-9);
    }
}
