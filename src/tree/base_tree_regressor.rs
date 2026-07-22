use std::collections::LinkedList;
use std::default::Default;
use std::fmt::Debug;
use std::marker::PhantomData;

use rand::seq::SliceRandom;
use rand::RngExt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::Failed;
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
    /// Getter for parameters used in the model
    ///
    /// # Returns
    /// `Some` with the parameters used to configure the model, or `None` if unavailable.
    fn parameters(&self) -> Option<&BaseTreeRegressorParameters> {
        self.parameters.as_ref()
    }
    /// Get estimate of intercept, return value
    fn depth(&self) -> u16 {
        self.depth
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
    BaseTreeRegressor<TX, TY, X, Y>
{
    /// Build a decision base_tree regressor from the training data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: BaseTreeRegressorParameters,
    ) -> Result<BaseTreeRegressor<TX, TY, X, Y>, Failed> {
        let (x_nrows, num_attributes) = x.shape();
        if x_nrows != y.shape() {
            return Err(Failed::fit("Size of x should equal size of y"));
        }

        let samples = vec![1; x_nrows];
        BaseTreeRegressor::fit_weak_learner(x, y, samples, num_attributes, parameters)
    }

    pub(crate) fn fit_weak_learner(
        x: &X,
        y: &Y,
        samples: Vec<usize>,
        mtry: usize,
        parameters: BaseTreeRegressorParameters,
    ) -> Result<BaseTreeRegressor<TX, TY, X, Y>, Failed> {
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

        let root = Node::new(sum / (n as f64));
        nodes.push(root);
        let mut order: Vec<Vec<usize>> = Vec::new();

        for i in 0..num_attributes {
            let mut col_i: Vec<TX> = x.get_col(i).iterator(0).copied().collect();
            order.push(col_i.argsort_mut());
        }

        let mut base_tree = BaseTreeRegressor {
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

        if base_tree.find_best_cutoff(&mut visitor, mtry, &mut rng) {
            visitor_queue.push_back(visitor);
        }

        while base_tree.depth()
            < base_tree
                .parameters()
                .expect("parameters not set — model not fitted")
                .max_depth
                .unwrap_or(u16::MAX)
        {
            match visitor_queue.pop_front() {
                Some(node) => base_tree.split(node, mtry, &mut visitor_queue, &mut rng),
                None => break,
            };
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
        let mut result = 0f64;
        let mut queue: LinkedList<usize> = LinkedList::new();

        queue.push_back(0);

        while !queue.is_empty() {
            match queue.pop_front() {
                Some(node_id) => {
                    let node = &self.nodes()[node_id];
                    if node.true_child.is_none() && node.false_child.is_none() {
                        result = node.output;
                    } else if x.get((row, node.split_feature)).to_f64().unwrap()
                        <= node.split_value.unwrap_or(f64::NAN)
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
        rng: &mut impl rand::Rng,
    ) -> bool {
        let (_, n_attr) = visitor.x.shape();

        let n: usize = visitor.samples.iter().sum();

        let parameters = self
            .parameters()
            .expect("parameters not set — model not fitted");

        if n < parameters.min_samples_split {
            return false;
        }

        let sum = self.nodes()[visitor.node].output * n as f64;

        let mut variables = (0..n_attr).collect::<Vec<_>>();

        if mtry < n_attr {
            variables.shuffle(rng);
        }

        let parent_gain =
            n as f64 * self.nodes()[visitor.node].output * self.nodes()[visitor.node].output;

        let splitter = parameters.splitter.clone();

        for variable in variables.iter().take(mtry) {
            match splitter {
                Splitter::Random => {
                    self.find_random_split(visitor, n, sum, parent_gain, *variable, rng);
                }
                Splitter::Best => {
                    self.find_best_split(visitor, n, sum, parent_gain, *variable);
                }
            }
        }

        self.nodes()[visitor.node].split_score.is_some()
    }

    fn find_random_split(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, TY, X, Y>,
        n: usize,
        sum: f64,
        parent_gain: f64,
        j: usize,
        rng: &mut impl rand::Rng,
    ) {
        let (min_val, max_val) = {
            let mut min_opt = None;
            let mut max_opt = None;
            for &i in &visitor.order[j] {
                if visitor.samples[i] > 0 {
                    min_opt = Some(*visitor.x.get((i, j)));
                    break;
                }
            }
            for &i in visitor.order[j].iter().rev() {
                if visitor.samples[i] > 0 {
                    max_opt = Some(*visitor.x.get((i, j)));
                    break;
                }
            }
            if min_opt.is_none() {
                return;
            }
            (min_opt.unwrap(), max_opt.unwrap())
        };

        if min_val >= max_val {
            return;
        }

        let split_value = rng.random_range(min_val.to_f64().unwrap()..max_val.to_f64().unwrap());

        let mut true_sum = 0f64;
        let mut true_count = 0;
        for &i in &visitor.order[j] {
            if visitor.samples[i] > 0 {
                if visitor.x.get((i, j)).to_f64().unwrap() <= split_value {
                    true_sum += visitor.samples[i] as f64 * visitor.y.get(i).to_f64().unwrap();
                    true_count += visitor.samples[i];
                } else {
                    break;
                }
            }
        }

        let false_count = n - true_count;

        let parameters = self
            .parameters()
            .expect("parameters not set — model not fitted");

        if true_count < parameters.min_samples_leaf || false_count < parameters.min_samples_leaf {
            return;
        }

        let true_mean = if true_count > 0 {
            true_sum / true_count as f64
        } else {
            0.0
        };
        let false_mean = if false_count > 0 {
            (sum - true_sum) / false_count as f64
        } else {
            0.0
        };
        let gain = (true_count as f64 * true_mean * true_mean
            + false_count as f64 * false_mean * false_mean)
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

                let parameters = self
                    .parameters()
                    .expect("parameters not set — model not fitted");

                if true_count < parameters.min_samples_leaf
                    || false_count < parameters.min_samples_leaf
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
        rng: &mut impl rand::Rng,
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
                    <= self.nodes()[visitor.node].split_value.unwrap_or(f64::NAN)
                {
                    *true_sample = visitor.samples[i];
                    tc += *true_sample;
                    visitor.samples[i] = 0;
                } else {
                    fc += visitor.samples[i];
                }
            }
        }

        let parameters = self
            .parameters()
            .expect("parameters not set — model not fitted");

        if tc < parameters.min_samples_leaf || fc < parameters.min_samples_leaf {
            self.nodes[visitor.node].split_feature = 0;
            self.nodes[visitor.node].split_value = Option::None;
            self.nodes[visitor.node].split_score = Option::None;

            return false;
        }

        let true_child_idx = self.nodes().len();

        self.nodes.push(Node::new(visitor.true_child_output));
        let false_child_idx = self.nodes().len();
        self.nodes.push(Node::new(visitor.false_child_output));

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
