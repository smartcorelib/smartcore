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
#[derive(Debug, Clone, Copy, Default)]
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
        if self.depth != other.depth || self.nodes.len() != other.nodes.len() {
            false
        } else {
            self.nodes
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

/// Node elements store row indices and counts as `u32`.
fn check_row_count(n_rows: usize) -> Result<(), Failed> {
    if u32::try_from(n_rows).is_err() {
        return Err(Failed::fit(
            "Number of rows in x must not be larger than u32::MAX",
        ));
    }
    Ok(())
}

// Trait representing an element that belongs logically to a Node. Implementations of it are stored in SplitWorkspace
trait NodeElement: Copy + Default {
    fn new(row: usize, count: usize, mass: f64) -> Self;
    // The row index in the dataset
    fn row(&self) -> usize;
    // The number of times this row was sampled, should always be > 0
    fn count(&self) -> usize;
    // The total mass of this element: count * sample_weight of row or count if no sample weights are used
    fn mass(&self) -> f64;
}

#[derive(Copy, Clone, Default)]
struct WeightedElement {
    row_idx: u32,
    count: u32,
    mass: f64,
}

impl NodeElement for WeightedElement {
    fn new(row_idx: usize, count: usize, mass: f64) -> Self {
        Self {
            row_idx: row_idx as u32,
            count: count as u32,
            mass,
        }
    }

    #[inline(always)]
    fn row(&self) -> usize {
        self.row_idx as usize
    }

    #[inline(always)]
    fn count(&self) -> usize {
        self.count as usize
    }

    #[inline(always)]
    fn mass(&self) -> f64 {
        self.mass
    }
}

// implement CountedElement as a single u64, so it can be retrieved in a single read
#[derive(Copy, Clone, Default)]
struct CountedElement(u64); // low 32 bits: row, high 32 bits: count

impl NodeElement for CountedElement {
    fn new(row_idx: usize, count: usize, mass: f64) -> Self {
        // mass() is derived from count, so sample weights must not be used
        debug_assert_eq!(mass, count as f64);
        Self(row_idx as u64 | ((count as u64) << 32))
    }
    #[inline(always)]
    fn row(&self) -> usize {
        self.0 as u32 as usize
    }
    #[inline(always)]
    fn count(&self) -> usize {
        (self.0 >> 32) as usize
    }
    #[inline(always)]
    fn mass(&self) -> f64 {
        (self.0 >> 32) as f64
    }
}

#[derive(Copy, Clone, Default)]
struct UnitElement {
    row_idx: u32,
}

impl NodeElement for UnitElement {
    fn new(row_idx: usize, count: usize, mass: f64) -> Self {
        // count() and mass() are always 1, so sample weights and counts > 1 must not be used.
        // A count of 0 is allowed: such elements are overwritten during setup.
        debug_assert!(count <= 1);
        debug_assert_eq!(mass, count as f64);
        Self {
            row_idx: row_idx as u32,
        }
    }

    #[inline(always)]
    fn row(&self) -> usize {
        self.row_idx as usize
    }

    #[inline(always)]
    fn count(&self) -> usize {
        1
    }

    #[inline(always)]
    fn mass(&self) -> f64 {
        1.0
    }
}

// Checks whether the example indicated by node_element is a "true child" for this node
fn is_true_sample<TX, X, E>(node_element: &E, x: &X, node: &Node) -> bool
where
    TX: Number + PartialOrd,
    X: Array2<TX>,
    E: NodeElement,
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
#[inline(never)] // inline(never) is faster as seen during profiling
fn stable_partition<E>(slice: &mut [E], scratch: &mut [E], is_true: &[bool]) -> usize
where
    E: NodeElement,
{
    // Note: this is intentionally written without an explicit if/else branch in the main loop
    let n = slice.len();
    let scratch = &mut scratch[..n];
    let (mut w, mut f) = (0usize, 0usize);
    for i in 0..n {
        let e = slice[i];
        let t = is_true[e.row()];
        slice[w] = e; // w <= i, so this never clobbers an unread element
        scratch[f] = e;
        // advance only one of the pointers
        w += t as usize;
        f += (!t) as usize;
    }
    slice[w..].copy_from_slice(&scratch[..f]);
    w
}

struct SplitWorkspace<E> {
    in_true_branch: Vec<bool>, // indexed by row index in X
    partition_buffer: Vec<E>,
    sorted_by_feature: Vec<Vec<E>>,
    variables: Vec<usize>, // feature indices which will be shuffled when mtry < n_features
}

impl<E> SplitWorkspace<E> {
    fn new(
        in_true_branch: Vec<bool>,
        partition_buffer: Vec<E>,
        sorted_by_feature: Vec<Vec<E>>,
        n_features: usize,
    ) -> Self {
        Self {
            in_true_branch,
            partition_buffer,
            sorted_by_feature,
            variables: (0..n_features).collect::<Vec<_>>(),
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
        check_row_count(y.shape())?;
        match sample_weights {
            Some(_) => Self::grow::<WeightedElement>(
                x,
                y,
                sample_weights,
                samples,
                mtry,
                order,
                parameters,
            ),
            None if samples.iter().all(|&s| s <= 1) => {
                // Note: if any sample has count zero, it will be filtered out, so we should
                // use the smallest possible NodeElement
                Self::grow::<UnitElement>(x, y, None, samples, mtry, order, parameters)
            }
            None => Self::grow::<CountedElement>(x, y, None, samples, mtry, order, parameters),
        }
    }

    fn grow<E: NodeElement>(
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

        let counts: Vec<u32> = samples.iter().map(|&s| s as u32).collect();
        // number of distinct rows of x in this tree
        let n_kept = counts.iter().filter(|&&c| c > 0).count();

        let sorted_by_feature: Vec<Vec<E>> = order
            .iter()
            .map(|col_order| {
                let mut out = vec![E::default(); n_kept + 1]; // preallocate, one additional place
                let mut w = 0usize; // index to write to
                for &i in col_order {
                    let c = counts[i];
                    out[w] = E::new(i, c as usize, mass_of(i, &samples, sample_weights));
                    w += (c > 0) as usize; // update w in a branchless way
                }
                out.truncate(n_kept);
                out
            })
            .collect();

        let end_idx = sorted_by_feature[0].len();

        let mut workspace = SplitWorkspace::new(
            vec![false; x.shape().0],
            vec![E::default(); end_idx],
            sorted_by_feature,
            x.shape().1,
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

        let mut visitor_stack: Vec<NodeVisitor<'_, TX, TY, X, Y>> = Vec::new();

        if base_tree.find_best_cutoff(&mut visitor, mtry, mass, &mut rng, &mut workspace) {
            visitor_stack.push(visitor);
        }

        let max_depth = base_tree.parameters().max_depth.unwrap_or(u16::MAX);
        while let Some(node) = visitor_stack.pop() {
            if node.level < max_depth {
                base_tree.split(node, mtry, &mut visitor_stack, &mut rng, &mut workspace);
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
            let node = &self.nodes[node_id];
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

    fn find_best_cutoff<E: NodeElement>(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, TY, X, Y>,
        mtry: usize,
        mass: f64,
        rng: &mut impl rand::Rng,
        workspace: &mut SplitWorkspace<E>,
    ) -> bool {
        let n: usize = workspace.sorted_by_feature[0][visitor.start_idx..visitor.end_idx]
            .iter()
            .map(|elem| elem.count())
            .sum();

        if n < self.parameters().min_samples_split {
            return false;
        }

        let sum = self.nodes[visitor.node].output * mass;

        workspace.variables.shuffle(rng);

        let parent_gain = mass * self.nodes[visitor.node].output * self.nodes[visitor.node].output;

        let splitter = self.parameters().splitter;

        for variable in workspace.variables.iter().take(mtry) {
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

        self.nodes[visitor.node].split_score.is_some()
    }

    fn find_random_split<E: NodeElement>(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, TY, X, Y>,
        n: usize,
        mass: f64,
        sum: f64,
        parent_gain: f64,
        j: usize,
        rng: &mut impl rand::Rng,
        workspace: &SplitWorkspace<E>,
    ) {
        if visitor.start_idx == visitor.end_idx {
            return;
        }
        let first_elem = workspace.sorted_by_feature[j][visitor.start_idx];
        let min_val = visitor.x.get((first_elem.row(), j));
        let last_elem = workspace.sorted_by_feature[j][visitor.end_idx - 1];
        let max_val = visitor.x.get((last_elem.row(), j));

        // Convert min_val and max_val to f64 first, so that values that are different in type TX
        // but equal in f64 do not end up panicking the call to the rng.random_range function.
        let min_val = min_val.to_f64().unwrap();
        let max_val = max_val.to_f64().unwrap();

        if min_val >= max_val {
            return;
        }

        let split_value = rng.random_range(min_val..max_val);

        let mut true_sum = 0f64;
        let mut true_mass = 0f64;
        let mut true_count = 0;
        for elem in &workspace.sorted_by_feature[j][visitor.start_idx..visitor.end_idx] {
            if visitor.x.get((elem.row(), j)).to_f64().unwrap() <= split_value {
                true_sum += elem.mass() * visitor.y.get(elem.row()).to_f64().unwrap();
                true_count += elem.count();
                true_mass += elem.mass();
            } else {
                break;
            }
        }

        let false_count = n - true_count;

        if true_count < self.parameters().min_samples_leaf
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

    fn find_best_split<E: NodeElement>(
        &mut self,
        visitor: &mut NodeVisitor<'_, TX, TY, X, Y>,
        n: usize,
        mass: f64,
        sum: f64,
        parent_gain: f64,
        j: usize,
        workspace: &SplitWorkspace<E>,
    ) {
        let mut true_sum = 0f64;
        let mut true_count = 0;
        let mut true_mass = 0f64;
        let mut prevx = Option::None;

        for elem in &workspace.sorted_by_feature[j][visitor.start_idx..visitor.end_idx] {
            // Comparison for equality is done directly on f64 to avoid later problems
            // with values being unequal in TX but equal in f64.
            let x_ij = visitor.x.get((elem.row(), j)).to_f64().unwrap();

            if prevx.is_none() || x_ij == prevx.unwrap() {
                prevx = Some(x_ij);
                true_count += elem.count();
                true_mass += elem.mass();
                true_sum += elem.mass() * visitor.y.get(elem.row()).to_f64().unwrap();
                continue;
            }

            let false_count = n - true_count;

            if true_count < self.parameters().min_samples_leaf
                || false_count < self.parameters().min_samples_leaf
            {
                prevx = Some(x_ij);
                true_count += elem.count();
                true_mass += elem.mass();
                true_sum += elem.mass() * visitor.y.get(elem.row()).to_f64().unwrap();
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

            if self.nodes[visitor.node].split_score.is_none()
                || gain > self.nodes[visitor.node].split_score.unwrap()
            {
                let prev = prevx.unwrap();
                // compute the midpoint in a way that does not overflow when x_ij and prevx have the same sign
                let mid = prev + (x_ij - prev) / 2.0;

                // The midpoint should put prevx in the true branch, and x_ij in the false branch.
                // However, mid could round up to x_ij, in which case we would also assign x_ij to the true branch.
                // In this case, we set the midpoint equal to prevx, which is always smaller
                // than x_ij.
                let threshold = if mid < x_ij { mid } else { prev };
                self.nodes[visitor.node].split_feature = j;
                self.nodes[visitor.node].split_value = Option::Some(threshold);
                self.nodes[visitor.node].split_score = Option::Some(gain);

                visitor.true_child_output = true_mean;
                visitor.false_child_output = false_mean;
            }

            prevx = Some(x_ij);
            true_sum += elem.mass() * visitor.y.get(elem.row()).to_f64().unwrap();
            true_count += elem.count();
            true_mass += elem.mass();
        }
    }

    /// Apply the split that was found for `visitor.node`: add its two child nodes and
    /// partition the node's range in each `sorted_by_feature` column into a true part
    /// and a false part. Then find the best cutoff for each child. If a child can be
    /// split, push it on `visitor_stack`.
    /// If a branch has fewer than `min_samples_leaf` samples, clear the split and
    /// keep the node as a leaf.
    fn split<'a, E: NodeElement>(
        &mut self,
        visitor: NodeVisitor<'a, TX, TY, X, Y>,
        mtry: usize,
        visitor_stack: &mut Vec<NodeVisitor<'a, TX, TY, X, Y>>,
        rng: &mut impl rand::Rng,
        workspace: &mut SplitWorkspace<E>,
    ) -> bool {
        let this_node = &self.nodes[visitor.node];

        let mut true_count = 0usize;
        let mut true_mass = 0f64;
        let mut false_count = 0usize;
        let mut false_mass = 0f64;
        // for each row_index, does it belong in the true branch or not?
        let in_true_branch = &mut workspace.in_true_branch;

        // the column for split_feature is in order, so all the true samples should come first, all the false samples second
        let split_feature = this_node.split_feature;
        let col = &workspace.sorted_by_feature[split_feature][visitor.start_idx..visitor.end_idx];
        let mut n_true = 0usize;
        for e in col {
            if !is_true_sample(e, visitor.x, this_node) {
                break;
            }
            in_true_branch[e.row()] = true;
            true_count += e.count();
            true_mass += e.mass();
            n_true += 1;
        }
        for e in &col[n_true..] {
            in_true_branch[e.row()] = false;
            false_count += e.count();
            false_mass += e.mass();
        }

        // Stop early if it is clear that there will be too few examples in the leaf
        if true_count < self.parameters().min_samples_leaf
            || false_count < self.parameters().min_samples_leaf
        {
            self.nodes[visitor.node].split_feature = 0;
            self.nodes[visitor.node].split_value = Option::None;
            self.nodes[visitor.node].split_score = Option::None;

            return false;
        }

        // Add the child nodes to the tree
        let split_idx = visitor.start_idx + n_true;

        let true_child_idx = self.nodes.len();

        self.nodes.push(Node::new(visitor.true_child_output));
        let false_child_idx = self.nodes.len();
        self.nodes.push(Node::new(visitor.false_child_output));

        self.nodes[visitor.node].true_child = Some(true_child_idx);
        self.nodes[visitor.node].false_child = Some(false_child_idx);

        let child_level = visitor.level + 1;
        self.depth = u16::max(self.depth, child_level);

        // If the child nodes can not be split any further, there is no point in partitioning the ranges
        let max_depth = self.parameters().max_depth.unwrap_or(u16::MAX);
        let min_split = self.parameters().min_samples_split;
        let true_can_split = child_level < max_depth && true_count >= min_split;
        let false_can_split = child_level < max_depth && false_count >= min_split;
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
            child_level,
        );

        if true_can_split
            && self.find_best_cutoff(&mut true_visitor, mtry, true_mass, rng, workspace)
        {
            visitor_stack.push(true_visitor);
        }

        let mut false_visitor = NodeVisitor::<TX, TY, X, Y>::new(
            false_child_idx,
            split_idx,
            visitor.end_idx,
            visitor.x,
            visitor.y,
            child_level,
        );

        if false_can_split
            && self.find_best_cutoff(&mut false_visitor, mtry, false_mass, rng, workspace)
        {
            visitor_stack.push(false_visitor);
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
    #[cfg(target_pointer_width = "64")]
    fn check_row_count_rejects_more_rows_than_u32_max() {
        assert!(check_row_count(0).is_ok());
        assert!(check_row_count(u32::MAX as usize).is_ok());
        assert_eq!(
            check_row_count(u32::MAX as usize + 1).err(),
            Some(Failed::fit(
                "Number of rows in x must not be larger than u32::MAX"
            ))
        );
    }

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

    // Above 2^53, f64 cannot hold every integer. Split search compares values in TX,
    // but partitioning and predict compare them as f64 against an f64 threshold.
    const TWO_POW_53: i64 = 1 << 53;

    fn split_value_parameters(splitter: Splitter) -> BaseTreeRegressorParameters {
        BaseTreeRegressorParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: Some(42),
            splitter,
        }
    }

    #[test]
    fn large_integer_midpoint_threshold_separates_values() {
        // Both values are exact in f64, but the midpoint 2^53 + 3 is not.
        // It rounds to 2^53 + 4 (ties to even), so both rows go to the true branch.
        let x =
            DenseMatrix::from_2d_vec(&vec![vec![TWO_POW_53 + 2], vec![TWO_POW_53 + 4]]).unwrap();
        let y = vec![0.0f64, 10.0];

        let tree =
            BaseTreeRegressor::fit_inner(&x, &y, None, split_value_parameters(Splitter::Best))
                .expect("Fit should work");

        let threshold = tree.nodes()[0].split_value.expect("Root should split");
        assert!((TWO_POW_53 + 2) as f64 <= threshold);
        assert!(((TWO_POW_53 + 4) as f64) > threshold);

        let y_hat = tree.predict(&x).expect("Predict should work");
        assert_eq!(y_hat, vec![0.0, 10.0]);
    }

    #[test]
    fn large_integer_values_equal_as_f64_do_not_split() {
        // 2^53 and 2^53 + 1 differ as i64 but are the same f64. No f64 threshold can
        // separate them, so the tree must not split and must predict the mean.
        let x = DenseMatrix::from_2d_vec(&vec![vec![TWO_POW_53], vec![TWO_POW_53 + 1]]).unwrap();
        let y = vec![0.0f64, 10.0];

        let tree =
            BaseTreeRegressor::fit_inner(&x, &y, None, split_value_parameters(Splitter::Best))
                .expect("Fit should work");

        let y_hat = tree.predict(&x).expect("Predict should work");
        assert_eq!(y_hat, vec![5.0, 5.0]);
    }

    #[test]
    fn large_integer_values_equal_as_f64_random_splitter() {
        // min_val < max_val as i64, but the f64 range is empty. This must not panic.
        let x = DenseMatrix::from_2d_vec(&vec![vec![TWO_POW_53], vec![TWO_POW_53 + 1]]).unwrap();
        let y = vec![0.0f64, 10.0];

        let tree =
            BaseTreeRegressor::fit_inner(&x, &y, None, split_value_parameters(Splitter::Random))
                .expect("Fit should work");

        let y_hat = tree.predict(&x).expect("Predict should work");
        assert_eq!(y_hat, vec![5.0, 5.0]);
    }

    #[test]
    fn adjacent_f64_midpoint_threshold_separates_values() {
        // a and b are adjacent f64 values. Their exact midpoint 1 + 1.5 * EPSILON is not
        // representable. It is a tie, and ties round to the even mantissa, which is b.
        // So the threshold becomes b, and both rows go to the true branch.
        let a = 1.0 + f64::EPSILON;
        let b = 1.0 + 2.0 * f64::EPSILON;
        assert_eq!((a + b) / 2.0, b); // the rounding that causes the problem

        let x = DenseMatrix::from_2d_vec(&vec![vec![a], vec![b]]).unwrap();
        let y = vec![0.0f64, 10.0];

        let tree =
            BaseTreeRegressor::fit_inner(&x, &y, None, split_value_parameters(Splitter::Best))
                .expect("Fit should work");

        let threshold = tree.nodes()[0].split_value.expect("Root should split");
        assert!(a <= threshold);
        assert!(b > threshold);

        let y_hat = tree.predict(&x).expect("Predict should work");
        assert_eq!(y_hat, vec![0.0, 10.0]);
    }

    #[test]
    fn test_node_element_sizes() {
        assert_eq!(std::mem::size_of::<UnitElement>(), 4);
        assert_eq!(std::mem::size_of::<CountedElement>(), 8);
        assert_eq!(std::mem::size_of::<WeightedElement>(), 16);
    }

    type TestTree = BaseTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>;

    fn params(
        max_depth: Option<u16>,
        min_samples_leaf: usize,
        min_samples_split: usize,
        splitter: Splitter,
    ) -> BaseTreeRegressorParameters {
        BaseTreeRegressorParameters {
            max_depth,
            min_samples_leaf,
            min_samples_split,
            seed: Some(42),
            splitter,
        }
    }

    /// Row order of each column, computed in the same way as in `fit_inner`.
    fn order_of(x: &DenseMatrix<f64>) -> Vec<Vec<usize>> {
        (0..x.shape().1)
            .map(|j| {
                let mut col: Vec<f64> = x.get_col(j).iterator(0).copied().collect();
                col.argsort_mut()
            })
            .collect()
    }

    /// Index of the leaf that gets row `row` of `x`. Uses the same rule as `predict_for_row`.
    fn leaf_of(tree: &TestTree, x: &DenseMatrix<f64>, row: usize) -> usize {
        let mut node_id = 0;
        while let Some(true_child) = tree.nodes()[node_id].true_child {
            let node = &tree.nodes()[node_id];
            node_id = if *x.get((row, node.split_feature)) <= node.split_value.unwrap() {
                true_child
            } else {
                node.false_child.unwrap()
            };
        }
        node_id
    }

    /// Number of training samples in each leaf of `tree`.
    /// `counts[i]` is the number of times that row `i` of `x` is in the training set.
    fn leaf_counts(tree: &TestTree, x: &DenseMatrix<f64>, counts: &[usize]) -> Vec<usize> {
        let mut per_node = vec![0usize; tree.nodes().len()];
        for (row, &count) in counts.iter().enumerate() {
            per_node[leaf_of(tree, x, row)] += count;
        }
        tree.nodes()
            .iter()
            .zip(per_node)
            .filter(|(node, _)| node.true_child.is_none()) // filter out the leaves
            .map(|(_, count)| count)
            .collect()
    }

    /// Both trees must have the same structure, the same splits and the same outputs.
    /// `split_score` is not compared, because a different summation order can change it.
    fn assert_same_tree(a: &TestTree, b: &TestTree) {
        assert_eq!(a.nodes().len(), b.nodes().len());
        assert_eq!(a.depth, b.depth);
        for (na, nb) in a.nodes().iter().zip(b.nodes().iter()) {
            assert_eq!(na.split_feature, nb.split_feature);
            assert_eq!(na.split_value, nb.split_value);
            assert_eq!(na.true_child, nb.true_child);
            assert_eq!(na.false_child, nb.false_child);
            assert!((na.output - nb.output).abs() < 1e-9);
        }
    }

    /// Repeat row `i` of `x`, `y` and `w` `counts[i]` times.
    fn repeat_rows(
        x: &[Vec<f64>],
        y: &[f64],
        w: &[f64],
        counts: &[usize],
    ) -> (DenseMatrix<f64>, Vec<f64>, Vec<f64>) {
        let mut x_rep = Vec::new();
        let mut y_rep = Vec::new();
        let mut w_rep = Vec::new();
        for (i, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                x_rep.push(x[i].clone());
                y_rep.push(y[i]);
                w_rep.push(w[i]);
            }
        }
        (DenseMatrix::from_2d_vec(&x_rep).unwrap(), y_rep, w_rep)
    }

    /// Data for the element path tests: ties in both columns, integer targets.
    fn element_path_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let x = vec![
            vec![1.0, 4.0],
            vec![2.0, 4.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0],
            vec![5.0, 2.0],
            vec![6.0, 1.0],
            vec![6.0, 5.0],
            vec![7.0, 3.0],
        ];
        let y = vec![3.0, 8.0, 1.0, 9.0, 4.0, 12.0, 7.0, 2.0];
        (x, y)
    }

    #[test]
    fn max_depth_boundary() {
        let x = DenseMatrix::from_iterator((1..=8).map(|i| i as f64), 8, 1, 0);
        let y = vec![0.0f64, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];

        // The root is at level 1. A node at level `max_depth` is not split.
        // (max_depth, number of nodes, depth)
        let cases = [
            (Some(1), 1, 0),
            (Some(2), 3, 2),
            (Some(3), 7, 3),
            (None, 15, 4),
        ];
        for (max_depth, n_nodes, depth) in cases {
            let tree: TestTree =
                BaseTreeRegressor::fit_inner(&x, &y, None, params(max_depth, 1, 2, Splitter::Best))
                    .expect("Fit should work");
            assert_eq!(tree.nodes().len(), n_nodes, "max_depth: {max_depth:?}");
            assert_eq!(tree.depth, depth, "max_depth: {max_depth:?}");
        }

        // With max_depth = 2 the root split separates the two halves.
        let tree: TestTree =
            BaseTreeRegressor::fit_inner(&x, &y, None, params(Some(2), 1, 2, Splitter::Best))
                .expect("Fit should work");
        assert_eq!(tree.nodes()[0].split_value, Some(4.5));
        let y_hat = tree.predict(&x).expect("Predict should work");
        assert_eq!(y_hat, vec![5.5, 5.5, 5.5, 5.5, 105.5, 105.5, 105.5, 105.5]);

        // No limit is the same as the largest limit, and the leaves memorize y.
        let tree_none: TestTree =
            BaseTreeRegressor::fit_inner(&x, &y, None, params(None, 1, 2, Splitter::Best))
                .expect("Fit should work");
        let tree_max: TestTree = BaseTreeRegressor::fit_inner(
            &x,
            &y,
            None,
            params(Some(u16::MAX), 1, 2, Splitter::Best),
        )
        .expect("Fit should work");
        assert_same_tree(&tree_none, &tree_max);
        assert_eq!(tree_none.predict(&x).expect("Predict should work"), y);

        for max_depth in 1..=4 {
            let tree: TestTree = BaseTreeRegressor::fit_inner(
                &x,
                &y,
                None,
                params(Some(max_depth), 1, 2, Splitter::Random),
            )
            .expect("Fit should work");
            assert!(tree.depth <= max_depth, "max_depth: {max_depth}");
        }
    }

    #[test]
    fn min_samples_split_one_child_only() {
        // The root split gives a true child with 3 samples and a false child with 2 samples.
        // With min_samples_split = 3, only the true child can split.
        let x = DenseMatrix::from_2d_vec(&vec![
            vec![1.0_f64],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ])
        .unwrap();
        let y = vec![0.0f64, 1.0, 2.0, 100.0, 101.0];

        let tree: TestTree =
            BaseTreeRegressor::fit_inner(&x, &y, None, params(None, 1, 3, Splitter::Best))
                .expect("Fit should work");
        assert_eq!(tree.nodes()[0].split_value, Some(3.5));
        let true_child = tree.nodes()[0].true_child.unwrap();
        let false_child = tree.nodes()[0].false_child.unwrap();
        assert!(tree.nodes()[true_child].true_child.is_some());
        assert!(tree.nodes()[false_child].true_child.is_none());
        assert_eq!(tree.nodes().len(), 5);
        assert_eq!(tree.depth, 3);

        // If min_samples_split is larger than the number of samples, the root is a leaf.
        for splitter in [Splitter::Best, Splitter::Random] {
            let tree: TestTree =
                BaseTreeRegressor::fit_inner(&x, &y, None, params(None, 1, 6, splitter))
                    .expect("Fit should work");
            assert_eq!(tree.nodes().len(), 1, "splitter: {splitter:?}");
        }
    }

    #[test]
    fn min_samples_leaf_boundary() {
        let x = DenseMatrix::from_2d_vec(&vec![vec![1.0_f64], vec![2.0], vec![3.0], vec![4.0]])
            .unwrap();
        let y = vec![0.0f64, 0.0, 0.0, 100.0];

        // With no limit, the best split isolates the last sample.
        let tree: TestTree =
            BaseTreeRegressor::fit_inner(&x, &y, None, params(None, 1, 2, Splitter::Best))
                .expect("Fit should work");
        assert_eq!(tree.nodes()[0].split_value, Some(3.5));

        // The best split is not permitted. The split with exactly 2 samples on each side is used.
        let tree: TestTree =
            BaseTreeRegressor::fit_inner(&x, &y, None, params(None, 2, 2, Splitter::Best))
                .expect("Fit should work");
        assert_eq!(tree.nodes()[0].split_value, Some(2.5));
        assert_eq!(tree.nodes().len(), 3);
        assert_eq!(leaf_counts(&tree, &x, &[1; 4]), vec![2, 2]);

        // No split can give 3 samples on each side.
        for splitter in [Splitter::Best, Splitter::Random] {
            let tree: TestTree =
                BaseTreeRegressor::fit_inner(&x, &y, None, params(None, 3, 2, splitter))
                    .expect("Fit should work");
            assert_eq!(tree.nodes().len(), 1, "splitter: {splitter:?}");
            assert_eq!(tree.nodes()[0].output, 25.0);
        }
    }

    // Fit many different trees with many different options and check that in each case min_samples_leaf is correct.
    #[test]
    fn min_samples_leaf_holds_for_all_element_paths() {
        let mut rng = get_rng_impl(Some(7));
        let n = 40;
        let x_rows: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..3).map(|_| rng.random_range(0..6) as f64).collect())
            .collect();
        let y: Vec<f64> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
        let counts: Vec<usize> = (0..n).map(|_| rng.random_range(0..4usize)).collect();
        let weights: Vec<f64> = (0..n).map(|_| rng.random_range(0.1..3.0)).collect();
        let ones = vec![1usize; n];
        let x = DenseMatrix::from_2d_vec(&x_rows).unwrap();
        let order = order_of(&x);

        // (weights, counts): UnitElement, CountedElement and WeightedElement
        let paths = [
            (None, &ones),
            (None, &counts),
            (Some(weights.as_slice()), &counts),
        ];
        for (sample_weights, samples) in paths {
            for splitter in [Splitter::Best, Splitter::Random] {
                for min_samples_leaf in [1, 3, 5] {
                    for seed in 0..5 {
                        let parameters = BaseTreeRegressorParameters {
                            seed: Some(seed),
                            ..params(None, min_samples_leaf, 2, splitter)
                        };
                        let tree: TestTree = BaseTreeRegressor::fit_weak_learner(
                            &x,
                            &y,
                            sample_weights,
                            samples.clone(),
                            3,
                            &order,
                            parameters,
                        )
                        .expect("Fit should work");
                        let message = format!(
                            "weights: {}, splitter: {splitter:?}, leaf: {min_samples_leaf}, seed: {seed}",
                            sample_weights.is_some()
                        );
                        assert!(tree.nodes().len() > 1, "{message}"); // make sure the tree has a split
                        let leaves = leaf_counts(&tree, &x, samples);
                        assert!(
                            leaves.iter().all(|&c| c >= min_samples_leaf),
                            "{message}, leaves: {leaves:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn constant_features_do_not_split() {
        // all features are constant, so no split is possible
        let x = DenseMatrix::from_2d_vec(&vec![
            vec![1.0_f64, 7.0],
            vec![1.0, 7.0],
            vec![1.0, 7.0],
            vec![1.0, 7.0],
        ])
        .unwrap();
        let y = vec![1.0f64, 2.0, 3.0, 6.0];

        for splitter in [Splitter::Best, Splitter::Random] {
            let tree: TestTree =
                BaseTreeRegressor::fit_inner(&x, &y, None, params(None, 1, 2, splitter))
                    .expect("Fit should work");
            assert_eq!(tree.nodes().len(), 1, "splitter: {splitter:?}");
            assert_eq!(tree.depth, 0, "splitter: {splitter:?}");
            let y_hat = tree.predict(&x).expect("Predict should work");
            assert_eq!(y_hat, vec![3.0; 4], "splitter: {splitter:?}");
        }
    }

    #[test]
    fn constant_column_is_skipped() {
        // Column 0 is constant, so only column 1 can give a split.
        let x = DenseMatrix::from_2d_vec(&vec![
            vec![7.0_f64, 1.0],
            vec![7.0, 2.0],
            vec![7.0, 3.0],
            vec![7.0, 4.0],
        ])
        .unwrap();
        let y = vec![0.0f64, 0.0, 10.0, 10.0];

        for splitter in [Splitter::Best, Splitter::Random] {
            let tree: TestTree =
                BaseTreeRegressor::fit_inner(&x, &y, None, params(None, 1, 2, splitter))
                    .expect("Fit should work");
            assert!(tree.nodes().len() > 1, "splitter: {splitter:?}");
            assert!(
                tree.nodes()
                    .iter()
                    .filter(|node| node.true_child.is_some())
                    .all(|node| node.split_feature == 1),
                "splitter: {splitter:?}"
            );
        }

        let tree: TestTree =
            BaseTreeRegressor::fit_inner(&x, &y, None, params(None, 1, 2, Splitter::Best))
                .expect("Fit should work");
        assert_eq!(tree.nodes()[0].split_value, Some(2.5));
    }

    #[test]
    fn duplicate_columns_pick_random_column() {
        // Both columns are identical. Their gains are equal. As in sklearn, the attributes
        // are considered in a random order, thus the column of each split is random.
        // Apart from the column, the tree is the same as the tree for one column.
        let values = [1.0_f64, 2.0, 2.0, 3.0, 5.0, 6.0];
        let y = vec![1.0f64, 4.0, 2.0, 9.0, 3.0, 8.0];
        let x_single =
            DenseMatrix::from_2d_vec(&values.iter().map(|&v| vec![v]).collect::<Vec<_>>()).unwrap();
        let x_double =
            DenseMatrix::from_2d_vec(&values.iter().map(|&v| vec![v, v]).collect::<Vec<_>>())
                .unwrap();

        let mut used_columns = [false; 2];
        for seed in 0..10 {
            let parameters = BaseTreeRegressorParameters {
                seed: Some(seed),
                ..params(None, 1, 2, Splitter::Best)
            };
            let tree_single: TestTree =
                BaseTreeRegressor::fit_inner(&x_single, &y, None, parameters.clone())
                    .expect("Fit should work");
            let tree_double: TestTree =
                BaseTreeRegressor::fit_inner(&x_double, &y, None, parameters)
                    .expect("Fit should work");

            assert!(tree_double.nodes().len() > 1);
            assert_eq!(tree_single.nodes().len(), tree_double.nodes().len());
            assert_eq!(tree_single.depth, tree_double.depth);
            for (a, b) in tree_single.nodes().iter().zip(tree_double.nodes().iter()) {
                assert_eq!(a.split_value, b.split_value);
                assert_eq!(a.true_child, b.true_child);
                assert_eq!(a.false_child, b.false_child);
                assert!((a.output - b.output).abs() < 1e-9);
                if b.true_child.is_some() {
                    used_columns[b.split_feature] = true;
                }
            }
        }
        assert_eq!(used_columns, [true, true], "both columns must be used");
    }

    #[test]
    fn counted_and_unit_elements_match_repeated_rows() {
        // Checks that repeating rows and giving them a count > 0 yields the same trees.
        // Do not pose constraints on min_samples_leaf and min_samples_split. At the moment
        // this should make no difference but it may do in the future.
        let (x_rows, y) = element_path_data();
        let x = DenseMatrix::from_2d_vec(&x_rows).unwrap();
        let order = order_of(&x);
        let n = y.len();

        // Counts 0 or 1 use UnitElement. Counts larger than 1 use CountedElement.
        // Rows with count 0 must have no effect.
        let all_counts = [vec![1, 0, 1, 1, 0, 1, 1, 0], vec![2, 0, 1, 3, 1, 0, 2, 1]];
        for counts in all_counts {
            let (x_rep, y_rep, _) = repeat_rows(&x_rows, &y, &vec![1.0; n], &counts);
            for splitter in [Splitter::Best, Splitter::Random] {
                for max_depth in [Some(2), None] {
                    let parameters = params(max_depth, 1, 2, splitter);
                    let tree: TestTree = BaseTreeRegressor::fit_weak_learner(
                        &x,
                        &y,
                        None,
                        counts.clone(),
                        2,
                        &order,
                        parameters.clone(),
                    )
                    .expect("Fit should work");
                    let tree_rep: TestTree =
                        BaseTreeRegressor::fit_inner(&x_rep, &y_rep, None, parameters)
                            .expect("Fit should work");

                    assert!(tree.nodes().len() > 1);
                    assert_same_tree(&tree, &tree_rep);
                    let y_hat = tree.predict(&x).expect("Predict should work");
                    let y_hat_rep = tree_rep.predict(&x).expect("Predict should work");
                    assert!(mean_absolute_error(&y_hat, &y_hat_rep) < 1e-9);
                }
            }
        }
    }

    #[test]
    fn weighted_element_with_counts_matches_repeated_rows() {
        // The mass of a row is count * weight.  Weights are chosen to be perfectly representable (power of two in the fraction denominator).
        // This keeps all sums exact.
        // Checks that WeightedElement handles counts different from one correctly.
        let (x_rows, y) = element_path_data();
        let x = DenseMatrix::from_2d_vec(&x_rows).unwrap();
        let order = order_of(&x);
        let weights = vec![0.5, 1.5, 2.0, 0.25, 1.0, 3.0, 0.75, 1.25];
        let counts = vec![2, 0, 1, 3, 1, 0, 2, 1];
        let (x_rep, y_rep, w_rep) = repeat_rows(&x_rows, &y, &weights, &counts);

        for splitter in [Splitter::Best, Splitter::Random] {
            for max_depth in [Some(2), None] {
                let parameters = params(max_depth, 1, 2, splitter);
                let tree: TestTree = BaseTreeRegressor::fit_weak_learner(
                    &x,
                    &y,
                    Some(&weights),
                    counts.clone(),
                    2,
                    &order,
                    parameters.clone(),
                )
                .expect("Fit should work");
                // in this tree, there are no counts but the rows are repeated physically
                let tree_rep: TestTree =
                    BaseTreeRegressor::fit_inner(&x_rep, &y_rep, Some(&w_rep), parameters)
                        .expect("Fit should work");

                assert!(tree.nodes().len() > 1);
                assert_same_tree(&tree, &tree_rep);
                let y_hat = tree.predict(&x).expect("Predict should work");
                let y_hat_rep = tree_rep.predict(&x).expect("Predict should work");
                assert!(mean_absolute_error(&y_hat, &y_hat_rep) < 1e-9);
            }
        }
    }
}
