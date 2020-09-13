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
//! use smartcore::linalg::naive::dense_matrix::*;
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
//!         ]);
//! let y = vec![ 0., 0., 0., 0., 0., 0., 0., 0.,
//!            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.];
//!
//! let tree = DecisionTreeClassifier::fit(&x, &y, Default::default());
//!
//! let y_hat = tree.predict(&x); // use the same data for prediction
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
use serde::{Deserialize, Serialize};

use crate::algorithm::sort::quick_sort::QuickArgSort;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;

#[derive(Serialize, Deserialize, Debug)]
/// Parameters of Decision Tree
pub struct DecisionTreeClassifierParameters {
    /// Split criteria to use when building a tree.
    pub criterion: SplitCriterion,
    /// The maximum depth of the tree.
    pub max_depth: Option<u16>,
    /// The minimum number of samples required to be at a leaf node.
    pub min_samples_leaf: usize,
    /// The minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
}

/// Decision Tree
#[derive(Serialize, Deserialize, Debug)]
pub struct DecisionTreeClassifier<T: RealNumber> {
    nodes: Vec<Node<T>>,
    parameters: DecisionTreeClassifierParameters,
    num_classes: usize,
    classes: Vec<T>,
    depth: u16,
}

/// The function to measure the quality of a split.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SplitCriterion {
    /// [Gini index](../decision_tree_classifier/index.html)
    Gini,
    /// [Entropy](../decision_tree_classifier/index.html)
    Entropy,
    /// [Classification error](../decision_tree_classifier/index.html)
    ClassificationError,
}

#[derive(Serialize, Deserialize, Debug)]
struct Node<T: RealNumber> {
    index: usize,
    output: usize,
    split_feature: usize,
    split_value: Option<T>,
    split_score: Option<T>,
    true_child: Option<usize>,
    false_child: Option<usize>,
}

impl<T: RealNumber> PartialEq for DecisionTreeClassifier<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.depth != other.depth
            || self.num_classes != other.num_classes
            || self.nodes.len() != other.nodes.len()
        {
            return false;
        } else {
            for i in 0..self.classes.len() {
                if (self.classes[i] - other.classes[i]).abs() > T::epsilon() {
                    return false;
                }
            }
            for i in 0..self.nodes.len() {
                if self.nodes[i] != other.nodes[i] {
                    return false;
                }
            }
            return true;
        }
    }
}

impl<T: RealNumber> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        self.output == other.output
            && self.split_feature == other.split_feature
            && match (self.split_value, other.split_value) {
                (Some(a), Some(b)) => (a - b).abs() < T::epsilon(),
                (None, None) => true,
                _ => false,
            }
            && match (self.split_score, other.split_score) {
                (Some(a), Some(b)) => (a - b).abs() < T::epsilon(),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Default for DecisionTreeClassifierParameters {
    fn default() -> Self {
        DecisionTreeClassifierParameters {
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
        }
    }
}

impl<T: RealNumber> Node<T> {
    fn new(index: usize, output: usize) -> Self {
        Node {
            index: index,
            output: output,
            split_feature: 0,
            split_value: Option::None,
            split_score: Option::None,
            true_child: Option::None,
            false_child: Option::None,
        }
    }
}

struct NodeVisitor<'a, T: RealNumber, M: Matrix<T>> {
    x: &'a M,
    y: &'a Vec<usize>,
    node: usize,
    samples: Vec<usize>,
    order: &'a Vec<Vec<usize>>,
    true_child_output: usize,
    false_child_output: usize,
    level: u16,
    phantom: PhantomData<&'a T>,
}

fn impurity<T: RealNumber>(criterion: &SplitCriterion, count: &Vec<usize>, n: usize) -> T {
    let mut impurity = T::zero();

    match criterion {
        SplitCriterion::Gini => {
            impurity = T::one();
            for i in 0..count.len() {
                if count[i] > 0 {
                    let p = T::from(count[i]).unwrap() / T::from(n).unwrap();
                    impurity = impurity - p * p;
                }
            }
        }

        SplitCriterion::Entropy => {
            for i in 0..count.len() {
                if count[i] > 0 {
                    let p = T::from(count[i]).unwrap() / T::from(n).unwrap();
                    impurity = impurity - p * p.log2();
                }
            }
        }
        SplitCriterion::ClassificationError => {
            for i in 0..count.len() {
                if count[i] > 0 {
                    impurity = impurity.max(T::from(count[i]).unwrap() / T::from(n).unwrap());
                }
            }
            impurity = (T::one() - impurity).abs();
        }
    }

    return impurity;
}

impl<'a, T: RealNumber, M: Matrix<T>> NodeVisitor<'a, T, M> {
    fn new(
        node_id: usize,
        samples: Vec<usize>,
        order: &'a Vec<Vec<usize>>,
        x: &'a M,
        y: &'a Vec<usize>,
        level: u16,
    ) -> Self {
        NodeVisitor {
            x: x,
            y: y,
            node: node_id,
            samples: samples,
            order: order,
            true_child_output: 0,
            false_child_output: 0,
            level: level,
            phantom: PhantomData,
        }
    }
}

pub(in crate) fn which_max(x: &Vec<usize>) -> usize {
    let mut m = x[0];
    let mut which = 0;

    for i in 1..x.len() {
        if x[i] > m {
            m = x[i];
            which = i;
        }
    }

    return which;
}

impl<T: RealNumber> DecisionTreeClassifier<T> {
    /// Build a decision tree classifier from the training data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        parameters: DecisionTreeClassifierParameters,
    ) -> DecisionTreeClassifier<T> {
        let (x_nrows, num_attributes) = x.shape();
        let samples = vec![1; x_nrows];
        DecisionTreeClassifier::fit_weak_learner(x, y, samples, num_attributes, parameters)
    }

    pub(crate) fn fit_weak_learner<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        samples: Vec<usize>,
        mtry: usize,
        parameters: DecisionTreeClassifierParameters,
    ) -> DecisionTreeClassifier<T> {
        let y_m = M::from_row_vector(y.clone());
        let (_, y_ncols) = y_m.shape();
        let (_, num_attributes) = x.shape();
        let classes = y_m.unique();
        let k = classes.len();
        if k < 2 {
            panic!("Incorrect number of classes: {}. Should be >= 2.", k);
        }

        let mut yi: Vec<usize> = vec![0; y_ncols];

        for i in 0..y_ncols {
            let yc = y_m.get(0, i);
            yi[i] = classes.iter().position(|c| yc == *c).unwrap();
        }

        let mut nodes: Vec<Node<T>> = Vec::new();

        let mut count = vec![0; k];
        for i in 0..y_ncols {
            count[yi[i]] += samples[i];
        }

        let root = Node::new(0, which_max(&count));
        nodes.push(root);
        let mut order: Vec<Vec<usize>> = Vec::new();

        for i in 0..num_attributes {
            order.push(x.get_col_as_vec(i).quick_argsort_mut());
        }

        let mut tree = DecisionTreeClassifier {
            nodes: nodes,
            parameters: parameters,
            num_classes: k,
            classes: classes,
            depth: 0,
        };

        let mut visitor = NodeVisitor::<T, M>::new(0, samples, &order, &x, &yi, 1);

        let mut visitor_queue: LinkedList<NodeVisitor<T, M>> = LinkedList::new();

        if tree.find_best_cutoff(&mut visitor, mtry) {
            visitor_queue.push_back(visitor);
        }

        while tree.depth < tree.parameters.max_depth.unwrap_or(std::u16::MAX) {
            match visitor_queue.pop_front() {
                Some(node) => tree.split(node, mtry, &mut visitor_queue),
                None => break,
            };
        }

        tree
    }

    /// Predict class value for `x`.
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict<M: Matrix<T>>(&self, x: &M) -> M::RowVector {
        let mut result = M::zeros(1, x.shape().0);

        let (n, _) = x.shape();

        for i in 0..n {
            result.set(0, i, self.classes[self.predict_for_row(x, i)]);
        }

        result.to_row_vector()
    }

    pub(in crate) fn predict_for_row<M: Matrix<T>>(&self, x: &M, row: usize) -> usize {
        let mut result = 0;
        let mut queue: LinkedList<usize> = LinkedList::new();

        queue.push_back(0);

        while !queue.is_empty() {
            match queue.pop_front() {
                Some(node_id) => {
                    let node = &self.nodes[node_id];
                    if node.true_child == None && node.false_child == None {
                        result = node.output;
                    } else {
                        if x.get(row, node.split_feature) <= node.split_value.unwrap_or(T::nan()) {
                            queue.push_back(node.true_child.unwrap());
                        } else {
                            queue.push_back(node.false_child.unwrap());
                        }
                    }
                }
                None => break,
            };
        }

        return result;
    }

    fn find_best_cutoff<M: Matrix<T>>(
        &mut self,
        visitor: &mut NodeVisitor<T, M>,
        mtry: usize,
    ) -> bool {
        let (n_rows, n_attr) = visitor.x.shape();

        let mut label = Option::None;
        let mut is_pure = true;
        for i in 0..n_rows {
            if visitor.samples[i] > 0 {
                if label == Option::None {
                    label = Option::Some(visitor.y[i]);
                } else if visitor.y[i] != label.unwrap() {
                    is_pure = false;
                    break;
                }
            }
        }

        if is_pure {
            return false;
        }

        let n = visitor.samples.iter().sum();

        if n <= self.parameters.min_samples_split {
            return false;
        }

        let mut count = vec![0; self.num_classes];
        let mut false_count = vec![0; self.num_classes];
        for i in 0..n_rows {
            if visitor.samples[i] > 0 {
                count[visitor.y[i]] += visitor.samples[i];
            }
        }

        let parent_impurity = impurity(&self.parameters.criterion, &count, n);

        let mut variables = vec![0; n_attr];
        for i in 0..n_attr {
            variables[i] = i;
        }

        if mtry < n_attr {
            variables.shuffle(&mut rand::thread_rng());
        }

        for j in 0..mtry {
            self.find_best_split(
                visitor,
                n,
                &count,
                &mut false_count,
                parent_impurity,
                variables[j],
            );
        }

        self.nodes[visitor.node].split_score != Option::None
    }

    fn find_best_split<M: Matrix<T>>(
        &mut self,
        visitor: &mut NodeVisitor<T, M>,
        n: usize,
        count: &Vec<usize>,
        false_count: &mut Vec<usize>,
        parent_impurity: T,
        j: usize,
    ) {
        let mut true_count = vec![0; self.num_classes];
        let mut prevx = T::nan();
        let mut prevy = 0;

        for i in visitor.order[j].iter() {
            if visitor.samples[*i] > 0 {
                if prevx.is_nan() || visitor.x.get(*i, j) == prevx || visitor.y[*i] == prevy {
                    prevx = visitor.x.get(*i, j);
                    prevy = visitor.y[*i];
                    true_count[visitor.y[*i]] += visitor.samples[*i];
                    continue;
                }

                let tc = true_count.iter().sum();
                let fc = n - tc;

                if tc < self.parameters.min_samples_leaf || fc < self.parameters.min_samples_leaf {
                    prevx = visitor.x.get(*i, j);
                    prevy = visitor.y[*i];
                    true_count[visitor.y[*i]] += visitor.samples[*i];
                    continue;
                }

                for l in 0..self.num_classes {
                    false_count[l] = count[l] - true_count[l];
                }

                let true_label = which_max(&true_count);
                let false_label = which_max(false_count);
                let gain = parent_impurity
                    - T::from(tc).unwrap() / T::from(n).unwrap()
                        * impurity(&self.parameters.criterion, &true_count, tc)
                    - T::from(fc).unwrap() / T::from(n).unwrap()
                        * impurity(&self.parameters.criterion, &false_count, fc);

                if self.nodes[visitor.node].split_score == Option::None
                    || gain > self.nodes[visitor.node].split_score.unwrap()
                {
                    self.nodes[visitor.node].split_feature = j;
                    self.nodes[visitor.node].split_value =
                        Option::Some((visitor.x.get(*i, j) + prevx) / T::two());
                    self.nodes[visitor.node].split_score = Option::Some(gain);
                    visitor.true_child_output = true_label;
                    visitor.false_child_output = false_label;
                }

                prevx = visitor.x.get(*i, j);
                prevy = visitor.y[*i];
                true_count[visitor.y[*i]] += visitor.samples[*i];
            }
        }
    }

    fn split<'a, M: Matrix<T>>(
        &mut self,
        mut visitor: NodeVisitor<'a, T, M>,
        mtry: usize,
        visitor_queue: &mut LinkedList<NodeVisitor<'a, T, M>>,
    ) -> bool {
        let (n, _) = visitor.x.shape();
        let mut tc = 0;
        let mut fc = 0;
        let mut true_samples: Vec<usize> = vec![0; n];

        for i in 0..n {
            if visitor.samples[i] > 0 {
                if visitor.x.get(i, self.nodes[visitor.node].split_feature)
                    <= self.nodes[visitor.node].split_value.unwrap_or(T::nan())
                {
                    true_samples[i] = visitor.samples[i];
                    tc += true_samples[i];
                    visitor.samples[i] = 0;
                } else {
                    fc += visitor.samples[i];
                }
            }
        }

        if tc < self.parameters.min_samples_leaf || fc < self.parameters.min_samples_leaf {
            self.nodes[visitor.node].split_feature = 0;
            self.nodes[visitor.node].split_value = Option::None;
            self.nodes[visitor.node].split_score = Option::None;
            return false;
        }

        let true_child_idx = self.nodes.len();
        self.nodes
            .push(Node::new(true_child_idx, visitor.true_child_output));
        let false_child_idx = self.nodes.len();
        self.nodes
            .push(Node::new(false_child_idx, visitor.false_child_output));

        self.nodes[visitor.node].true_child = Some(true_child_idx);
        self.nodes[visitor.node].false_child = Some(false_child_idx);

        self.depth = u16::max(self.depth, visitor.level + 1);

        let mut true_visitor = NodeVisitor::<T, M>::new(
            true_child_idx,
            true_samples,
            visitor.order,
            visitor.x,
            visitor.y,
            visitor.level + 1,
        );

        if self.find_best_cutoff(&mut true_visitor, mtry) {
            visitor_queue.push_back(true_visitor);
        }

        let mut false_visitor = NodeVisitor::<T, M>::new(
            false_child_idx,
            visitor.samples,
            visitor.order,
            visitor.x,
            visitor.y,
            visitor.level + 1,
        );

        if self.find_best_cutoff(&mut false_visitor, mtry) {
            visitor_queue.push_back(false_visitor);
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn gini_impurity() {
        assert!(
            (impurity::<f64>(&SplitCriterion::Gini, &vec![7, 3], 10) - 0.42).abs()
                < std::f64::EPSILON
        );
        assert!(
            (impurity::<f64>(&SplitCriterion::Entropy, &vec![7, 3], 10) - 0.8812908992306927).abs()
                < std::f64::EPSILON
        );
        assert!(
            (impurity::<f64>(&SplitCriterion::ClassificationError, &vec![7, 3], 10) - 0.3).abs()
                < std::f64::EPSILON
        );
    }

    #[test]
    fn fit_predict_iris() {
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
        ]);
        let y = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        assert_eq!(
            y,
            DecisionTreeClassifier::fit(&x, &y, Default::default()).predict(&x)
        );

        assert_eq!(
            3,
            DecisionTreeClassifier::fit(
                &x,
                &y,
                DecisionTreeClassifierParameters {
                    criterion: SplitCriterion::Entropy,
                    max_depth: Some(3),
                    min_samples_leaf: 1,
                    min_samples_split: 2
                }
            )
            .depth
        );
    }

    #[test]
    fn fit_predict_baloons() {
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
        ]);
        let y = vec![
            1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
        ];

        assert_eq!(
            y,
            DecisionTreeClassifier::fit(&x, &y, Default::default()).predict(&x)
        );
    }

    #[test]
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
        ]);
        let y = vec![
            1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
        ];

        let tree = DecisionTreeClassifier::fit(&x, &y, Default::default());

        let deserialized_tree: DecisionTreeClassifier<f64> =
            bincode::deserialize(&bincode::serialize(&tree).unwrap()).unwrap();

        assert_eq!(tree, deserialized_tree);
    }
}
