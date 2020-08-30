use std::collections::LinkedList;
use std::default::Default;
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::algorithm::sort::quick_sort::QuickArgSort;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;

#[derive(Serialize, Deserialize, Debug)]
pub struct DecisionTreeRegressorParameters {
    pub max_depth: Option<u16>,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DecisionTreeRegressor<T: RealNumber> {
    nodes: Vec<Node<T>>,
    parameters: DecisionTreeRegressorParameters,
    depth: u16,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Node<T: RealNumber> {
    index: usize,
    output: T,
    split_feature: usize,
    split_value: Option<T>,
    split_score: Option<T>,
    true_child: Option<usize>,
    false_child: Option<usize>,
}

impl Default for DecisionTreeRegressorParameters {
    fn default() -> Self {
        DecisionTreeRegressorParameters {
            max_depth: None,
            min_samples_leaf: 1,
            min_samples_split: 2,
        }
    }
}

impl<T: RealNumber> Node<T> {
    fn new(index: usize, output: T) -> Self {
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

impl<T: RealNumber> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.output - other.output).abs() < T::epsilon()
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

impl<T: RealNumber> PartialEq for DecisionTreeRegressor<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.depth != other.depth || self.nodes.len() != other.nodes.len() {
            return false;
        } else {
            for i in 0..self.nodes.len() {
                if self.nodes[i] != other.nodes[i] {
                    return false;
                }
            }
            return true;
        }
    }
}

struct NodeVisitor<'a, T: RealNumber, M: Matrix<T>> {
    x: &'a M,
    y: &'a M,
    node: usize,
    samples: Vec<usize>,
    order: &'a Vec<Vec<usize>>,
    true_child_output: T,
    false_child_output: T,
    level: u16,
}

impl<'a, T: RealNumber, M: Matrix<T>> NodeVisitor<'a, T, M> {
    fn new(
        node_id: usize,
        samples: Vec<usize>,
        order: &'a Vec<Vec<usize>>,
        x: &'a M,
        y: &'a M,
        level: u16,
    ) -> Self {
        NodeVisitor {
            x: x,
            y: y,
            node: node_id,
            samples: samples,
            order: order,
            true_child_output: T::zero(),
            false_child_output: T::zero(),
            level: level,
        }
    }
}

impl<T: RealNumber> DecisionTreeRegressor<T> {
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        parameters: DecisionTreeRegressorParameters,
    ) -> DecisionTreeRegressor<T> {
        let (x_nrows, num_attributes) = x.shape();
        let samples = vec![1; x_nrows];
        DecisionTreeRegressor::fit_weak_learner(x, y, samples, num_attributes, parameters)
    }

    pub fn fit_weak_learner<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        samples: Vec<usize>,
        mtry: usize,
        parameters: DecisionTreeRegressorParameters,
    ) -> DecisionTreeRegressor<T> {
        let y_m = M::from_row_vector(y.clone());

        let (_, y_ncols) = y_m.shape();
        let (_, num_attributes) = x.shape();
        let classes = y_m.unique();
        let k = classes.len();
        if k < 2 {
            panic!("Incorrect number of classes: {}. Should be >= 2.", k);
        }

        let mut nodes: Vec<Node<T>> = Vec::new();

        let mut n = 0;
        let mut sum = T::zero();
        for i in 0..y_ncols {
            n += samples[i];
            sum = sum + T::from(samples[i]).unwrap() * y_m.get(0, i);
        }

        let root = Node::new(0, sum / T::from(n).unwrap());
        nodes.push(root);
        let mut order: Vec<Vec<usize>> = Vec::new();

        for i in 0..num_attributes {
            order.push(x.get_col_as_vec(i).quick_argsort_mut());
        }

        let mut tree = DecisionTreeRegressor {
            nodes: nodes,
            parameters: parameters,
            depth: 0,
        };

        let mut visitor = NodeVisitor::<T, M>::new(0, samples, &order, &x, &y_m, 1);

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

    pub fn predict<M: Matrix<T>>(&self, x: &M) -> M::RowVector {
        let mut result = M::zeros(1, x.shape().0);

        let (n, _) = x.shape();

        for i in 0..n {
            result.set(0, i, self.predict_for_row(x, i));
        }

        result.to_row_vector()
    }

    pub(in crate) fn predict_for_row<M: Matrix<T>>(&self, x: &M, row: usize) -> T {
        let mut result = T::zero();
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
        let (_, n_attr) = visitor.x.shape();

        let n: usize = visitor.samples.iter().sum();

        if n < self.parameters.min_samples_split {
            return false;
        }

        let sum = self.nodes[visitor.node].output * T::from(n).unwrap();

        let mut variables = vec![0; n_attr];
        for i in 0..n_attr {
            variables[i] = i;
        }

        let parent_gain =
            T::from(n).unwrap() * self.nodes[visitor.node].output * self.nodes[visitor.node].output;

        for j in 0..mtry {
            self.find_best_split(visitor, n, sum, parent_gain, variables[j]);
        }

        self.nodes[visitor.node].split_score != Option::None
    }

    fn find_best_split<M: Matrix<T>>(
        &mut self,
        visitor: &mut NodeVisitor<T, M>,
        n: usize,
        sum: T,
        parent_gain: T,
        j: usize,
    ) {
        let mut true_sum = T::zero();
        let mut true_count = 0;
        let mut prevx = T::nan();

        for i in visitor.order[j].iter() {
            if visitor.samples[*i] > 0 {
                if prevx.is_nan() || visitor.x.get(*i, j) == prevx {
                    prevx = visitor.x.get(*i, j);
                    true_count += visitor.samples[*i];
                    true_sum =
                        true_sum + T::from(visitor.samples[*i]).unwrap() * visitor.y.get(0, *i);
                    continue;
                }

                let false_count = n - true_count;

                if true_count < self.parameters.min_samples_leaf
                    || false_count < self.parameters.min_samples_leaf
                {
                    prevx = visitor.x.get(*i, j);
                    true_count += visitor.samples[*i];
                    true_sum =
                        true_sum + T::from(visitor.samples[*i]).unwrap() * visitor.y.get(0, *i);
                    continue;
                }

                let true_mean = true_sum / T::from(true_count).unwrap();
                let false_mean = (sum - true_sum) / T::from(false_count).unwrap();

                let gain = (T::from(true_count).unwrap() * true_mean * true_mean
                    + T::from(false_count).unwrap() * false_mean * false_mean)
                    - parent_gain;

                if self.nodes[visitor.node].split_score == Option::None
                    || gain > self.nodes[visitor.node].split_score.unwrap()
                {
                    self.nodes[visitor.node].split_feature = j;
                    self.nodes[visitor.node].split_value =
                        Option::Some((visitor.x.get(*i, j) + prevx) / T::two());
                    self.nodes[visitor.node].split_score = Option::Some(gain);
                    visitor.true_child_output = true_mean;
                    visitor.false_child_output = false_mean;
                }

                prevx = visitor.x.get(*i, j);
                true_sum = true_sum + T::from(visitor.samples[*i]).unwrap() * visitor.y.get(0, *i);
                true_count += visitor.samples[*i];
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
    fn fit_longley() {
        let x = DenseMatrix::from_array(&[
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

        let y_hat = DecisionTreeRegressor::fit(&x, &y, Default::default()).predict(&x);

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
            },
        )
        .predict(&x);

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
            },
        )
        .predict(&x);

        for i in 0..y_hat.len() {
            assert!((y_hat[i] - expected_y[i]).abs() < 0.1);
        }
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::from_array(&[
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

        let tree = DecisionTreeRegressor::fit(&x, &y, Default::default());

        let deserialized_tree: DecisionTreeRegressor<f64> =
            bincode::deserialize(&bincode::serialize(&tree).unwrap()).unwrap();

        assert_eq!(tree, deserialized_tree);
    }
}
