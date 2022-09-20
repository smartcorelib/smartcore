use std::fmt::Debug;

use crate::linalg::base::Array2;
use crate::metrics::distance::euclidian::*;
use crate::num::Number;

#[derive(Debug)]
pub struct BBDTree {
    nodes: Vec<BBDTreeNode>,
    index: Vec<usize>,
    root: usize,
}

#[derive(Debug)]
struct BBDTreeNode {
    count: usize,
    index: usize,
    center: Vec<f64>,
    radius: Vec<f64>,
    sum: Vec<f64>,
    cost: f64,
    lower: Option<usize>,
    upper: Option<usize>,
}

impl BBDTreeNode {
    fn new(d: usize) -> BBDTreeNode {
        BBDTreeNode {
            count: 0,
            index: 0,
            center: vec![0f64; d],
            radius: vec![0f64; d],
            sum: vec![0f64; d],
            cost: 0f64,
            lower: Option::None,
            upper: Option::None,
        }
    }
}

impl BBDTree {
    pub fn new<T: Number, M: Array2<T>>(data: &M) -> BBDTree {
        let nodes = Vec::new();

        let (n, _) = data.shape();

        let index = (0..n).collect::<Vec<_>>();

        let mut tree = BBDTree {
            nodes,
            index,
            root: 0,
        };

        let root = tree.build_node(data, 0, n);

        tree.root = root;

        tree
    }

    pub(crate) fn clustering(
        &self,
        centroids: &[Vec<f64>],
        sums: &mut Vec<Vec<f64>>,
        counts: &mut Vec<usize>,
        membership: &mut Vec<usize>,
    ) -> f64 {
        let k = centroids.len();

        counts.iter_mut().for_each(|v| *v = 0);
        let mut candidates = vec![0; k];
        for i in 0..k {
            candidates[i] = i;
            sums[i].iter_mut().for_each(|v| *v = 0f64);
        }

        self.filter(
            self.root,
            centroids,
            &candidates,
            k,
            sums,
            counts,
            membership,
        )
    }

    fn filter(
        &self,
        node: usize,
        centroids: &[Vec<f64>],
        candidates: &[usize],
        k: usize,
        sums: &mut Vec<Vec<f64>>,
        counts: &mut Vec<usize>,
        membership: &mut Vec<usize>,
    ) -> f64 {
        let d = centroids[0].len();

        let mut min_dist =
            Euclidian::squared_distance(&self.nodes[node].center, &centroids[candidates[0]]);
        let mut closest = candidates[0];
        for i in 1..k {
            let dist =
                Euclidian::squared_distance(&self.nodes[node].center, &centroids[candidates[i]]);
            if dist < min_dist {
                min_dist = dist;
                closest = candidates[i];
            }
        }

        if self.nodes[node].lower.is_some() {
            let mut new_candidates = vec![0; k];
            let mut newk = 0;

            for candidate in candidates.iter().take(k) {
                if !BBDTree::prune(
                    &self.nodes[node].center,
                    &self.nodes[node].radius,
                    centroids,
                    closest,
                    *candidate,
                ) {
                    new_candidates[newk] = *candidate;
                    newk += 1;
                }
            }

            if newk > 1 {
                return self.filter(
                    self.nodes[node].lower.unwrap(),
                    centroids,
                    &new_candidates,
                    newk,
                    sums,
                    counts,
                    membership,
                ) + self.filter(
                    self.nodes[node].upper.unwrap(),
                    centroids,
                    &new_candidates,
                    newk,
                    sums,
                    counts,
                    membership,
                );
            }
        }

        for i in 0..d {
            sums[closest][i] += self.nodes[node].sum[i];
        }

        counts[closest] += self.nodes[node].count;

        let last = self.nodes[node].index + self.nodes[node].count;
        for i in self.nodes[node].index..last {
            membership[self.index[i]] = closest;
        }

        BBDTree::node_cost(&self.nodes[node], &centroids[closest])
    }

    fn prune(
        center: &[f64],
        radius: &[f64],
        centroids: &[Vec<f64>],
        best_index: usize,
        test_index: usize,
    ) -> bool {
        if best_index == test_index {
            return false;
        }

        let d = centroids[0].len();

        let best = &centroids[best_index];
        let test = &centroids[test_index];
        let mut lhs = 0f64;
        let mut rhs = 0f64;
        for i in 0..d {
            let diff = test[i] - best[i];
            lhs += diff * diff;
            if diff > 0f64 {
                rhs += (center[i] + radius[i] - best[i]) * diff;
            } else {
                rhs += (center[i] - radius[i] - best[i]) * diff;
            }
        }

        lhs >= 2f64 * rhs
    }

    fn build_node<T: Number, M: Array2<T>>(&mut self, data: &M, begin: usize, end: usize) -> usize {
        let (_, d) = data.shape();

        let mut node = BBDTreeNode::new(d);

        node.count = end - begin;
        node.index = begin;

        let mut lower_bound = vec![0f64; d];
        let mut upper_bound = vec![0f64; d];

        for i in 0..d {
            lower_bound[i] = data.get((self.index[begin], i)).to_f64().unwrap();
            upper_bound[i] = data.get((self.index[begin], i)).to_f64().unwrap();
        }

        for i in begin..end {
            for j in 0..d {
                let c = data.get((self.index[i], j)).to_f64().unwrap();
                if lower_bound[j] > c {
                    lower_bound[j] = c;
                }
                if upper_bound[j] < c {
                    upper_bound[j] = c;
                }
            }
        }

        let mut max_radius = -1f64;
        let mut split_index = 0;
        for i in 0..d {
            node.center[i] = (lower_bound[i] + upper_bound[i]) / 2f64;
            node.radius[i] = (upper_bound[i] - lower_bound[i]) / 2f64;
            if node.radius[i] > max_radius {
                max_radius = node.radius[i];
                split_index = i;
            }
        }

        if max_radius < 1E-10 {
            node.lower = Option::None;
            node.upper = Option::None;
            for i in 0..d {
                node.sum[i] = data.get((self.index[begin], i)).to_f64().unwrap();
            }

            if end > begin + 1 {
                let len = end - begin;
                for i in 0..d {
                    node.sum[i] *= len as f64;
                }
            }

            node.cost = 0f64;
            return self.add_node(node);
        }

        let split_cutoff = node.center[split_index];
        let mut i1 = begin;
        let mut i2 = end - 1;
        let mut size = 0;
        while i1 <= i2 {
            let mut i1_good =
                data.get((self.index[i1], split_index)).to_f64().unwrap() < split_cutoff;
            let mut i2_good =
                data.get((self.index[i2], split_index)).to_f64().unwrap() >= split_cutoff;

            if !i1_good && !i2_good {
                self.index.swap(i1, i2);
                i1_good = true;
                i2_good = true;
            }

            if i1_good {
                i1 += 1;
                size += 1;
            }

            if i2_good {
                i2 -= 1;
            }
        }

        node.lower = Option::Some(self.build_node(data, begin, begin + size));
        node.upper = Option::Some(self.build_node(data, begin + size, end));

        for i in 0..d {
            node.sum[i] =
                self.nodes[node.lower.unwrap()].sum[i] + self.nodes[node.upper.unwrap()].sum[i];
        }

        let mut mean = vec![0f64; d];
        for (i, mean_i) in mean.iter_mut().enumerate().take(d) {
            *mean_i = node.sum[i] / node.count as f64;
        }

        node.cost = BBDTree::node_cost(&self.nodes[node.lower.unwrap()], &mean)
            + BBDTree::node_cost(&self.nodes[node.upper.unwrap()], &mean);

        self.add_node(node)
    }

    fn node_cost(node: &BBDTreeNode, center: &[f64]) -> f64 {
        let d = center.len();
        let mut scatter = 0f64;
        for (i, center_i) in center.iter().enumerate().take(d) {
            let x = (node.sum[i] / node.count as f64) - *center_i;
            scatter += x * x;
        }
        node.cost + node.count as f64 * scatter
    }

    fn add_node(&mut self, new_node: BBDTreeNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(new_node);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::dense::matrix::DenseMatrix;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bbdtree_iris() {
        let data = DenseMatrix::from_2d_array(&[
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

        let tree = BBDTree::new(&data);

        let centroids = vec![vec![4.86, 3.22, 1.61, 0.29], vec![6.23, 2.92, 4.48, 1.42]];

        let mut sums = vec![vec![0f64; 4], vec![0f64; 4]];

        let mut counts = vec![11, 9];

        let mut membership = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1];

        let dist = tree.clustering(&centroids, &mut sums, &mut counts, &mut membership);
        assert!((dist - 10.68).abs() < 1e-2);
        assert!((sums[0][0] - 48.6).abs() < 1e-2);
        assert!((sums[1][3] - 13.8).abs() < 1e-2);
        assert_eq!(membership[17], 1);
    }
}
