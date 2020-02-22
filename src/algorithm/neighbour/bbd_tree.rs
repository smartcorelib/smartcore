use crate::linalg::Matrix;
use crate::math::distance::euclidian;

#[derive(Debug)]
pub struct BBDTree {    
    nodes: Vec<BBDTreeNode>,
    index: Vec<usize>,
    root: usize
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
    upper: Option<usize>
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
            upper: Option::None
        }
    }
}

impl BBDTree {
    pub fn new<M: Matrix>(data: &M) -> BBDTree {
        let nodes = Vec::new();

        let (n, _) = data.shape();

        let mut index = vec![0; n];
        for i in 0..n {
            index[i] = i;
        }

        let mut tree = BBDTree{
            nodes: nodes,
            index: index,
            root: 0
        };

        let root = tree.build_node(data, 0, n);

        tree.root = root;

        tree
    }    

    pub(in crate) fn clustering(&self, centroids: &Vec<Vec<f64>>, sums: &mut Vec<Vec<f64>>, counts: &mut Vec<usize>, membership: &mut Vec<usize>) -> f64 {
        let k = centroids.len();

        counts.iter_mut().for_each(|x| *x = 0);
        let mut candidates = vec![0; k];
        for i in 0..k {
            candidates[i] = i;
            sums[i].iter_mut().for_each(|x| *x = 0f64);            
        }
                
        self.filter(self.root, centroids, &candidates, k, sums, counts, membership)
    }

    fn filter(&self, node: usize, centroids: &Vec<Vec<f64>>, candidates: &Vec<usize>, k: usize, sums: &mut Vec<Vec<f64>>, counts: &mut Vec<usize>, membership: &mut Vec<usize>) -> f64{
        let d = centroids[0].len();

        // Determine which mean the node mean is closest to
        let mut min_dist = euclidian::squared_distance(&self.nodes[node].center, &centroids[candidates[0]]);
        let mut closest = candidates[0];
        for i in 1..k {
            let dist = euclidian::squared_distance(&self.nodes[node].center, &centroids[candidates[i]]);
            if dist < min_dist {
                min_dist = dist;
                closest = candidates[i];
            }
        }

        // If this is a non-leaf node, recurse if necessary
        if !self.nodes[node].lower.is_none() {
            // Build the new list of candidates
            let mut new_candidates = vec![0;k];
            let mut newk = 0;

            for i in 0..k {
                if !BBDTree::prune(&self.nodes[node].center, &self.nodes[node].radius, &centroids, closest, candidates[i]) {
                    new_candidates[newk] = candidates[i];
                    newk += 1;
                }
            }

            // Recurse if there's at least two
            if newk > 1 {
                let result = self.filter(self.nodes[node].lower.unwrap(), centroids, &mut new_candidates, newk, sums, counts, membership) + 
                            self.filter(self.nodes[node].upper.unwrap(), centroids, &mut new_candidates, newk, sums, counts, membership);
                return result;
            }
        }

        // Assigns all data within this node to a single mean
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

    fn prune(center: &Vec<f64>, radius: &Vec<f64>, centroids: &Vec<Vec<f64>>, best_index: usize, test_index: usize) -> bool {
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

        return lhs >= 2f64 * rhs;
    }    

    fn build_node<M: Matrix>(&mut self, data: &M, begin: usize, end: usize) -> usize {
        let (_, d) = data.shape();

        // Allocate the node
        let mut node = BBDTreeNode::new(d);

        // Fill in basic info
        node.count = end - begin;
        node.index = begin;

        // Calculate the bounding box
        let mut lower_bound = vec![0f64; d];
        let mut upper_bound = vec![0f64; d];

        for i in 0..d {
            lower_bound[i] = data.get(self.index[begin],i);
            upper_bound[i] = data.get(self.index[begin],i);
        }

        for i in begin..end {
            for j in 0..d {
                let c = data.get(self.index[i], j);
                if lower_bound[j] > c {
                    lower_bound[j] = c;
                }
                if upper_bound[j] < c {
                    upper_bound[j] = c;
                }
            }
        }

        // Calculate bounding box stats
        let mut max_radius = -1.;
        let mut split_index = 0;
        for i in 0..d {
            node.center[i] = (lower_bound[i] + upper_bound[i]) / 2.;
            node.radius[i] = (upper_bound[i] - lower_bound[i]) / 2.;
            if node.radius[i] > max_radius {
                max_radius = node.radius[i];
                split_index = i;
            }
        }

        // If the max spread is 0, make this a leaf node
        if max_radius < 1E-10 {
            node.lower = Option::None;
            node.upper = Option::None;
            for i in 0..d {
                node.sum[i] = data.get(self.index[begin], i);
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

        // Partition the data around the midpoint in this dimension. The
        // partitioning is done in-place by iterating from left-to-right and
        // right-to-left in the same way that partioning is done in quicksort.
        let split_cutoff = node.center[split_index];
        let mut i1 = begin;
        let mut i2 = end - 1;
        let mut size = 0;
        while i1 <= i2 {
            let mut i1_good = data.get(self.index[i1], split_index) < split_cutoff;
            let mut i2_good = data.get(self.index[i2], split_index) >= split_cutoff;

            if !i1_good && !i2_good {
                let temp = self.index[i1];
                self.index[i1] = self.index[i2];
                self.index[i2] = temp;
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

        // Create the child nodes
        node.lower = Option::Some(self.build_node(data, begin, begin + size));
        node.upper = Option::Some(self.build_node(data, begin + size, end));

        // Calculate the new sum and opt cost
        for i in 0..d {
            node.sum[i] = self.nodes[node.lower.unwrap()].sum[i] + self.nodes[node.upper.unwrap()].sum[i];
        }

        let mut mean = vec![0f64; d];
        for i in 0..d {
            mean[i] = node.sum[i] / node.count as f64;
        }

        node.cost = BBDTree::node_cost(&self.nodes[node.lower.unwrap()], &mean) + BBDTree::node_cost(&self.nodes[node.upper.unwrap()], &mean);

        self.add_node(node)
    }

    fn node_cost(node: &BBDTreeNode, center: &Vec<f64>) -> f64 {
        let d = center.len();
        let mut scatter = 0f64;
        for i in 0..d {
            let x = (node.sum[i] / node.count as f64) - center[i];
            scatter += x * x;
        }
        node.cost + node.count as f64 * scatter
    }

    fn add_node(&mut self, new_node: BBDTreeNode) -> usize{
        let idx = self.nodes.len();
        self.nodes.push(new_node);
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*; 
    use crate::linalg::naive::dense_matrix::DenseMatrix;    

    #[test]
    fn fit_predict_iris() {             

        let data = DenseMatrix::from_array(&[
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
            &[5.2, 2.7, 3.9, 1.4]]);        

        let tree = BBDTree::new(&data);        

        let centroids = vec![
            vec![4.86, 3.22, 1.61, 0.29],
            vec![6.23, 2.92, 4.48, 1.42]
        ];

        let mut sums = vec![
            vec![0f64; 4],
            vec![0f64; 4]
        ];

        let mut counts = vec![11, 9];

        let mut membership = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1];

        let dist = tree.clustering(&centroids, &mut sums, &mut counts, &mut membership);        
        assert!((dist - 10.68).abs() < 1e-2);
        assert!((sums[0][0] - 48.6).abs() < 1e-2);
        assert!((sums[1][3] - 13.8).abs() < 1e-2);
        assert_eq!(membership[17], 1);        
            
    }

}