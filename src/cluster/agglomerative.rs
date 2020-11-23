#![allow(non_snake_case)]
#![allow(unused_imports)]
//! #  Agglomerative hierarchical clustering
//!
//! ## Definition
//! "SAHN" (sequential, agglomerative, hierarchic, nonoverlapping methods) as defined
//!  **by Müllner, 2011** in <https://arxiv.org/pdf/1109.2378.pdf>
//!
//! > Agglomerative clustering schemes start from the partition of
//! > the data set into singleton nodes and merge step by step the current pair of mutually closest
//! > nodes into a new node until there is one final node left, which comprises the entire data set.
//! > Various clustering schemes share this procedure as a common definition, but differ in the way
//! > in which the measure of inter-cluster dissimilarity is updated after each step. The seven most
//! > common methods are termed single, complete, average (UPGMA), weighted (WPGMA, McQuitty), Ward,
//! > centroid (UPGMC) and median (WPGMC) linkage.
//!
//! Or in addition also "Hierarchical Clustering" as defined **by Eppstein, 2000**
//!  in <https://www.ics.uci.edu/~eppstein/projects/pairs/Talks/ClusterGroup.pdf>.
//!
//! ## Algorithms:
//!
//! ### Mentioned by Müllner, 2011
//! Algorithms specified and tested by
//! > The specific class of clustering algorithms which is dealt with in this paper has been
//! > characterized by the acronym SAHN (sequential, agglomerative, hierarchic, nonoverlapping methods)"
//!
//! * (availalbe in ver. 0.2) `PRIMITIVE_CLUSTERING`, as in Fig.1 in Müllner, 2011
//! * `GENERIC_LINKAGE` Anderberg, 1973 and later improvements
//! * `NN_CHAIN_LINKAGE` Murtagh, 1985 and later improvements
//! * (to be implemented) MST-linkage` (The single linkage algorithm, aka fastcluster), as in Fig.6 in Müllner, 2011
//!
//! `MST-linkage` is an implmenetation of `MST-linkage-core` plus two post-processing steps:
//! * Sort by distance
//! * LABEL (aka union-find), as in Fig.5 in Müllner, 2011
//!
//! ### Mentioned by Eppstein, 2000
//! More generic algorithms for **Closest Pair Data Structures**.
//! As listed in <https://www.ics.uci.edu/~eppstein/projects/pairs/Methods/>. among others:
//!
//! > Conga line. We partition the objects into O(log n) subsets and maintain a graph in each subset, such
//! > that the closest pair is guaranteed to correspond to an edge in the graph. Each insertion creates a
//! > new subset for the new object; each deletion may move an object from each existing subset to a new subset.
//! > In each case, if necessary some pair of subsets is merged to maintain the desired number of subsets.
//! > Amortized time per insertion is O(Q log n); amortized time per deletion is O(Q log2 n). Space is linear.
//!
//! > FastPair. We further simplify conga lines by making separate singleton subsets for the objects moved to
//! > new subsets by a deletion. This can alternately be viewed as a modification to the neighbor heuristic, in
//! > which the initial construction of all nearest neighbors is replaced by a conga line computation, and in
//! > which each insertion does not update previously computed neighbors. Its time both theoretically and in practice
//! > is qualitatively similar to the neighbor heuristic, but it typically runs 30% or so faster.
//!
//! ## Interface
//! Reference:
//!     >>> from sklearn.cluster import AgglomerativeClustering
//!     >>> import numpy as np
//!     >>> X = np.array([[1, 2], [1, 4], [1, 0],
//!     ...               [4, 2], [4, 4], [4, 0]])
//!     >>> clustering = AgglomerativeClustering().fit(X)
//!     >>> clustering
//!     AgglomerativeClustering()
//!     >>> clustering.labels_
//!     array([1, 1, 1, 0, 0, 0])
//!
//! ## Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::math::num::RealNumber;
//! use smartcore::cluster::agglomerative::*;
//! use smartcore::algorithm::neighbour::fastpair::FastPair;
//!
//! // Iris data
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
//!            ]);
//!
//! // Fit to data, with a threshold
//! // example using FastPair, compute dendrogram
//! let clustering = AggregativeFastPair::fit(&x).unwrap();
//! let dendrogram = clustering.labels();
//! ```
//!
use std::marker::PhantomData;

use crate::algorithm::neighbour::fastpair::FastPair;
use crate::error::Failed;
use crate::linalg::Matrix;
use crate::math::distance::euclidian::Euclidian;
use crate::math::num::RealNumber;

///
/// Abstract trait for sequential, agglomerative, hierarchic, non-overlapping methods
///
pub trait SAHNClustering<T: RealNumber, M: Matrix<T>> {
    ///
    /// Aggregate the data according to given distance threshold
    ///
    fn fit(data: &M) -> Result<AggregativeFastPair<T, M>, Failed>;

    ///
    /// Return clusters labels
    ///
    fn labels(&self) -> &Vec<Vec<T>>;
}

///
/// An implementation of Bottom-Up (Agglomerative) Hierarchical
///  Clustering with `FastPair`
///
pub struct AggregativeFastPair<T: RealNumber, M: Matrix<T>> {
    dendrogram: Vec<Vec<T>>, // computed labels
    _marker: PhantomData<M>,
}

// Add linkage algorithms
impl<T: RealNumber, M: Matrix<T>> FastCluster<T> for AggregativeFastPair<T, M> {}

// Implement aggregative clustering using FastPair and fastcluster
impl<T: RealNumber, M: Matrix<T>> SAHNClustering<T, M> for AggregativeFastPair<T, M> {
    //
    // 1. Compute `FastPair` on matrix's rows
    // 2. Compute sparse dissimilarities into upper-trinagular matrix
    // 3. Compute dendrogram
    //
    // The linkage distance threshold above which clusters will not be merged.
    fn fit(data: &M) -> Result<AggregativeFastPair<T, M>, Failed> {
        let fastpair = FastPair(data).unwrap();

        // compute full connectivity from sparse matrix
        let full_connectivity: M =
            AggregativeFastPair::<T, M>::condensed_matrix(fastpair.connectivity.unwrap(), data);

        // compute clusters
        let dendrogram: Vec<Vec<T>> =
            AggregativeFastPair::<T, M>::mst_single_linkage(full_connectivity, data.shape().0);

        let n: usize = dendrogram.len();

        Ok(AggregativeFastPair {
            // assign computed dendrogram to structure
            dendrogram: AggregativeFastPair::<T, M>::label(dendrogram, n),
            _marker: PhantomData,
        })
    }

    fn labels(&self) -> &Vec<Vec<T>> {
        &self.dendrogram
    }
}

///
/// Abstract trait for FastCluster (MST-Linkage and post-processing: Union-Find, labels)
///   Müllner, 2011 in Fig. 6 <https://arxiv.org/pdf/1109.2378.pdf>
///
pub trait FastCluster<T: RealNumber> {
    ///
    /// Return distances condensed matrix
    ///  "which is the upper triangle (without the diagonal elements) of the full distance matrix"
    ///  <https://lionel.kr.hs-niederrhein.de/~dalitz/data/hclust/>
    ///
    /// Closest pairs dissimilarity structure is a sparse matrix, return full connectivity matrix
    fn condensed_matrix<M: Matrix<T>>(sparse_matrix: Box<M>, samples: &M) -> M {
        let len = samples.shape().0;
        let mut full_connectivity: M = *(sparse_matrix).clone();

        for i in 0..len {
            for j in 0..len {
                if full_connectivity.get(i, j) == T::zero() {
                    full_connectivity.set(
                        i,
                        j,
                        Euclidian::squared_distance(
                            &samples.get_row_as_vec(i),
                            &samples.get_row_as_vec(j),
                        ),
                    );
                }
            }
        }
        full_connectivity
    }

    /// Perform linkage using MST-Linkage (fastcluster)
    /// scipy: https://github.com/scipy/scipy/blob/d286f8525c16b2cd4e179dea2c77b6b09622aff9/scipy/cluster/_hierarchy.pyx#L1016
    /// MST_linkage_core https://github.com/cdalitz/hclust-cpp/blob/dc68e86cda36aea724ba19cae2f645cedfb65ce6/fastcluster_dm.cpp#L395
    ///
    /// Parameters
    /// ----------
    /// full_connectivity :
    ///    A condensed matrix stores the pairwise distances of the observations.
    /// n :
    ///    The number of observations.
    /// Returns
    /// -------
    /// Z : shape (n - 1, 4)
    ///     Computed linkage matrix.
    fn mst_single_linkage<M: Matrix<T>>(full_connectivity: M, n: usize) -> Vec<Vec<T>> {
        let mut Z = vec![vec![T::zero(); 4]; n - 1];

        // Which nodes were already merged.
        let mut merged = vec![-1; n];

        let mut D = vec![T::max_value(); n];

        let (mut x, mut y, mut dist, mut current_min): (usize, usize, T, T);

        x = 0;
        y = 0;
        for k in 0..(n - 1) {
            current_min = T::max_value();
            merged[x] = 1;
            for i in 0..n {
                if merged[i] == 1 {
                    continue;
                }

                dist = full_connectivity.get(x, i);
                if D[i] > dist {
                    D[i] = dist;
                }

                if D[i] < current_min {
                    y = i;
                    current_min = D[i];
                }
            }
            Z[k][0] = T::from(x).unwrap();
            Z[k][1] = T::from(y).unwrap();
            Z[k][2] = current_min;
            x = y;
        }
        // Z is now an unsorted dendrogram

        // Sort Z by cluster distances.
        Z.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
        Z
    }

    ///
    /// Correctly label clusters in unsorted dendrogram.
    ///
    fn label(mut Z: Vec<Vec<T>>, n: usize) -> Vec<Vec<T>> {
        let mut uf: LinkageUnionFind = LinkageUnionFind(n);
        let (mut x, mut y, mut x_root, mut y_root): (T, T, usize, usize);

        for i in 0..(n - 1) {
            x = Z[i][0];
            y = Z[i][1];
            let x_tmp: usize = T::to_usize(&x).unwrap();
            let y_tmp: usize = T::to_usize(&y).unwrap();
            x_root = uf.find(x_tmp);
            y_root = uf.find(y_tmp);
            if x_root < y_root {
                Z[i][0] = T::from(x_root).unwrap();
                Z[i][1] = T::from(y_root).unwrap();
            } else {
                Z[i][0] = T::from(y_root).unwrap();
                Z[i][1] = T::from(x_root).unwrap();
            }
            let merged = uf.merge(x_root, y_root);
            Z[i][3] = T::from(merged).unwrap();
        }
        Z
    }
}

///
/// fastcluster post-processing linkage
///
fn LinkageUnionFind(n: usize) -> LinkageUnionFind {
    LinkageUnionFind {
        parent: (0..(2 * n - 1)).collect(),
        next_label: n,
        size: vec![1; 2 * n - 1],
    }
}

struct LinkageUnionFind {
    parent: Vec<usize>,
    next_label: usize,
    size: Vec<usize>,
}

impl LinkageUnionFind {
    fn merge(&mut self, x: usize, y: usize) -> usize {
        self.parent[x] = self.next_label;
        self.parent[y] = self.next_label;
        let size: usize = self.size[x] + self.size[y];
        self.size[self.next_label] = size;
        self.next_label += 1;
        size
    }

    fn find(&mut self, x: usize) -> usize {
        let mut x: usize = x;
        let mut p: usize = x;

        while self.parent[x] != x {
            x = self.parent[x];
        }

        while self.parent[p] != x {
            p = self.parent[p];
            self.parent[p] = x;
        }

        x
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn aggregative_fit() {
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
        let cluster = AggregativeFastPair::fit(&x).unwrap();
        let labels = cluster.labels();

        let expected = [
            //  index, index, distance, label
            [0.0, 4.0, 0.01999999999999995, 2.0],
            [7.0, 19.0, 0.029999999999999964, 3.0],
            [1.0, 9.0, 0.030000000000000037, 2.0],
            [14.0, 18.0, 0.05999999999999993, 2.0],
            [2.0, 3.0, 0.0600000000000001, 2.0],
            [6.0, 23.0, 0.06999999999999998, 3.0],
            [10.0, 12.0, 0.07000000000000003, 2.0],
            [11.0, 16.0, 0.07000000000000013, 2.0],
            [8.0, 24.0, 0.08999999999999982, 4.0],
            [21.0, 27.0, 0.09000000000000011, 6.0],
            [20.0, 28.0, 0.10999999999999982, 9.0],
            [22.0, 26.0, 0.1799999999999998, 4.0],
            [25.0, 30.0, 0.2600000000000009, 6.0],
            [13.0, 29.0, 0.27000000000000013, 10.0],
            [5.0, 32.0, 0.3800000000000002, 11.0],
            [15.0, 33.0, 0.54, 12.0],
            [31.0, 34.0, 0.6899999999999995, 18.0],
            [17.0, 35.0, 0.7000000000000001, 19.0],
            [5.0, 17.0, 4.460000000000001, 0.0],
        ];

        let mut i = 0;
        for v in expected.iter() {
            assert_eq!(labels[i], v);
            i += 1;
        }
    }
}
