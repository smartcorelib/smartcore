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
//! // example using FastPair
//! let cluster = AggregativeFastPair::fit(&x, 1.5).unwrap();
//! // return results/labels/dendrogram
//! let labels = cluster.labels();
//! ```
//!
use std::collections::{HashMap, LinkedList};

use crate::algorithm::neighbour::dissimilarities::PairwiseDissimilarity;
use crate::algorithm::neighbour::fastpair::{FastPair, _FastPair};
use crate::error::{Failed, FailedError};
use crate::linalg::{BaseMatrix, Matrix};
use crate::linalg::naive::dense_matrix::DenseMatrix;
use crate::math::num::RealNumber;
use crate::math::distance::euclidian::Euclidian;

///
/// Abstract trait for sequential, agglomerative, hierarchic, non-overlapping methods
///
pub trait SAHNClustering<T: RealNumber, M: Matrix<T>> {
    //
    // Aggregate the data according to given distance threshold
    //
    fn fit(data: &M, threshold: T) -> Result<AggregativeFastPair<T, M>, Failed>;

    //
    // Return clusters, assign labels to dissimilarities, according
    // to threshold.
    //
    fn labels(&self) -> &Box<Vec<usize>>;
}


///
/// An implementation of Bottom-Up (Agglomerative) Hierarchical
///  Clustering with `FastPair`
///
pub struct AggregativeFastPair<T: RealNumber, M: Matrix<T>> {
    labels: Box<Vec<usize>>,
    dendrogram: Box<M>,
    current: Option<T>
}

///
/// Return distances condensed matrix
///  "which is the upper triangle (without the diagonal elements) of the full distance matrix"
///  <https://lionel.kr.hs-niederrhein.de/~dalitz/data/hclust/>
///
/// Closest pairs dissimilarity structure is a sparse matrix, return full connectivity matrix 
fn condensed_matrix<T: RealNumber, M: Matrix<T>>(sparse_matrix: Box<M>, samples: &M) -> M {
    let len = samples.shape().0;
    let mut full_connectivity: M = *(sparse_matrix).clone();

    for i in 0..len {
        for j in 0..len {
            if full_connectivity.get(i, j) == T::zero() { 
                full_connectivity.set(
                    i, j,
                    Euclidian::squared_distance(
                        &samples.get_row_as_vec(i),
                        &samples.get_row_as_vec(j)
                    ),
                );
            }
        }
    }
    full_connectivity
}

// Add linkage algorithms
impl<T: RealNumber, M: Matrix<T>> FastCluster<T> for AggregativeFastPair<T, M> {}

// 
impl<T: RealNumber, M: Matrix<T>> SAHNClustering<T, M> for AggregativeFastPair<T, M> {
    //
    // 1. Compute `FastPair` on matrix's rows
    // 2. Port dissimilarities into upper-trinagular matrix
    //
    // The linkage distance threshold above which clusters will not be merged.
    fn fit(data: &M, threshold: T) -> Result<AggregativeFastPair<T, M>, Failed> {
        let fastpair = FastPair(data).unwrap();
        
        // compute full connectivity from sparse matrix
        let full_connectivity = condensed_matrix(fastpair.connectivity.unwrap(), data);

        // compute clusters

        let labels: Box<Vec<usize>> = Box::new(vec![0]);
        Ok(AggregativeFastPair { 
            labels: labels,
            dendrogram: Box::new(full_connectivity),
            current: None
        })
    }

    fn labels(&self) -> &Box<Vec<usize>> {
        &self.labels
    }

}


/// 
/// Abstract trait for FastCluster (MST-Linkage and post-processing: Union-Find, labels)
///   Müllner, 2011 in Fig. 6 <https://arxiv.org/pdf/1109.2378.pdf>
/// 
pub trait FastCluster<T: RealNumber> {
    // Perform hierarchy clustering using MST-Linkage (fastcluster)
    // scipy: https://github.com/scipy/scipy/blob/d286f8525c16b2cd4e179dea2c77b6b09622aff9/scipy/cluster/_hierarchy.pyx#L1016
    // MST_linkage_core https://github.com/cdalitz/hclust-cpp/blob/dc68e86cda36aea724ba19cae2f645cedfb65ce6/fastcluster_dm.cpp#L395
    // 
    // Parameters
    // ----------
    // dists : ndarray
    //    A condensed matrix stores the pairwise distances of the observations.
    // n : int
    //    The number of observations.
    // Returns
    // -------
    // Z : ndarray, shape (n - 1, 4)
    //     Computed linkage matrix.
    fn mst_single_linkage<M: Matrix<T>>(full_connectivity: M, n: usize) -> Option<M> {    
        // cdef class LinkageUnionFind:
        //     """Structure for fast cluster labeling in unsorted dendrogram."""
        //     cdef int[:] parent
        //     cdef int[:] size
        //     cdef int next_label

        //     def __init__(self, int n):
        //         self.parent = np.arange(2 * n - 1, dtype=np.intc)
        //         self.next_label = n
        //         self.size = np.ones(2 * n - 1, dtype=np.intc)

        //     cdef int merge(self, int x, int y):
        //         self.parent[x] = self.next_label
        //         self.parent[y] = self.next_label
        //         cdef int size = self.size[x] + self.size[y]
        //         self.size[self.next_label] = size
        //         self.next_label += 1
        //         return size

        //     cdef find(self, int x):
        //         cdef int p = x

        //         while self.parent[x] != x:
        //             x = self.parent[x]

        //         while self.parent[p] != x:
        //             p, self.parent[p] = self.parent[p], x

        //         return x

        // cdef label(double[:, :] Z, int n):
        //     """Correctly label clusters in unsorted dendrogram."""
        //     cdef LinkageUnionFind uf = LinkageUnionFind(n)
        //     cdef int i, x, y, x_root, y_root

        //     for i in range(n - 1):
        //         x, y = int(Z[i, 0]), int(Z[i, 1])
        //         x_root, y_root = uf.find(x), uf.find(y)
        //         if x_root < y_root:
        //             Z[i, 0], Z[i, 1] = x_root, y_root
        //         else:
        //             Z[i, 0], Z[i, 1] = y_root, x_root
        //         Z[i, 3] = uf.merge(x_root, y_root)
        let mut Z = M::zeros((n-1), 4);

        // Which nodes were already merged.
        let mut merged = Vec::with_capacity(n);

        let mut D = vec![T::max_value(); n];

        let (mut i, mut k, mut x, mut y, mut dist, mut current_min): (usize, usize, usize, usize, T, T);

        x = 0;
        y = 0;
        for k in 0..(n - 1) {
            current_min = T::max_value();
            merged[x] = 1;
            for i in 0..n {
                if (merged[i] == 1) {
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
            Z.set(k, 0, T::from(x).unwrap());
            Z.set(k, 1, T::from(y).unwrap());
            Z.set(k, 2, current_min);
            x = y;
        }
        // Z is now a stepwise dendrogram

        // # Sort Z by cluster distances.
        // order = np.argsort(Z_arr[:, 2], kind='mergesort')
        // Z_arr = Z_arr[order]

        // # Find correct cluster labels and compute cluster sizes inplace.
        // label(Z_arr, n)

        Some(Z)
    }
}


// impl<T: RealNumber> FastCluster<T> for AggregativeFast<T> {
//     fn fit<M: Matrix<T>>(data: &M, threshold: T) -> Result<AggregativeFast<T>, Failed> {
//         let fastpair = FastPair(data).unwrap();
//         // fastpair.distances  -- connectivity matrix

//         // linkage? single, ward, complete and average
//         // mst linkage works only with single

//         // compute labels
//         let dendrogram = mst_single_linkage(fastpair.distances, data.shape().0);
//         let labels: Box<Vec<usize>> = Box::new(vec![0]);

//         Ok(AggregativeFast { 
//             labels: labels,
//             dendrogram: None,
//         })
//     }

//     fn labels(&self) -> &Box<Vec<usize>> {
        
//  }

// ///
// /// Struct (dendrogram) to hold result of linkage and clustering
// ///
// /// > The term 'dendrogram' has been used with three different
// /// > meanings: a mathematical object, a data structure and
// /// > a graphical representation of the former two. In the course of this section, we
// /// > define a data structure and call it 'stepwise dendrogram'.
// ///
// /// Use `std::collections::LinkedList`
// pub struct Dendrogram<T: RealNumber> {
//     Z: LinkedList<Box<PairwiseDissimilarity<T>>>, // list of nodes in clustering process
//     current: Option<usize>,                       // used to read as a doubly linked list
// }

// // impl ClusterLabels<T: RealNumber> {
// //     pub fn new(&self, size: usize) -> Self {
// //         let mut z = LinkedList::with_capacity(size);
// //         Self {
// //             Z: z,
// //             current: None,
// //         }
// //     }
// //     // add a node to the dendrogram
// //     pub fn append(node1: PairwiseDissimilarity, node2: PairwiseDissimilarity, dist: T) -> () {
// //         let idx = Z.len();
// //         let node: Box<PairwiseDissimilarity> = Box::new(PairwiseDissimilarity {
// //             node1: node1,
// //             node2: node2,
// //             distance: dist,
// //             position: idx,
// //         });
// //         self.Z.push(node);
// //     }

// //     //
// //     // Return list of labels according to number of desired clusters
// //     //
// //     pub fn labels_by_k(k: usize) {}

// //     //
// //     // Return list of labels by distance threshold
// //     //
// //     pub fn labels_by_threshold(threshold: T) {}

// //     // Methods for distances post-processing.
// //     // All of those have to be monotone or the ordering will change
// //     pub fn sqrt() -> () {
// //         for node in self.Z {
// //             *(node).distance = *(node).distance.sqrt();
// //         }
// //     }
// //     pub fn sqrt_double() -> () {
// //         for node in self.Z {
// //             *(node).distance = 2 * (*(node).distance.sqrt());
// //         }
// //     }
// //     pub fn power(exp: RealNumber) -> () {
// //         let inv: RealNumber = 1 / exp;
// //         for node in self.Z {
// //             *(node).distance = *(node).distance.powf(inv);
// //         }
// //     }
// //     pub fn plusone() -> () {
// //         for node in self.Z {
// //             *(node).distance += 1;
// //         }
// //     }
// //     pub fn divide(denom: RealNumber) -> () {
// //         for node in self.Z {
// //             *(node).distance = *(node).distance / denom;
// //         }
// //     }
// // }
