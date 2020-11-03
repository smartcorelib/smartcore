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
//! let cluster = ClusterFastPair::fit(&x, 1.5).unwrap();
//! // return results/labels/dendrogram
//! let dissimilarities_pairs = cluster.edges().unwrap()
//! let labels = cluster.labels().unwrap();
//!
use std::collections::{HashMap, LinkedList};

// use serde::{Deserialize, Serialize};
use crate::algorithm::neighbour::dissimilarities::PairwiseDissimilarity;
use crate::algorithm::neighbour::fastpair::{FastPair, _FastPair};
use crate::error::{Failed, FailedError};
use crate::linalg::Matrix;
use crate::math::num::RealNumber;

pub trait SAHNClustering<T: RealNumber> {
    //
    // Aggregate the data according to given distance threshold
    //
    fn fit<M: Matrix<T>>(data: &M, threshold: T) -> Result<ClusterFastPair, Failed>;

    //
    // Return clusters, assign labels to dissimilarities, according
    // to threshold.
    //
    fn labels(&self) -> &Box<Vec<usize>>;
}

///
/// An implementation of Top-Down (Agglomerative) Hierarchical
///  Clustering with `FastPair`
///
pub struct ClusterFastPair {
    labels: Box<Vec<usize>>,
}

impl<T: RealNumber> SAHNClustering<T> for ClusterFastPair {
    ///
    /// Run `FastPair` on matrix's rows
    ///
    fn fit<M: Matrix<T>>(data: &M, threshold: T) -> Result<ClusterFastPair, Failed> {
        let fastpair = FastPair(data).unwrap();

        // compute labels
        // WIP
        let labels: Box<Vec<usize>> = Box::new(vec![0]);

        Ok(ClusterFastPair { labels: labels })
    }

    fn labels(&self) -> &Box<Vec<usize>> {
        &self.labels
    }
}

///
/// Struct (dendrogram) to hold result of linkage and clustering
///
/// > The term 'dendrogram' has been used with three different
/// > meanings: a mathematical object, a data structure and
/// > a graphical representation of the former two. In the course of this section, we
/// > define a data structure and call it 'stepwise dendrogram'.
///
/// Use `std::collections::LinkedList`
pub struct ClusterLabels<T: RealNumber> {
    Z: LinkedList<Box<PairwiseDissimilarity<T>>>, // list of nodes in clustering process
    current: Option<usize>,                       // used to read as a doubly linked list
}

// impl ClusterLabels<T: RealNumber> {
//     pub fn new(&self, size: usize) -> Self {
//         let mut z = LinkedList::with_capacity(size);
//         Self {
//             Z: z,
//             current: None,
//         }
//     }
//     // add a node to the dendrogram
//     pub fn append(node1: PairwiseDissimilarity, node2: PairwiseDissimilarity, dist: T) -> () {
//         let idx = Z.len();
//         let node: Box<PairwiseDissimilarity> = Box::new(PairwiseDissimilarity {
//             node1: node1,
//             node2: node2,
//             distance: dist,
//             position: idx,
//         });
//         self.Z.push(node);
//     }

//     //
//     // Return list of labels according to number of desired clusters
//     //
//     pub fn labels_by_k(k: usize) {}

//     //
//     // Return list of labels by distance threshold
//     //
//     pub fn labels_by_threshold(threshold: T) {}

//     // Methods for distances post-processing.
//     // All of those have to be monotone or the ordering will change
//     pub fn sqrt() -> () {
//         for node in self.Z {
//             *(node).distance = *(node).distance.sqrt();
//         }
//     }
//     pub fn sqrt_double() -> () {
//         for node in self.Z {
//             *(node).distance = 2 * (*(node).distance.sqrt());
//         }
//     }
//     pub fn power(exp: RealNumber) -> () {
//         let inv: RealNumber = 1 / exp;
//         for node in self.Z {
//             *(node).distance = *(node).distance.powf(inv);
//         }
//     }
//     pub fn plusone() -> () {
//         for node in self.Z {
//             *(node).distance += 1;
//         }
//     }
//     pub fn divide(denom: RealNumber) -> () {
//         for node in self.Z {
//             *(node).distance = *(node).distance / denom;
//         }
//     }
// }
