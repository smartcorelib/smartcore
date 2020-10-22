//! #  Agglomerative hierarchical clustering
//! 
//! ## Definition
//! SAHN (sequential, agglomerative, hierarchic, nonoverlapping methods)
//!
//! as defined in https://arxiv.org/pdf/1109.2378.pdf **by Müllner, 2011**
//! 
//! > Agglomerative clustering schemes start from the partition of
//! > the data set into singleton nodes and merge step by step the current pair of mutually closest
//! > nodes into a new node until there is one final node left, which comprises the entire data set.
//! > Various clustering schemes share this procedure as a common definition, but differ in the way
//! > in which the measure of inter-cluster dissimilarity is updated after each step. The seven most
//! > common methods are termed single, complete, average (UPGMA), weighted (WPGMA, McQuitty), Ward,
//! > centroid (UPGMC) and median (WPGMC) linkage.
//! 
//! ## Algorithms:
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
//! 
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::cluster::agglomerative::*;
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
//! let primitive_clustering = PrimitiveClustering::fit(&x, 2, Default::default()).unwrap(); // Fit to data, 2 clusters
//!
use std::cmp;
use crate::math::num::RealNumber;
use smartcore::math::distance::euclidian::Euclidian;


pub trait SAHNClustering<T: RealNumber> {
  pub fn fit<M: Matrix<T>>(
    data: &M,
    k: usize,
  ) -> Result<ClusterResult<T>, Failed>;
}

#[derive(Serialize, Deserialize, Debug)]
struct PrimitiveClustering {}

impl PrimitiveClustering {
  fn compute(&self) ->  {
    // 1: procedure Primitive_clustering(S, d) . S: node labels, d: pairwise dissimilarities
    // 2: N ← |S| . Number of input nodes
    // 3: L ← [ ] . Output list
    // 4: size[x] ← 1 for all x ∈ S
    // 5: for i ← 0, . . . , N − 2 do
    // 6: (a, b) ← argmin(S×S)\∆ d
    // 7: Append (a, b, d[a, b]) to L.
    // 8: S ← S \ {a, b}
    // 9: Create a new node label n /∈ S.
    // 10: Update d with the information
    // d[n, x] = d[x, n] = Formula(d[a, x], d[b, x], d[a, b], size[a], size[b], size[x])
    // for all x ∈ S.
    // 11: size[n] ← size[a] + size[b]
    // 12: S ← S ∪ {n}
    // 13: end for
    // 14: return L . the stepwise dendrogram, an ((N − 1) × 3)-matrix
    // 15: end procedure
    // (As usual, ∆ denotes the diagonal in the Cartesian product S × S.)
  }
}

///
/// An implementation of `PRIMITIVE_CLUSTERING`
///
impl SAHNClustering for PrimitiveClustering {

}


///
/// The triple defined as (observation_1, observation_2, distance).
/// This defines the type for the elements of the list that 
///   represents the Stepwise Dendrogram (Müllner, 2011)
/// 
#[derive(Debug)]
struct PairwiseDissimilarity<'a> {
    node1: Vec<RealNumber>;
    node2: Vec<RealNumber>;
    distance: f32;
    position: usize;
};

///
/// Compare distance
/// 
impl Ord for PairwiseDissimilarity {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}

impl PartialOrd for PairwiseDissimilarity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for PairwiseDissimilarity {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

///
/// Distance update formulas
/// Formula for d(I ∪ J, K)  as in Fig.2 in Müllner, 2011
/// 
impl PairwiseDissimilarity<'a> {
  fn single_euclidian(&self, I: &Vec<f32>, J: &Vec<f32>, K: &Vec<f32>) {
    // min(d(I, K), d(J, K))
    cmp::min(Euclidian {}.distance(I, K), Euclidian {}.distance(J, K))
  }

  fn complete_euclidian(&self, I: &Vec<f32>, J: &Vec<f32>, K: &Vec<f32>) {
    // max(d(I, K), d(J, K))
    cmp::max(Euclidian {}.distance(I, K), Euclidian {}.distance(J, K))
  }

  // ...
}

///
/// Struct to hold result of linkage and clustering
/// 
/// > The term 'dendrogram' has been used with three different
/// > meanings: a mathematical object, a data structure and
/// > a graphical representation of the former two. In the course of this section, we
/// > define a data structure and call it 'stepwise dendrogram'.
/// 
/// Rust implementation: see options at <https://stackoverflow.com/a/40897053/2536357>
pub struct ClusterResult<'a> {
    Z: mut Vec<Box<PairwiseDissimilarity<'a`>>>;  // list of nodes in clustering process
    current: Option<usize>;  // used to read as a doubly linked list
}

impl ClusterResult {
    pub fn new(&self, size: usize) -> Self {
        Self { Z: Vec::with_capacity(size), current: None }
    }
    // add a node to the dendrogram
    pub fn append(node1: PairwiseDissimilarity, node2: PairwiseDissimilarity, dist: f32) -> () {
        let idx = Z.len()
        let node: Box<PairwiseDissimilarity> = Box::new(PairwiseDissimilarity {
            node1: node1,
            node2: node2,
            distance: dist,
            position: idx 
        });
        self.Z.push(node);
    }

    // Methods for distances post-processing. 
    // All of those have to be monotone or the ordering will change
    pub fn sqrt() -> () {
        for node in self.Z {
            *(node).distance = *(node).distance.sqrt();
        }
    }
    pub fn sqrt_double() -> () {
        for node in self.Z {
            *(node).distance = 2 * (*(node).distance.sqrt());
        }
    }
    pub fn power(exp: f32) -> () {
        let inv: f32 = 1 / exp;
        for node in self.Z {
            *(node).distance = *(node).distance.powf(inv);
        }        
    }
    pub fn plusone() -> () {
        for node in self.Z {
            *(node).distance += 1;
        }   
    }
    pub fn divide(denom: f32) -> () {
        for node in self.Z {
            *(node).distance = *(node).distance / denom;
        }   
    }
}

