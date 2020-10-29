//! 
//! ## Dissimilarities for vector-vector distance
//! 
//! Different algorithms based on Closest Pairs use to store graph's edges to 
//!  compute clusters or closest neighbors in Matrices rows.
//!  The struct `PairwiseDissimilarity` can be used to represent edges between
//!  closest pairs by storing the nodes' indeces.
use std::cmp;
use std::cmp::{Ordering, PartialOrd, Eq};

// use serde::{Deserialize, Serialize};
use crate::math::num::RealNumber;
use crate::math::distance::euclidian::Euclidian;

///
/// The edge of the subgraph is defined by this structure.
/// The calling algorithm can store a list of dissimilarities as 
/// a list of these structures.
/// 
#[derive(Debug)]
pub(crate) struct PairwiseDissimilarity<T: RealNumber> {
    // index of the vector in the original `Matrix` or list
    pub node: usize,
    
    // index of the closest neighbor in the original `Matrix` or same list
    pub neighbour: Option<usize>,
    
    // measure of distance, according to the algorithm distance function
    // if the distance is None, the edge has value "infinite" or max distance
    // each algorithm has to match
    pub distance: Option<T>
}

///
/// Compare distance
///
impl<T: RealNumber> Eq for PairwiseDissimilarity<T> {}

impl<T: RealNumber> Ord for PairwiseDissimilarity<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.distance.is_none() {
           Ordering::Greater
        }
        else if other.distance.is_none() {
            Ordering::Less
        }
        else {
            self.distance.unwrap().cmp(&other.distance.unwrap())
        }

    }
}

impl<T: RealNumber> PartialOrd for PairwiseDissimilarity<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: RealNumber> PartialEq for PairwiseDissimilarity<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

///
/// Distance update formulas
/// Formula for d(I ∪ J, K)  as in Fig.2 in Müllner, 2011
///
impl<T: RealNumber> PairwiseDissimilarity<T> {
    fn single_euclidian(&self, I: &Vec<T>, J: &Vec<T>, K: &Vec<T>) -> T {
        // min(d(I, K), d(J, K))
        cmp::min(Euclidian::squared_distance(I, K), Euclidian::squared_distance(J, K))
    }

    fn complete_euclidian(&self, I: &Vec<T>, J: &Vec<T>, K: &Vec<T>) -> T {
        // max(d(I, K), d(J, K))
        cmp::max(Euclidian::squared_distance(I, K), Euclidian::squared_distance(J, K))
    }

    // ...
}
