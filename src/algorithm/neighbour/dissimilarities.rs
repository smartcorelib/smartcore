//!
//! ## Dissimilarities for vector-vector distance
//!
//! Different algorithms based on Closest Pairs use to store graph's edges to
//!  compute clusters or closest neighbors in Matrices rows.
//!  The struct `PairwiseDissimilarity` can be used to represent edges between
//!  closest pairs by storing the nodes' indeces.
use std::cmp::{Eq, Ordering, PartialOrd};

use crate::math::num::RealNumber;

///
/// The edge of the subgraph is defined by this structure.
/// The calling algorithm can store a list of dissimilarities as
/// a list of these structures.
///
#[derive(Debug, Clone, Copy)]
pub struct PairwiseDissimilarity<T: RealNumber> {
    // index of the vector in the original `Matrix` or list
    pub node: usize,

    // index of the closest neighbor in the original `Matrix` or same list
    pub neighbour: Option<usize>,

    // measure of distance, according to the algorithm distance function
    // if the distance is None, the edge has value "infinite" or max distance
    // each algorithm has to match
    pub distance: Option<T>,
}

impl<T: RealNumber> Eq for PairwiseDissimilarity<T> {}

impl<T: RealNumber> PartialEq for PairwiseDissimilarity<T> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
            && self.neighbour == other.neighbour
            && self.distance == other.distance
    }
}

impl<T: RealNumber> PartialOrd for PairwiseDissimilarity<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
