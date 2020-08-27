//! # Nearest Neighbors

use crate::algorithm::neighbour::cover_tree::CoverTree;
use crate::algorithm::neighbour::linear_search::LinearKNNSearch;
use crate::math::distance::Distance;
use crate::math::num::FloatExt;
use serde::{Deserialize, Serialize};

///
pub mod knn_classifier;
pub mod knn_regressor;

#[derive(Serialize, Deserialize, Debug)]
pub enum KNNAlgorithmName {
    LinearSearch,
    CoverTree,
}

#[derive(Serialize, Deserialize, Debug)]
enum KNNAlgorithm<T: FloatExt, D: Distance<Vec<T>, T>> {
    LinearSearch(LinearKNNSearch<Vec<T>, T, D>),
    CoverTree(CoverTree<Vec<T>, T, D>),
}

impl KNNAlgorithmName {
    fn fit<T: FloatExt, D: Distance<Vec<T>, T>>(
        &self,
        data: Vec<Vec<T>>,
        distance: D,
    ) -> KNNAlgorithm<T, D> {
        match *self {
            KNNAlgorithmName::LinearSearch => {
                KNNAlgorithm::LinearSearch(LinearKNNSearch::new(data, distance))
            }
            KNNAlgorithmName::CoverTree => KNNAlgorithm::CoverTree(CoverTree::new(data, distance)),
        }
    }
}

impl<T: FloatExt, D: Distance<Vec<T>, T>> KNNAlgorithm<T, D> {
    fn find(&self, from: &Vec<T>, k: usize) -> Vec<usize> {
        match *self {
            KNNAlgorithm::LinearSearch(ref linear) => linear.find(from, k),
            KNNAlgorithm::CoverTree(ref cover) => cover.find(from, k),
        }
    }
}
