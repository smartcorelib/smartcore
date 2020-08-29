//! # Nearest Neighbors
//!
//! The k-nearest neighbors (KNN) algorithm is a simple supervised machine learning algorithm that can be used to solve both classification and regression problems.
//! KNN is a non-parametric method that assumes that similar things exist in close proximity.
//!
//! During training the algorithms memorizes all training samples. To make a prediction it finds a predefined set of training samples closest in distance to the new
//! point and uses labels of found samples to calculate value of new point. The number of samples (k) is defined by user and does not change after training.
//!
//! The distance can be any metric measure that is defined as \\( d(x, y) \geq 0\\)
//! and follows three conditions:
//! 1. \\( d(x, y) = 0 \\) if and only \\( x = y \\), positive definiteness
//! 1. \\( d(x, y) = d(y, x) \\), symmetry
//! 1. \\( d(x, y) \leq d(x, z) + d(z, y) \\), 	subadditivity or triangle inequality
//!
//! for all \\(x, y, z \in Z \\)
//!
//! Neighbors-based methods are very simple and are known as non-generalizing machine learning methods since they simply remember all of its training data and is prone to overfitting.
//! Despite its disadvantages, nearest neighbors algorithms has been very successful in a large number of applications because of its flexibility and speed.
//!
//! __Advantages__
//! * The algorithm is simple and fast.
//! * The algorithm is non-parametric: there’s no need to build a model, the algorithm simply stores all training samples in memory.
//! * The algorithm is versatile. It can be used for classification, regression.
//!
//! __Disadvantages__
//! * The algorithm gets significantly slower as the number of examples and/or predictors/independent variables increase.
//!
//! ## References:
//! * ["Nearest Neighbor Pattern Classification" Cover, T.M., IEEE Transactions on Information Theory (1967)](http://ssg.mit.edu/cal/abs/2000_spring/np_dens/classification/cover67.pdf)
//! * ["The Elements of Statistical Learning: Data Mining, Inference, and Prediction" Trevor et al., 2nd edition, chapter 13](https://web.stanford.edu/~hastie/ElemStatLearn/)
//!
//! <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

use crate::algorithm::neighbour::cover_tree::CoverTree;
use crate::algorithm::neighbour::linear_search::LinearKNNSearch;
use crate::math::distance::Distance;
use crate::math::num::FloatExt;
use serde::{Deserialize, Serialize};

/// K Nearest Neighbors Classifier
pub mod knn_classifier;
/// K Nearest Neighbors Regressor
pub mod knn_regressor;

/// Both, KNN classifier and regressor benefits from underlying search algorithms that helps to speed up queries.
/// `KNNAlgorithmName` maintains a list of supported search algorithms, see [KNN algorithms](../algorithm/neighbour/index.html)
#[derive(Serialize, Deserialize, Debug)]
pub enum KNNAlgorithmName {
    /// Heap Search algorithm, see [`LinearSearch`](../algorithm/neighbour/linear_search/index.html)
    LinearSearch,
    /// Cover Tree Search algorithm, see [`CoverTree`](../algorithm/neighbour/cover_tree/index.html)
    CoverTree,
}

/// Weight function that is used to determine estimated value.
#[derive(Serialize, Deserialize, Debug)]
pub enum KNNWeightFunction {
    /// All k nearest points are weighted equally
    Uniform,
    /// k nearest points are weighted by the inverse of their distance. Closer neighbors will have a greater influence than neighbors which are further away.
    Distance,
}

#[derive(Serialize, Deserialize, Debug)]
enum KNNAlgorithm<T: FloatExt, D: Distance<Vec<T>, T>> {
    LinearSearch(LinearKNNSearch<Vec<T>, T, D>),
    CoverTree(CoverTree<Vec<T>, T, D>),
}

impl KNNWeightFunction {
    fn calc_weights<T: FloatExt>(&self, distances: Vec<T>) -> std::vec::Vec<T> {
        match *self {
            KNNWeightFunction::Distance => {
                // if there are any points that has zero distance from one or more training points,
                // those training points are weighted as 1.0 and the other points as 0.0
                if distances.iter().any(|&e| e == T::zero()) {
                    distances
                        .iter()
                        .map(|e| if *e == T::zero() { T::one() } else { T::zero() })
                        .collect()
                } else {
                    distances.iter().map(|e| T::one() / *e).collect()
                }
            }
            KNNWeightFunction::Uniform => vec![T::one(); distances.len()],
        }
    }
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
    fn find(&self, from: &Vec<T>, k: usize) -> Vec<(usize, T)> {
        match *self {
            KNNAlgorithm::LinearSearch(ref linear) => linear.find(from, k),
            KNNAlgorithm::CoverTree(ref cover) => cover.find(from, k),
        }
    }
}
