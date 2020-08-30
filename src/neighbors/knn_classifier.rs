//! # K Nearest Neighbors Classifier
//!
//! SmartCore relies on 2 backend algorithms to speedup KNN queries:
//! * [`LinearSearch`](../../algorithm/neighbour/linear_search/index.html)
//! * [`CoverTree`](../../algorithm/neighbour/cover_tree/index.html)
//!
//! The parameter `k` controls the stability of the KNN estimate: when `k` is small the algorithm is sensitive to the noise in data. When `k` increases the estimator becomes more stable.
//! In terms of the bias variance trade-off the variance decreases with `k` and the bias is likely to increase with `k`.
//!
//! When you don't know which search algorithm and `k` value to use go with default parameters defined by `Default::default()`
//!
//! To fit the model to a 4 x 2 matrix with 4 training samples, 2 features per sample:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::neighbors::knn_classifier::*;
//! use smartcore::math::distance::*;
//!
//! //your explanatory variables. Each row is a training sample with 2 numerical features
//! let x = DenseMatrix::from_array(&[
//!     &[1., 2.],
//!     &[3., 4.],
//!     &[5., 6.],
//!     &[7., 8.],
//! &[9., 10.]]);
//! let y = vec![2., 2., 2., 3., 3.]; //your class labels
//!
//! let knn = KNNClassifier::fit(&x, &y, Distances::euclidian(), Default::default());
//! let y_hat = knn.predict(&x);
//! ```
//!
//! variable `y_hat` will hold a vector with estimates of class labels
//!

use serde::{Deserialize, Serialize};

use crate::linalg::{row_iter, Matrix};
use crate::math::distance::Distance;
use crate::math::num::RealNumber;
use crate::neighbors::{KNNAlgorithm, KNNAlgorithmName, KNNWeightFunction};

/// `KNNClassifier` parameters. Use `Default::default()` for default values.
#[derive(Serialize, Deserialize, Debug)]
pub struct KNNClassifierParameters {
    /// backend search algorithm. See [`knn search algorithms`](../../algorithm/neighbour/index.html). `CoverTree` is default.
    pub algorithm: KNNAlgorithmName,
    /// weighting function that is used to calculate estimated class value. Default function is `KNNWeightFunction::Uniform`.
    pub weight: KNNWeightFunction,
    /// number of training samples to consider when estimating class for new point. Default value is 3.
    pub k: usize,
}

/// K Nearest Neighbors Classifier
#[derive(Serialize, Deserialize, Debug)]
pub struct KNNClassifier<T: RealNumber, D: Distance<Vec<T>, T>> {
    classes: Vec<T>,
    y: Vec<usize>,
    knn_algorithm: KNNAlgorithm<T, D>,
    weight: KNNWeightFunction,
    k: usize,
}

impl Default for KNNClassifierParameters {
    fn default() -> Self {
        KNNClassifierParameters {
            algorithm: KNNAlgorithmName::CoverTree,
            weight: KNNWeightFunction::Uniform,
            k: 3,
        }
    }
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> PartialEq for KNNClassifier<T, D> {
    fn eq(&self, other: &Self) -> bool {
        if self.classes.len() != other.classes.len()
            || self.k != other.k
            || self.y.len() != other.y.len()
        {
            return false;
        } else {
            for i in 0..self.classes.len() {
                if (self.classes[i] - other.classes[i]).abs() > T::epsilon() {
                    return false;
                }
            }
            for i in 0..self.y.len() {
                if self.y[i] != other.y[i] {
                    return false;
                }
            }
            true
        }
    }
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> KNNClassifier<T, D> {
    /// Fits KNN classifier to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data
    /// * `y` - vector with target values (classes) of length N
    /// * `distance` - a function that defines a distance between each pair of point in training data.
    ///    This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    ///    See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    /// * `parameters` - additional parameters like search algorithm and k
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        distance: D,
        parameters: KNNClassifierParameters,
    ) -> KNNClassifier<T, D> {
        let y_m = M::from_row_vector(y.clone());

        let (_, y_n) = y_m.shape();
        let (x_n, _) = x.shape();

        let data = row_iter(x).collect();

        let mut yi: Vec<usize> = vec![0; y_n];
        let classes = y_m.unique();

        for i in 0..y_n {
            let yc = y_m.get(0, i);
            yi[i] = classes.iter().position(|c| yc == *c).unwrap();
        }

        assert!(
            x_n == y_n,
            format!(
                "Size of x should equal size of y; |x|=[{}], |y|=[{}]",
                x_n, y_n
            )
        );

        assert!(
            parameters.k > 1,
            format!("k should be > 1, k=[{}]", parameters.k)
        );

        KNNClassifier {
            classes: classes,
            y: yi,
            k: parameters.k,
            knn_algorithm: parameters.algorithm.fit(data, distance),
            weight: parameters.weight,
        }
    }

    /// Estimates the class labels for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    /// Returns a vector of size N with class estimates.
    pub fn predict<M: Matrix<T>>(&self, x: &M) -> M::RowVector {
        let mut result = M::zeros(1, x.shape().0);

        row_iter(x)
            .enumerate()
            .for_each(|(i, x)| result.set(0, i, self.classes[self.predict_for_row(x)]));

        result.to_row_vector()
    }

    fn predict_for_row(&self, x: Vec<T>) -> usize {
        let search_result = self.knn_algorithm.find(&x, self.k);

        let weights = self
            .weight
            .calc_weights(search_result.iter().map(|v| v.1).collect());
        let w_sum = weights.iter().map(|w| *w).sum();

        let mut c = vec![T::zero(); self.classes.len()];
        let mut max_c = T::zero();
        let mut max_i = 0;
        for (r, w) in search_result.iter().zip(weights.iter()) {
            c[self.y[r.0]] = c[self.y[r.0]] + (*w / w_sum);
            if c[self.y[r.0]] > max_c {
                max_c = c[self.y[r.0]];
                max_i = self.y[r.0];
            }
        }

        max_i
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    use crate::math::distance::Distances;

    #[test]
    fn knn_fit_predict() {
        let x = DenseMatrix::from_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y = vec![2., 2., 2., 3., 3.];
        let knn = KNNClassifier::fit(&x, &y, Distances::euclidian(), Default::default());
        let y_hat = knn.predict(&x);
        assert_eq!(5, Vec::len(&y_hat));
        assert_eq!(y.to_vec(), y_hat);
    }

    #[test]
    fn knn_fit_predict_weighted() {
        let x = DenseMatrix::from_array(&[&[1.], &[2.], &[3.], &[4.], &[5.]]);
        let y = vec![2., 2., 2., 3., 3.];
        let knn = KNNClassifier::fit(
            &x,
            &y,
            Distances::euclidian(),
            KNNClassifierParameters {
                k: 5,
                algorithm: KNNAlgorithmName::LinearSearch,
                weight: KNNWeightFunction::Distance,
            },
        );
        let y_hat = knn.predict(&DenseMatrix::from_array(&[&[4.1]]));
        assert_eq!(vec![3.0], y_hat);
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::from_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y = vec![2., 2., 2., 3., 3.];

        let knn = KNNClassifier::fit(&x, &y, Distances::euclidian(), Default::default());

        let deserialized_knn = bincode::deserialize(&bincode::serialize(&knn).unwrap()).unwrap();

        assert_eq!(knn, deserialized_knn);
    }
}
