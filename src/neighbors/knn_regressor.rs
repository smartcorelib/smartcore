//! # K Nearest Neighbors Regressor
//!
//! Regressor that predicts estimated values as a function of k nearest neightbours.
//!
//! `KNNRegressor` relies on 2 backend algorithms to speedup KNN queries:
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
//! use smartcore::neighbors::knn_regressor::*;
//! use smartcore::math::distance::*;
//!
//! //your explanatory variables. Each row is a training sample with 2 numerical features
//! let x = DenseMatrix::from_2d_array(&[
//!     &[1., 1.],
//!     &[2., 2.],
//!     &[3., 3.],
//!     &[4., 4.],
//!     &[5., 5.]]);
//! let y = vec![1., 2., 3., 4., 5.]; //your target values
//!
//! let knn = KNNRegressor::fit(&x, &y, Distances::euclidian(), Default::default()).unwrap();
//! let y_hat = knn.predict(&x).unwrap();
//! ```
//!
//! variable `y_hat` will hold predicted value
//!
//!
use serde::{Deserialize, Serialize};

use crate::error::Failed;
use crate::linalg::{row_iter, BaseVector, Matrix};
use crate::math::distance::Distance;
use crate::math::num::RealNumber;
use crate::neighbors::{KNNAlgorithm, KNNAlgorithmName, KNNWeightFunction};

/// `KNNRegressor` parameters. Use `Default::default()` for default values.
#[derive(Serialize, Deserialize, Debug)]
pub struct KNNRegressorParameters {
    /// backend search algorithm. See [`knn search algorithms`](../../algorithm/neighbour/index.html). `CoverTree` is default.
    pub algorithm: KNNAlgorithmName,
    /// weighting function that is used to calculate estimated class value. Default function is `KNNWeightFunction::Uniform`.
    pub weight: KNNWeightFunction,
    /// number of training samples to consider when estimating class for new point. Default value is 3.
    pub k: usize,
}

/// K Nearest Neighbors Regressor
#[derive(Serialize, Deserialize, Debug)]
pub struct KNNRegressor<T: RealNumber, D: Distance<Vec<T>, T>> {
    y: Vec<T>,
    knn_algorithm: KNNAlgorithm<T, D>,
    weight: KNNWeightFunction,
    k: usize,
}

impl Default for KNNRegressorParameters {
    fn default() -> Self {
        KNNRegressorParameters {
            algorithm: KNNAlgorithmName::CoverTree,
            weight: KNNWeightFunction::Uniform,
            k: 3,
        }
    }
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> PartialEq for KNNRegressor<T, D> {
    fn eq(&self, other: &Self) -> bool {
        if self.k != other.k || self.y.len() != other.y.len() {
            return false;
        } else {
            for i in 0..self.y.len() {
                if (self.y[i] - other.y[i]).abs() > T::epsilon() {
                    return false;
                }
            }
            true
        }
    }
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> KNNRegressor<T, D> {
    /// Fits KNN regressor to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data
    /// * `y` - vector with real values
    /// * `distance` - a function that defines a distance between each pair of point in training data.
    ///    This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    ///    See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    /// * `parameters` - additional parameters like search algorithm and k
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        distance: D,
        parameters: KNNRegressorParameters,
    ) -> Result<KNNRegressor<T, D>, Failed> {
        let y_m = M::from_row_vector(y.clone());

        let (_, y_n) = y_m.shape();
        let (x_n, _) = x.shape();

        let data = row_iter(x).collect();

        if x_n != y_n {
            return Err(Failed::fit(&format!(
                "Size of x should equal size of y; |x|=[{}], |y|=[{}]",
                x_n, y_n
            )));
        }

        if parameters.k <= 1 {
            return Err(Failed::fit(&format!(
                "k should be > 1, k=[{}]",
                parameters.k
            )));
        }

        Ok(KNNRegressor {
            y: y.to_vec(),
            k: parameters.k,
            knn_algorithm: parameters.algorithm.fit(data, distance)?,
            weight: parameters.weight,
        })
    }

    /// Predict the target for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    /// Returns a vector of size N with estimates.
    pub fn predict<M: Matrix<T>>(&self, x: &M) -> Result<M::RowVector, Failed> {
        let mut result = M::zeros(1, x.shape().0);

        for (i, x) in row_iter(x).enumerate() {
            result.set(0, i, self.predict_for_row(x)?);
        }

        Ok(result.to_row_vector())
    }

    fn predict_for_row(&self, x: Vec<T>) -> Result<T, Failed> {
        let search_result = self.knn_algorithm.find(&x, self.k)?;
        let mut result = T::zero();

        let weights = self
            .weight
            .calc_weights(search_result.iter().map(|v| v.1).collect());
        let w_sum = weights.iter().map(|w| *w).sum();

        for (r, w) in search_result.iter().zip(weights.iter()) {
            result = result + self.y[r.0] * (*w / w_sum);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    use crate::math::distance::Distances;

    #[test]
    fn knn_fit_predict_weighted() {
        let x =
            DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y: Vec<f64> = vec![1., 2., 3., 4., 5.];
        let y_exp = vec![1., 2., 3., 4., 5.];
        let knn = KNNRegressor::fit(
            &x,
            &y,
            Distances::euclidian(),
            KNNRegressorParameters {
                k: 3,
                algorithm: KNNAlgorithmName::LinearSearch,
                weight: KNNWeightFunction::Distance,
            },
        )
        .unwrap();
        let y_hat = knn.predict(&x).unwrap();
        assert_eq!(5, Vec::len(&y_hat));
        for i in 0..y_hat.len() {
            assert!((y_hat[i] - y_exp[i]).abs() < std::f64::EPSILON);
        }
    }

    #[test]
    fn knn_fit_predict_uniform() {
        let x =
            DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y: Vec<f64> = vec![1., 2., 3., 4., 5.];
        let y_exp = vec![2., 2., 3., 4., 4.];
        let knn = KNNRegressor::fit(&x, &y, Distances::euclidian(), Default::default()).unwrap();
        let y_hat = knn.predict(&x).unwrap();
        assert_eq!(5, Vec::len(&y_hat));
        for i in 0..y_hat.len() {
            assert!((y_hat[i] - y_exp[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn serde() {
        let x =
            DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y = vec![1., 2., 3., 4., 5.];

        let knn = KNNRegressor::fit(&x, &y, Distances::euclidian(), Default::default()).unwrap();

        let deserialized_knn = bincode::deserialize(&bincode::serialize(&knn).unwrap()).unwrap();

        assert_eq!(knn, deserialized_knn);
    }
}
