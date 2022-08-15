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
//! let knn = KNNRegressor::fit(&x, &y, Default::default()).unwrap();
//! let y_hat = knn.predict(&x).unwrap();
//! ```
//!
//! variable `y_hat` will hold predicted value
//!
//!
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::algorithm::neighbour::{KNNAlgorithm, KNNAlgorithmName};
use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::{row_iter, BaseVector, Matrix};
use crate::math::distance::euclidian::Euclidian;
use crate::math::distance::{Distance, Distances};
use crate::math::num::RealNumber;
use crate::neighbors::KNNWeightFunction;

/// `KNNRegressor` parameters. Use `Default::default()` for default values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct KNNRegressorParameters<T: RealNumber, D: Distance<Vec<T>, T>> {
    /// a function that defines a distance between each pair of point in training data.
    /// This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    /// See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    distance: D,
    /// backend search algorithm. See [`knn search algorithms`](../../algorithm/neighbour/index.html). `CoverTree` is default.
    pub algorithm: KNNAlgorithmName,
    /// weighting function that is used to calculate estimated class value. Default function is `KNNWeightFunction::Uniform`.
    pub weight: KNNWeightFunction,
    /// number of training samples to consider when estimating class for new point. Default value is 3.
    pub k: usize,
    /// this parameter is not used
    t: PhantomData<T>,
}

/// K Nearest Neighbors Regressor
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct KNNRegressor<T: RealNumber, D: Distance<Vec<T>, T>> {
    y: Vec<T>,
    knn_algorithm: KNNAlgorithm<T, D>,
    weight: KNNWeightFunction,
    k: usize,
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> KNNRegressorParameters<T, D> {
    /// number of training samples to consider when estimating class for new point. Default value is 3.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
    /// a function that defines a distance between each pair of point in training data.
    /// This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    /// See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    pub fn with_distance<DD: Distance<Vec<T>, T>>(
        self,
        distance: DD,
    ) -> KNNRegressorParameters<T, DD> {
        KNNRegressorParameters {
            distance,
            algorithm: self.algorithm,
            weight: self.weight,
            k: self.k,
            t: PhantomData,
        }
    }
    /// backend search algorithm. See [`knn search algorithms`](../../algorithm/neighbour/index.html). `CoverTree` is default.
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithmName) -> Self {
        self.algorithm = algorithm;
        self
    }
    /// weighting function that is used to calculate estimated class value. Default function is `KNNWeightFunction::Uniform`.
    pub fn with_weight(mut self, weight: KNNWeightFunction) -> Self {
        self.weight = weight;
        self
    }
}

impl<T: RealNumber> Default for KNNRegressorParameters<T, Euclidian> {
    fn default() -> Self {
        KNNRegressorParameters {
            distance: Distances::euclidian(),
            algorithm: KNNAlgorithmName::CoverTree,
            weight: KNNWeightFunction::Uniform,
            k: 3,
            t: PhantomData,
        }
    }
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> PartialEq for KNNRegressor<T, D> {
    fn eq(&self, other: &Self) -> bool {
        if self.k != other.k || self.y.len() != other.y.len() {
            false
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

impl<T: RealNumber, M: Matrix<T>, D: Distance<Vec<T>, T>>
    SupervisedEstimator<M, M::RowVector, KNNRegressorParameters<T, D>> for KNNRegressor<T, D>
{
    fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: KNNRegressorParameters<T, D>,
    ) -> Result<Self, Failed> {
        KNNRegressor::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>, D: Distance<Vec<T>, T>> Predictor<M, M::RowVector>
    for KNNRegressor<T, D>
{
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> KNNRegressor<T, D> {
    /// Fits KNN regressor to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data
    /// * `y` - vector with real values    
    /// * `parameters` - additional parameters like search algorithm and k
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        parameters: KNNRegressorParameters<T, D>,
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

        if parameters.k < 1 {
            return Err(Failed::fit(&format!(
                "k should be > 0, k=[{}]",
                parameters.k
            )));
        }

        Ok(KNNRegressor {
            y: y.to_vec(),
            k: parameters.k,
            knn_algorithm: parameters.algorithm.fit(data, parameters.distance)?,
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
        let w_sum = weights.iter().copied().sum();

        for (r, w) in search_result.iter().zip(weights.iter()) {
            result += self.y[r.0] * (*w / w_sum);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    use crate::math::distance::Distances;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn knn_fit_predict_weighted() {
        let x =
            DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y: Vec<f64> = vec![1., 2., 3., 4., 5.];
        let y_exp = vec![1., 2., 3., 4., 5.];
        let knn = KNNRegressor::fit(
            &x,
            &y,
            KNNRegressorParameters::default()
                .with_k(3)
                .with_distance(Distances::euclidian())
                .with_algorithm(KNNAlgorithmName::LinearSearch)
                .with_weight(KNNWeightFunction::Distance),
        )
        .unwrap();
        let y_hat = knn.predict(&x).unwrap();
        assert_eq!(5, Vec::len(&y_hat));
        for i in 0..y_hat.len() {
            assert!((y_hat[i] - y_exp[i]).abs() < std::f64::EPSILON);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn knn_fit_predict_uniform() {
        let x =
            DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y: Vec<f64> = vec![1., 2., 3., 4., 5.];
        let y_exp = vec![2., 2., 3., 4., 4.];
        let knn = KNNRegressor::fit(&x, &y, Default::default()).unwrap();
        let y_hat = knn.predict(&x).unwrap();
        assert_eq!(5, Vec::len(&y_hat));
        for i in 0..y_hat.len() {
            assert!((y_hat[i] - y_exp[i]).abs() < 1e-7);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x =
            DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]]);
        let y = vec![1., 2., 3., 4., 5.];

        let knn = KNNRegressor::fit(&x, &y, Default::default()).unwrap();

        let deserialized_knn = bincode::deserialize(&bincode::serialize(&knn).unwrap()).unwrap();

        assert_eq!(knn, deserialized_knn);
    }
}
