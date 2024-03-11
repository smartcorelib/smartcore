//! # OPTICS Clustering
//!
//! Description goes here
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::cluster::optics::*;
//! use smartcore::math::distance::Distances;
//! use smartcore::algorithm::neighbour::KNNAlgorithmName;
//! use smartcore::dataset::generator;
//!
//! // Generate three blobs
//! let blobs = generator::make_blobs(100, 2, 3);
//! let x = DenseMatrix::from_vec(blobs.num_samples, blobs.num_features, &blobs.data);
//! // Fit the algorithm and predict cluster labels
//! let labels = OPTICS::fit(&x, OPTICSParameters::default().with_eps(3.0)).
//!     and_then(|optics| optics.predict(&x));
//!
//! println!("{:?}", labels);
//! ```
//!
//! ## References:
//!

use std::fmt::Debug;
use std::iter::Sum;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::algorithm::neighbour::{KNNAlgorithm, KNNAlgorithmName};
use crate::api::{Predictor, UnsupervisedEstimator};
use crate::error::Failed;
use crate::linalg::{row_iter, Matrix};
use crate::math::distance::euclidian::Euclidian;
use crate::math::distance::{Distance, Distances};
use crate::math::num::RealNumber;
use crate::tree::decision_tree_classifier::which_max;

/// OPTICS clustering algorithm
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct OPTICS<T: RealNumber, D: Distance<Vec<T>, T>> {
    cluster_labels: Vec<i16>,
    num_classes: usize,
    knn_algorithm: KNNAlgorithm<T, D>,
    eps: T,
}

#[derive(Debug, Clone)]
/// DBSCAN clustering algorithm parameters
pub struct OPTICSParameters<T: RealNumber, D: Distance<Vec<T>, T>> {
    /// a function that defines a distance between each pair of point in training data.
    /// This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    /// See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    pub distance: D,
    /// The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    pub min_samples: usize,
    /// The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    pub eps: T,
    /// KNN algorithm to use.
    pub algorithm: KNNAlgorithmName,
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> OPTICSParameters<T, D> {
    /// a function that defines a distance between each pair of point in training data.
    /// This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    /// See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    pub fn with_distance<DD: Distance<Vec<T>, T>>(self, distance: DD) -> OPTICSParameters<T, DD> {
        OPTICSParameters {
            distance,
            min_samples: self.min_samples,
            eps: self.eps,
            algorithm: self.algorithm,
        }
    }
    /// The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }
    /// The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    pub fn with_eps(mut self, eps: T) -> Self {
        self.eps = eps;
        self
    }
    /// KNN algorithm to use.
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithmName) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> PartialEq for OPTICS<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.cluster_labels.len() == other.cluster_labels.len()
            && self.num_classes == other.num_classes
            && self.eps == other.eps
            && self.cluster_labels == other.cluster_labels
    }
}

impl<T: RealNumber> Default for OPTICSParameters<T, Euclidian> {
    fn default() -> Self {
        OPTICSParameters {
            distance: Distances::euclidian(),
            min_samples: 5,
            eps: T::half(),
            algorithm: KNNAlgorithmName::CoverTree,
        }
    }
}

impl<T: RealNumber + Sum, M: Matrix<T>, D: Distance<Vec<T>, T>>
    UnsupervisedEstimator<M, OPTICSParameters<T, D>> for OPTICS<T, D>
{
    fn fit(x: &M, parameters: OPTICSParameters<T, D>) -> Result<Self, Failed> {
        OPTICS::fit(x, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>, D: Distance<Vec<T>, T>> Predictor<M, M::RowVector>
    for OPTICS<T, D>
{
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber + Sum, D: Distance<Vec<T>, T>> OPTICS<T, D> {
    /// Fit algorithm to _NxM_ matrix where _N_ is number of samples and _M_ is number of features.
    /// * `data` - training instances to cluster
    /// * `k` - number of clusters
    /// * `parameters` - cluster parameters
    pub fn fit<M: Matrix<T>>(
        x: &M,
        parameters: OPTICSParameters<T, D>,
    ) -> Result<OPTICS<T, D>, Failed> {
        if parameters.min_samples < 1 {
            return Err(Failed::fit(&"Invalid minPts".to_string()));
        }

        if parameters.eps <= T::zero() {
            return Err(Failed::fit(&"Invalid radius: ".to_string()));
        }

        let mut k = 0;
        let queued = -2;
        let outlier = -1;
        let undefined = -3;

        let n = x.shape().0;
        let mut y = vec![undefined; n];

        let algo = parameters
            .algorithm
            .fit(row_iter(x).collect(), parameters.distance)?;

        for (i, e) in row_iter(x).enumerate() {
            if y[i] == undefined {
                let mut neighbors = algo.find_radius(&e, parameters.eps)?;
                if neighbors.len() < parameters.min_samples {
                    y[i] = outlier;
                } else {
                    y[i] = k;

                    for j in 0..neighbors.len() {
                        if y[neighbors[j].0] == undefined {
                            y[neighbors[j].0] = queued;
                        }
                    }

                    while !neighbors.is_empty() {
                        let neighbor = neighbors.pop().unwrap();
                        let index = neighbor.0;

                        if y[index] == outlier {
                            y[index] = k;
                        }

                        if y[index] == undefined || y[index] == queued {
                            y[index] = k;

                            let secondary_neighbors =
                                algo.find_radius(neighbor.2, parameters.eps)?;

                            if secondary_neighbors.len() >= parameters.min_samples {
                                for j in 0..secondary_neighbors.len() {
                                    let label = y[secondary_neighbors[j].0];
                                    if label == undefined {
                                        y[secondary_neighbors[j].0] = queued;
                                    }

                                    if label == undefined || label == outlier {
                                        neighbors.push(secondary_neighbors[j]);
                                    }
                                }
                            }
                        }
                    }

                    k += 1;
                }
            }
        }

        Ok(OPTICS {
            cluster_labels: y,
            num_classes: k as usize,
            knn_algorithm: algo,
            eps: parameters.eps,
        })
    }

    /// Predict clusters for `x`
    /// * `x` - matrix with new data to transform of size _KxM_ , where _K_ is number of new samples and _M_ is number of features.
    pub fn predict<M: Matrix<T>>(&self, x: &M) -> Result<M::RowVector, Failed> {
        let (n, m) = x.shape();
        let mut result = M::zeros(1, n);
        let mut row = vec![T::zero(); m];

        for i in 0..n {
            x.copy_row_as_vec(i, &mut row);
            let neighbors = self.knn_algorithm.find_radius(&row, self.eps)?;
            let mut label = vec![0usize; self.num_classes + 1];
            for neighbor in neighbors {
                let yi = self.cluster_labels[neighbor.0];
                if yi < 0 {
                    label[self.num_classes] += 1;
                } else {
                    label[yi as usize] += 1;
                }
            }
            let class = which_max(&label);
            if class != self.num_classes {
                result.set(0, i, T::from(class).unwrap());
            } else {
                result.set(0, i, -T::one());
            }
        }

        Ok(result.to_row_vector())
    }

    fn compute_optics_graph(&self) {}

    fn compute_core_distances(&self) {}

    fn set_reach_dist(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    #[cfg(feature = "serde")]
    use crate::math::distance::euclidian::Euclidian;

    #[test]
    fn fit_predict_optics() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0, 2.0],
            &[1.1, 2.1],
            &[0.9, 1.9],
            &[1.2, 2.2],
            &[0.8, 1.8],
            &[2.0, 1.0],
            &[2.1, 1.1],
            &[1.9, 0.9],
            &[2.2, 1.2],
            &[1.8, 0.8],
            &[3.0, 5.0],
        ]);

        let expected_labels = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0];

        let optics = OPTICS::fit(
            &x,
            OPTICSParameters::default()
                .with_eps(0.5)
                .with_min_samples(2),
        )
        .unwrap();

        let predicted_labels = optics.predict(&x).unwrap();

        assert_eq!(expected_labels, predicted_labels);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ]);

        let optics = OPTICS::fit(&x, Default::default()).unwrap();

        let deserialized_optics: OPTICS<f64, Euclidian> =
            serde_json::from_str(&serde_json::to_string(&optics).unwrap()).unwrap();

        assert_eq!(optics, deserialized_optics);
    }
}
