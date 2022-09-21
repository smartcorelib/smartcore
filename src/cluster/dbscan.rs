//! # DBSCAN Clustering
//!
//! DBSCAN stands for density-based spatial clustering of applications with noise. This algorithms is good for arbitrary shaped clusters and clusters with noise.
//! The main idea behind DBSCAN is that a point belongs to a cluster if it is close to many points from that cluster. There are two key parameters of DBSCAN:
//!
//! * `eps`, the maximum distance that specifies a neighborhood. Two points are considered to be neighbors if the distance between them are less than or equal to `eps`.
//! * `min_samples`, minimum number of data points that defines a cluster.
//!
//! Based on these two parameters, points are classified as core point, border point, or outlier:
//!
//! * A point is a core point if there are at least `min_samples` number of points, including the point itself in its vicinity.
//! * A point is a border point if it is reachable from a core point and there are less than `min_samples` number of points within its surrounding area.
//! * All points not reachable from any other point are outliers or noise points.
//!
//! The algorithm starts from picking up an arbitrarily point in the dataset.
//! If there are at least `min_samples` points within a radius of `eps` to the point then we consider all these points to be part of the same cluster.
//! The clusters are then expanded by recursively repeating the neighborhood calculation for each neighboring point.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::cluster::dbscan::*;
//! use smartcore::math::distance::Distances;
//! use smartcore::neighbors::KNNAlgorithmName;
//! use smartcore::dataset::generator;
//!
//! // Generate three blobs
//! let blobs = generator::make_blobs(100, 2, 3);
//! let x = DenseMatrix::from_vec(blobs.num_samples, blobs.num_features, &blobs.data);
//! // Fit the algorithm and predict cluster labels
//! let labels = DBSCAN::fit(&x, DBSCANParameters::default().with_eps(3.0)).
//!     and_then(|dbscan| dbscan.predict(&x));
//!
//! println!("{:?}", labels);
//! ```
//!
//! ## References:
//!
//! * ["A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise", Ester M., Kriegel HP., Sander J., Xu X.](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["Density-Based Clustering in Spatial Databases: The Algorithm GDBSCAN and its Applications", Sander J., Ester M., Kriegel HP., Xu X.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.1629&rep=rep1&type=pdf)

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

/// DBSCAN clustering algorithm
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct DBSCAN<T: RealNumber, D: Distance<Vec<T>, T>> {
    cluster_labels: Vec<i16>,
    num_classes: usize,
    knn_algorithm: KNNAlgorithm<T, D>,
    eps: T,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// DBSCAN clustering algorithm parameters
pub struct DBSCANParameters<T: RealNumber, D: Distance<Vec<T>, T>> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// a function that defines a distance between each pair of point in training data.
    /// This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    /// See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    pub distance: D,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    pub min_samples: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    pub eps: T,
    #[cfg_attr(feature = "serde", serde(default))]
    /// KNN algorithm to use.
    pub algorithm: KNNAlgorithmName,
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> DBSCANParameters<T, D> {
    /// a function that defines a distance between each pair of point in training data.
    /// This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    /// See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    pub fn with_distance<DD: Distance<Vec<T>, T>>(self, distance: DD) -> DBSCANParameters<T, DD> {
        DBSCANParameters {
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

/// DBSCAN grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DBSCANSearchParameters<T: RealNumber, D: Distance<Vec<T>, T>> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// a function that defines a distance between each pair of point in training data.
    /// This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    /// See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    pub distance: Vec<D>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    pub min_samples: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    pub eps: Vec<T>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// KNN algorithm to use.
    pub algorithm: Vec<KNNAlgorithmName>,
}

/// DBSCAN grid search iterator
pub struct DBSCANSearchParametersIterator<T: RealNumber, D: Distance<Vec<T>, T>> {
    dbscan_search_parameters: DBSCANSearchParameters<T, D>,
    current_distance: usize,
    current_min_samples: usize,
    current_eps: usize,
    current_algorithm: usize,
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> IntoIterator for DBSCANSearchParameters<T, D> {
    type Item = DBSCANParameters<T, D>;
    type IntoIter = DBSCANSearchParametersIterator<T, D>;

    fn into_iter(self) -> Self::IntoIter {
        DBSCANSearchParametersIterator {
            dbscan_search_parameters: self,
            current_distance: 0,
            current_min_samples: 0,
            current_eps: 0,
            current_algorithm: 0,
        }
    }
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> Iterator for DBSCANSearchParametersIterator<T, D> {
    type Item = DBSCANParameters<T, D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_distance == self.dbscan_search_parameters.distance.len()
            && self.current_min_samples == self.dbscan_search_parameters.min_samples.len()
            && self.current_eps == self.dbscan_search_parameters.eps.len()
            && self.current_algorithm == self.dbscan_search_parameters.algorithm.len()
        {
            return None;
        }

        let next = DBSCANParameters {
            distance: self.dbscan_search_parameters.distance[self.current_distance].clone(),
            min_samples: self.dbscan_search_parameters.min_samples[self.current_min_samples],
            eps: self.dbscan_search_parameters.eps[self.current_eps],
            algorithm: self.dbscan_search_parameters.algorithm[self.current_algorithm].clone(),
        };

        if self.current_distance + 1 < self.dbscan_search_parameters.distance.len() {
            self.current_distance += 1;
        } else if self.current_min_samples + 1 < self.dbscan_search_parameters.min_samples.len() {
            self.current_distance = 0;
            self.current_min_samples += 1;
        } else if self.current_eps + 1 < self.dbscan_search_parameters.eps.len() {
            self.current_distance = 0;
            self.current_min_samples = 0;
            self.current_eps += 1;
        } else if self.current_algorithm + 1 < self.dbscan_search_parameters.algorithm.len() {
            self.current_distance = 0;
            self.current_min_samples = 0;
            self.current_eps = 0;
            self.current_algorithm += 1;
        } else {
            self.current_distance += 1;
            self.current_min_samples += 1;
            self.current_eps += 1;
            self.current_algorithm += 1;
        }

        Some(next)
    }
}

impl<T: RealNumber> Default for DBSCANSearchParameters<T, Euclidian> {
    fn default() -> Self {
        let default_params = DBSCANParameters::default();

        DBSCANSearchParameters {
            distance: vec![default_params.distance],
            min_samples: vec![default_params.min_samples],
            eps: vec![default_params.eps],
            algorithm: vec![default_params.algorithm],
        }
    }
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> PartialEq for DBSCAN<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.cluster_labels.len() == other.cluster_labels.len()
            && self.num_classes == other.num_classes
            && self.eps == other.eps
            && self.cluster_labels == other.cluster_labels
    }
}

impl<T: RealNumber> Default for DBSCANParameters<T, Euclidian> {
    fn default() -> Self {
        DBSCANParameters {
            distance: Distances::euclidian(),
            min_samples: 5,
            eps: T::half(),
            algorithm: KNNAlgorithmName::default(),
        }
    }
}

impl<T: RealNumber + Sum, M: Matrix<T>, D: Distance<Vec<T>, T>>
    UnsupervisedEstimator<M, DBSCANParameters<T, D>> for DBSCAN<T, D>
{
    fn fit(x: &M, parameters: DBSCANParameters<T, D>) -> Result<Self, Failed> {
        DBSCAN::fit(x, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>, D: Distance<Vec<T>, T>> Predictor<M, M::RowVector>
    for DBSCAN<T, D>
{
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber + Sum, D: Distance<Vec<T>, T>> DBSCAN<T, D> {
    /// Fit algorithm to _NxM_ matrix where _N_ is number of samples and _M_ is number of features.
    /// * `data` - training instances to cluster
    /// * `k` - number of clusters
    /// * `parameters` - cluster parameters
    pub fn fit<M: Matrix<T>>(
        x: &M,
        parameters: DBSCANParameters<T, D>,
    ) -> Result<DBSCAN<T, D>, Failed> {
        if parameters.min_samples < 1 {
            return Err(Failed::fit("Invalid minPts"));
        }

        if parameters.eps <= T::zero() {
            return Err(Failed::fit("Invalid radius: "));
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

        Ok(DBSCAN {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    #[cfg(feature = "serde")]
    use crate::math::distance::euclidian::Euclidian;

    #[test]
    fn search_parameters() {
        let parameters = DBSCANSearchParameters {
            min_samples: vec![10, 100],
            eps: vec![1., 2.],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.min_samples, 10);
        assert_eq!(next.eps, 1.);
        let next = iter.next().unwrap();
        assert_eq!(next.min_samples, 100);
        assert_eq!(next.eps, 1.);
        let next = iter.next().unwrap();
        assert_eq!(next.min_samples, 10);
        assert_eq!(next.eps, 2.);
        let next = iter.next().unwrap();
        assert_eq!(next.min_samples, 100);
        assert_eq!(next.eps, 2.);
        assert!(iter.next().is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fit_predict_dbscan() {
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

        let dbscan = DBSCAN::fit(
            &x,
            DBSCANParameters::default()
                .with_eps(0.5)
                .with_min_samples(2),
        )
        .unwrap();

        let predicted_labels = dbscan.predict(&x).unwrap();

        assert_eq!(expected_labels, predicted_labels);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

        let dbscan = DBSCAN::fit(&x, Default::default()).unwrap();

        let deserialized_dbscan: DBSCAN<f64, Euclidian> =
            serde_json::from_str(&serde_json::to_string(&dbscan).unwrap()).unwrap();

        assert_eq!(dbscan, deserialized_dbscan);
    }
}
