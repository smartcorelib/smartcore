//! # DBSCAN Clustering
//!
//! DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
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
//! let labels = DBSCAN::fit(&x, Distances::euclidian(), DBSCANParameters{
//!     min_samples: 5,
//!     eps: 3.0,
//!     algorithm: KNNAlgorithmName::CoverTree
//! }).and_then(|dbscan| dbscan.predict(&x));
//!
//! println!("{:?}", labels);
//! ```
//!
//! ## References:
//!
//! * ["A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise", Ester M., Kriegel HP., Sander J., Xu X.](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["Density-Based Clustering in Spatial Databases: The Algorithm GDBSCAN and its Applications", Sander J., Ester M., Kriegel HP., Xu X.](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.1629&rep=rep1&type=pdf)

extern crate rand;

use std::fmt::Debug;
use std::iter::Sum;

use serde::{Deserialize, Serialize};

use crate::algorithm::neighbour::{KNNAlgorithm, KNNAlgorithmName};
use crate::error::Failed;
use crate::linalg::{row_iter, Matrix};
use crate::math::distance::Distance;
use crate::math::num::RealNumber;
use crate::tree::decision_tree_classifier::which_max;

/// DBSCAN clustering algorithm
#[derive(Serialize, Deserialize, Debug)]
pub struct DBSCAN<T: RealNumber, D: Distance<Vec<T>, T>> {
    cluster_labels: Vec<i16>,
    num_classes: usize,
    knn_algorithm: KNNAlgorithm<T, D>,
    eps: T,
}

#[derive(Debug, Clone)]
/// DBSCAN clustering algorithm parameters
pub struct DBSCANParameters<T: RealNumber> {
    /// Maximum number of iterations of the k-means algorithm for a single run.
    pub min_samples: usize,
    /// The number of samples in a neighborhood for a point to be considered as a core point.
    pub eps: T,
    /// KNN algorithm to use.
    pub algorithm: KNNAlgorithmName,
}

impl<T: RealNumber, D: Distance<Vec<T>, T>> PartialEq for DBSCAN<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.cluster_labels.len() == other.cluster_labels.len()
            && self.num_classes == other.num_classes
            && self.eps == other.eps
            && self.cluster_labels == other.cluster_labels
    }
}

impl<T: RealNumber> Default for DBSCANParameters<T> {
    fn default() -> Self {
        DBSCANParameters {
            min_samples: 5,
            eps: T::half(),
            algorithm: KNNAlgorithmName::CoverTree,
        }
    }
}

impl<T: RealNumber + Sum, D: Distance<Vec<T>, T>> DBSCAN<T, D> {
    /// Fit algorithm to _NxM_ matrix where _N_ is number of samples and _M_ is number of features.
    /// * `data` - training instances to cluster
    /// * `k` - number of clusters
    /// * `parameters` - cluster parameters
    pub fn fit<M: Matrix<T>>(
        x: &M,
        distance: D,
        parameters: DBSCANParameters<T>,
    ) -> Result<DBSCAN<T, D>, Failed> {
        if parameters.min_samples < 1 {
            return Err(Failed::fit(&"Invalid minPts".to_string()));
        }

        if parameters.eps <= T::zero() {
            return Err(Failed::fit(&"Invalid radius: ".to_string()));
        }

        let mut k = 0;
        let unassigned = -2;
        let outlier = -1;

        let n = x.shape().0;
        let mut y = vec![unassigned; n];

        let algo = parameters.algorithm.fit(row_iter(x).collect(), distance)?;

        for (i, e) in row_iter(x).enumerate() {
            if y[i] == unassigned {
                let mut neighbors = algo.find_radius(&e, parameters.eps)?;
                if neighbors.len() < parameters.min_samples {
                    y[i] = outlier;
                } else {
                    y[i] = k;
                    for j in 0..neighbors.len() {
                        if y[neighbors[j].0] == unassigned {
                            y[neighbors[j].0] = k;

                            let mut secondary_neighbors =
                                algo.find_radius(neighbors[j].2, parameters.eps)?;

                            if secondary_neighbors.len() >= parameters.min_samples {
                                neighbors.append(&mut secondary_neighbors);
                            }
                        }

                        if y[neighbors[j].0] == outlier {
                            y[neighbors[j].0] = k;
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
    use crate::math::distance::euclidian::Euclidian;
    use crate::math::distance::Distances;

    #[test]
    fn fit_predict_dbscan() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0, 2.0],
            &[1.1, 2.1],
            &[0.9, 1.9],
            &[1.2, 1.2],
            &[0.8, 1.8],
            &[2.0, 1.0],
            &[2.1, 1.1],
            &[2.2, 1.2],
            &[1.9, 0.9],
            &[1.8, 0.8],
            &[3.0, 5.0],
        ]);

        let expected_labels = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0];

        let dbscan = DBSCAN::fit(
            &x,
            Distances::euclidian(),
            DBSCANParameters {
                min_samples: 5,
                eps: 1.0,
                algorithm: KNNAlgorithmName::CoverTree,
            },
        )
        .unwrap();

        let predicted_labels = dbscan.predict(&x).unwrap();

        assert_eq!(expected_labels, predicted_labels);
    }

    #[test]
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

        let dbscan = DBSCAN::fit(&x, Distances::euclidian(), Default::default()).unwrap();

        let deserialized_dbscan: DBSCAN<f64, Euclidian> =
            serde_json::from_str(&serde_json::to_string(&dbscan).unwrap()).unwrap();

        assert_eq!(dbscan, deserialized_dbscan);
    }
}
