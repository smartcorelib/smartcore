//! # K-Means Clustering
//!
//! K-means clustering partitions data into k clusters in a way that data points in the same cluster are similar and data points in the different clusters are farther apart.
//! Similarity of two points is determined by the [Euclidian Distance](../../math/distance/euclidian/index.html) between them.
//!
//! K-means algorithm is not capable of determining the number of clusters. You need to choose this number yourself.
//! One way to choose optimal number of clusters is to use [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)).
//!
//! At the high level K-Means algorithm works as follows. K data points are randomly chosen from a given dataset as cluster centers (centroids) and
//! all training instances are added to the closest cluster. After that the centroids, representing the mean of the instances of each cluster are re-calculated and
//! these re-calculated centroids becoming the new centers of their respective clusters. Next all instances of the training set are re-assigned to their closest cluster again.
//! This iterative process continues until convergence is achieved and the clusters are considered settled.
//!
//! Initial choice of K data points is very important and has big effect on performance of the algorithm. SmartCore uses k-means++ algorithm to initialize cluster centers.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::cluster::kmeans::*;
//!
//! // Iris data
//! let x = DenseMatrix::from_2d_array(&[
//!            &[5.1, 3.5, 1.4, 0.2],
//!            &[4.9, 3.0, 1.4, 0.2],
//!            &[4.7, 3.2, 1.3, 0.2],
//!            &[4.6, 3.1, 1.5, 0.2],
//!            &[5.0, 3.6, 1.4, 0.2],
//!            &[5.4, 3.9, 1.7, 0.4],
//!            &[4.6, 3.4, 1.4, 0.3],
//!            &[5.0, 3.4, 1.5, 0.2],
//!            &[4.4, 2.9, 1.4, 0.2],
//!            &[4.9, 3.1, 1.5, 0.1],
//!            &[7.0, 3.2, 4.7, 1.4],
//!            &[6.4, 3.2, 4.5, 1.5],
//!            &[6.9, 3.1, 4.9, 1.5],
//!            &[5.5, 2.3, 4.0, 1.3],
//!            &[6.5, 2.8, 4.6, 1.5],
//!            &[5.7, 2.8, 4.5, 1.3],
//!            &[6.3, 3.3, 4.7, 1.6],
//!            &[4.9, 2.4, 3.3, 1.0],
//!            &[6.6, 2.9, 4.6, 1.3],
//!            &[5.2, 2.7, 3.9, 1.4],
//!            ]);
//!
//! let kmeans = KMeans::fit(&x, KMeansParameters::default().with_k(2)).unwrap(); // Fit to data, 2 clusters
//! let y_hat = kmeans.predict(&x).unwrap(); // use the same points for prediction
//! ```
//!
//! ## References:
//!
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 10.3.1 K-Means Clustering](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["k-means++: The Advantages of Careful Seeding", Arthur D., Vassilvitskii S.](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

use rand::Rng;
use std::fmt::Debug;
use std::iter::Sum;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::algorithm::neighbour::bbd_tree::BBDTree;
use crate::api::{Predictor, UnsupervisedEstimator};
use crate::error::Failed;
use crate::linalg::Matrix;
use crate::math::distance::euclidian::*;
use crate::math::num::RealNumber;

/// K-Means clustering algorithm
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct KMeans<T: RealNumber> {
    k: usize,
    y: Vec<usize>,
    size: Vec<usize>,
    distortion: T,
    centroids: Vec<Vec<T>>,
}

impl<T: RealNumber> PartialEq for KMeans<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.k != other.k
            || self.size != other.size
            || self.centroids.len() != other.centroids.len()
        {
            false
        } else {
            let n_centroids = self.centroids.len();
            for i in 0..n_centroids {
                if self.centroids[i].len() != other.centroids[i].len() {
                    return false;
                }
                for j in 0..self.centroids[i].len() {
                    if (self.centroids[i][j] - other.centroids[i][j]).abs() > T::epsilon() {
                        return false;
                    }
                }
            }
            true
        }
    }
}

#[derive(Debug, Clone)]
/// K-Means clustering algorithm parameters
pub struct KMeansParameters {
    /// Number of clusters.
    pub k: usize,
    /// Maximum number of iterations of the k-means algorithm for a single run.
    pub max_iter: usize,
}

impl KMeansParameters {
    /// Number of clusters.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
    /// Maximum number of iterations of the k-means algorithm for a single run.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl Default for KMeansParameters {
    fn default() -> Self {
        KMeansParameters {
            k: 2,
            max_iter: 100,
        }
    }
}

impl<T: RealNumber + Sum, M: Matrix<T>> UnsupervisedEstimator<M, KMeansParameters> for KMeans<T> {
    fn fit(x: &M, parameters: KMeansParameters) -> Result<Self, Failed> {
        KMeans::fit(x, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>> Predictor<M, M::RowVector> for KMeans<T> {
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber + Sum> KMeans<T> {
    /// Fit algorithm to _NxM_ matrix where _N_ is number of samples and _M_ is number of features.
    /// * `data` - training instances to cluster    
    /// * `parameters` - cluster parameters
    pub fn fit<M: Matrix<T>>(data: &M, parameters: KMeansParameters) -> Result<KMeans<T>, Failed> {
        let bbd = BBDTree::new(data);

        if parameters.k < 2 {
            return Err(Failed::fit(&format!(
                "invalid number of clusters: {}",
                parameters.k
            )));
        }

        if parameters.max_iter == 0 {
            return Err(Failed::fit(&format!(
                "invalid maximum number of iterations: {}",
                parameters.max_iter
            )));
        }

        let (n, d) = data.shape();

        let mut distortion = T::max_value();
        let mut y = KMeans::kmeans_plus_plus(data, parameters.k);
        let mut size = vec![0; parameters.k];
        let mut centroids = vec![vec![T::zero(); d]; parameters.k];

        for i in 0..n {
            size[y[i]] += 1;
        }

        for i in 0..n {
            for j in 0..d {
                centroids[y[i]][j] += data.get(i, j);
            }
        }

        for i in 0..parameters.k {
            for j in 0..d {
                centroids[i][j] /= T::from(size[i]).unwrap();
            }
        }

        let mut sums = vec![vec![T::zero(); d]; parameters.k];
        for _ in 1..=parameters.max_iter {
            let dist = bbd.clustering(&centroids, &mut sums, &mut size, &mut y);
            for i in 0..parameters.k {
                if size[i] > 0 {
                    for j in 0..d {
                        centroids[i][j] = T::from(sums[i][j]).unwrap() / T::from(size[i]).unwrap();
                    }
                }
            }

            if distortion <= dist {
                break;
            } else {
                distortion = dist;
            }
        }

        Ok(KMeans {
            k: parameters.k,
            y,
            size,
            distortion,
            centroids,
        })
    }

    /// Predict clusters for `x`
    /// * `x` - matrix with new data to transform of size _KxM_ , where _K_ is number of new samples and _M_ is number of features.
    pub fn predict<M: Matrix<T>>(&self, x: &M) -> Result<M::RowVector, Failed> {
        let (n, m) = x.shape();
        let mut result = M::zeros(1, n);

        let mut row = vec![T::zero(); m];

        for i in 0..n {
            let mut min_dist = T::max_value();
            let mut best_cluster = 0;

            for j in 0..self.k {
                x.copy_row_as_vec(i, &mut row);
                let dist = Euclidian::squared_distance(&row, &self.centroids[j]);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            result.set(0, i, T::from(best_cluster).unwrap());
        }

        Ok(result.to_row_vector())
    }

    fn kmeans_plus_plus<M: Matrix<T>>(data: &M, k: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let (n, m) = data.shape();
        let mut y = vec![0; n];
        let mut centroid = data.get_row_as_vec(rng.gen_range(0..n));

        let mut d = vec![T::max_value(); n];

        let mut row = vec![T::zero(); m];

        for j in 1..k {
            for i in 0..n {
                data.copy_row_as_vec(i, &mut row);
                let dist = Euclidian::squared_distance(&row, &centroid);

                if dist < d[i] {
                    d[i] = dist;
                    y[i] = j - 1;
                }
            }

            let mut sum: T = T::zero();
            for i in d.iter() {
                sum += *i;
            }
            let cutoff = T::from(rng.gen::<f64>()).unwrap() * sum;
            let mut cost = T::zero();
            let mut index = 0;
            while index < n {
                cost += d[index];
                if cost >= cutoff {
                    break;
                }
                index += 1;
            }

            data.copy_row_as_vec(index, &mut centroid);
        }

        for i in 0..n {
            data.copy_row_as_vec(i, &mut row);
            let dist = Euclidian::squared_distance(&row, &centroid);

            if dist < d[i] {
                d[i] = dist;
                y[i] = k - 1;
            }
        }

        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_k() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]);

        assert!(KMeans::fit(&x, KMeansParameters::default().with_k(0)).is_err());
        assert_eq!(
            "Fit failed: invalid number of clusters: 1",
            KMeans::fit(&x, KMeansParameters::default().with_k(1))
                .unwrap_err()
                .to_string()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn fit_predict_iris() {
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

        let kmeans = KMeans::fit(&x, Default::default()).unwrap();

        let y = kmeans.predict(&x).unwrap();

        for i in 0..y.len() {
            assert_eq!(y[i] as usize, kmeans.y[i]);
        }
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

        let kmeans = KMeans::fit(&x, Default::default()).unwrap();

        let deserialized_kmeans: KMeans<f64> =
            serde_json::from_str(&serde_json::to_string(&kmeans).unwrap()).unwrap();

        assert_eq!(kmeans, deserialized_kmeans);
    }
}
