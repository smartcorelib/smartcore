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
//! let kmeans = KMeans::new(&x, 2, Default::default()); // Fit to data, 2 clusters
//! let y_hat = kmeans.predict(&x); // use the same points for prediction
//! ```
//!
//! ## References:
//!
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 10.3.1 K-Means Clustering](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["k-means++: The Advantages of Careful Seeding", Arthur D., Vassilvitskii S.](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

extern crate rand;

use rand::Rng;
use std::fmt::Debug;
use std::iter::Sum;

use serde::{Deserialize, Serialize};

use crate::algorithm::neighbour::bbd_tree::BBDTree;
use crate::linalg::Matrix;
use crate::math::distance::euclidian::*;
use crate::math::num::RealNumber;

#[derive(Serialize, Deserialize, Debug)]
/// K-Means clustering algorithm
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
    /// Maximum number of iterations of the k-means algorithm for a single run.
    pub max_iter: usize,
}

impl Default for KMeansParameters {
    fn default() -> Self {
        KMeansParameters { max_iter: 100 }
    }
}

impl<T: RealNumber + Sum> KMeans<T> {
    /// Fit algorithm to _NxM_ matrix where _N_ is number of samples and _M_ is number of features.
    /// * `data` - training instances to cluster
    /// * `k` - number of clusters
    /// * `parameters` - cluster parameters
    pub fn new<M: Matrix<T>>(data: &M, k: usize, parameters: KMeansParameters) -> KMeans<T> {
        let bbd = BBDTree::new(data);

        if k < 2 {
            panic!("Invalid number of clusters: {}", k);
        }

        if parameters.max_iter <= 0 {
            panic!(
                "Invalid maximum number of iterations: {}",
                parameters.max_iter
            );
        }

        let (n, d) = data.shape();

        let mut distortion = T::max_value();
        let mut y = KMeans::kmeans_plus_plus(data, k);
        let mut size = vec![0; k];
        let mut centroids = vec![vec![T::zero(); d]; k];

        for i in 0..n {
            size[y[i]] += 1;
        }

        for i in 0..n {
            for j in 0..d {
                centroids[y[i]][j] = centroids[y[i]][j] + data.get(i, j);
            }
        }

        for i in 0..k {
            for j in 0..d {
                centroids[i][j] = centroids[i][j] / T::from(size[i]).unwrap();
            }
        }

        let mut sums = vec![vec![T::zero(); d]; k];
        for _ in 1..=parameters.max_iter {
            let dist = bbd.clustering(&centroids, &mut sums, &mut size, &mut y);
            for i in 0..k {
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

        KMeans {
            k: k,
            y: y,
            size: size,
            distortion: distortion,
            centroids: centroids,
        }
    }

    /// Predict clusters for `x`
    /// * `x` - matrix with new data to transform of size _KxM_ , where _K_ is number of new samples and _M_ is number of features.
    pub fn predict<M: Matrix<T>>(&self, x: &M) -> M::RowVector {
        let (n, _) = x.shape();
        let mut result = M::zeros(1, n);

        for i in 0..n {
            let mut min_dist = T::max_value();
            let mut best_cluster = 0;

            for j in 0..self.k {
                let dist = Euclidian::squared_distance(&x.get_row_as_vec(i), &self.centroids[j]);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            result.set(0, i, T::from(best_cluster).unwrap());
        }

        result.to_row_vector()
    }

    fn kmeans_plus_plus<M: Matrix<T>>(data: &M, k: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let (n, _) = data.shape();
        let mut y = vec![0; n];
        let mut centroid = data.get_row_as_vec(rng.gen_range(0, n));

        let mut d = vec![T::max_value(); n];

        // pick the next center
        for j in 1..k {
            // Loop over the samples and compare them to the most recent center.  Store
            // the distance from each sample to its closest center in scores.
            for i in 0..n {
                // compute the distance between this sample and the current center
                let dist = Euclidian::squared_distance(&data.get_row_as_vec(i), &centroid);

                if dist < d[i] {
                    d[i] = dist;
                    y[i] = j - 1;
                }
            }

            let mut sum: T = T::zero();
            for i in d.iter() {
                sum = sum + *i;
            }
            let cutoff = T::from(rng.gen::<f64>()).unwrap() * sum;
            let mut cost = T::zero();
            let index = 0;
            for index in 0..n {
                cost = cost + d[index];
                if cost >= cutoff {
                    break;
                }
            }

            centroid = data.get_row_as_vec(index);
        }

        for i in 0..n {
            // compute the distance between this sample and the current center
            let dist = Euclidian::squared_distance(&data.get_row_as_vec(i), &centroid);

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

        let kmeans = KMeans::new(&x, 2, Default::default());

        let y = kmeans.predict(&x);

        for i in 0..y.len() {
            assert_eq!(y[i] as usize, kmeans.y[i]);
        }
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

        let kmeans = KMeans::new(&x, 2, Default::default());

        let deserialized_kmeans: KMeans<f64> =
            serde_json::from_str(&serde_json::to_string(&kmeans).unwrap()).unwrap();

        assert_eq!(kmeans, deserialized_kmeans);
    }
}
