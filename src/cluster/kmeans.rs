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
//! use smartcore::linalg::basic::matrix::DenseMatrix;
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
//! let y_hat: Vec<u8> = kmeans.predict(&x).unwrap(); // use the same points for prediction
//! ```
//!
//! ## References:
//!
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 10.3.1 K-Means Clustering](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["k-means++: The Advantages of Careful Seeding", Arthur D., Vassilvitskii S.](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

use std::fmt::Debug;
use std::marker::PhantomData;

use ::rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::algorithm::neighbour::bbd_tree::BBDTree;
use crate::api::{Predictor, UnsupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::metrics::distance::euclidian::*;
use crate::numbers::basenum::Number;
use crate::rand_custom::get_rng_impl;

/// K-Means clustering algorithm
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct KMeans<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    k: usize,
    _y: Vec<usize>,
    size: Vec<usize>,
    distortion: f64,
    centroids: Vec<Vec<f64>>,
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq for KMeans<TX, TY, X, Y> {
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
                    if (self.centroids[i][j] - other.centroids[i][j]).abs() > std::f64::EPSILON {
                        return false;
                    }
                }
            }
            true
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// K-Means clustering algorithm parameters
pub struct KMeansParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of clusters.
    pub k: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Maximum number of iterations of the k-means algorithm for a single run.
    pub max_iter: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Determines random number generation for centroid initialization.
    /// Use an int to make the randomness deterministic
    pub seed: Option<u64>,
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
            seed: Option::None,
        }
    }
}

/// KMeans grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct KMeansSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of clusters.
    pub k: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Maximum number of iterations of the k-means algorithm for a single run.
    pub max_iter: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Determines random number generation for centroid initialization.
    /// Use an int to make the randomness deterministic
    pub seed: Vec<Option<u64>>,
}

/// KMeans grid search iterator
pub struct KMeansSearchParametersIterator {
    kmeans_search_parameters: KMeansSearchParameters,
    current_k: usize,
    current_max_iter: usize,
    current_seed: usize,
}

impl IntoIterator for KMeansSearchParameters {
    type Item = KMeansParameters;
    type IntoIter = KMeansSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        KMeansSearchParametersIterator {
            kmeans_search_parameters: self,
            current_k: 0,
            current_max_iter: 0,
            current_seed: 0,
        }
    }
}

impl Iterator for KMeansSearchParametersIterator {
    type Item = KMeansParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_k == self.kmeans_search_parameters.k.len()
            && self.current_max_iter == self.kmeans_search_parameters.max_iter.len()
            && self.current_seed == self.kmeans_search_parameters.seed.len()
        {
            return None;
        }

        let next = KMeansParameters {
            k: self.kmeans_search_parameters.k[self.current_k],
            max_iter: self.kmeans_search_parameters.max_iter[self.current_max_iter],
            seed: self.kmeans_search_parameters.seed[self.current_seed],
        };

        if self.current_k + 1 < self.kmeans_search_parameters.k.len() {
            self.current_k += 1;
        } else if self.current_max_iter + 1 < self.kmeans_search_parameters.max_iter.len() {
            self.current_k = 0;
            self.current_max_iter += 1;
        } else if self.current_seed + 1 < self.kmeans_search_parameters.seed.len() {
            self.current_k = 0;
            self.current_max_iter = 0;
            self.current_seed += 1;
        } else {
            self.current_k += 1;
            self.current_max_iter += 1;
            self.current_seed += 1;
        }

        Some(next)
    }
}

impl Default for KMeansSearchParameters {
    fn default() -> Self {
        let default_params = KMeansParameters::default();

        KMeansSearchParameters {
            k: vec![default_params.k],
            max_iter: vec![default_params.max_iter],
            seed: vec![default_params.seed],
        }
    }
}

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    UnsupervisedEstimator<X, KMeansParameters> for KMeans<TX, TY, X, Y>
{
    fn fit(x: &X, parameters: KMeansParameters) -> Result<Self, Failed> {
        KMeans::fit(x, parameters)
    }
}

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> Predictor<X, Y>
    for KMeans<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> KMeans<TX, TY, X, Y> {
    /// Fit algorithm to _NxM_ matrix where _N_ is number of samples and _M_ is number of features.
    /// * `data` - training instances to cluster    
    /// * `parameters` - cluster parameters
    pub fn fit(data: &X, parameters: KMeansParameters) -> Result<KMeans<TX, TY, X, Y>, Failed> {
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

        let mut distortion = std::f64::MAX;
        let mut y = KMeans::<TX, TY, X, Y>::kmeans_plus_plus(data, parameters.k, parameters.seed);
        let mut size = vec![0; parameters.k];
        let mut centroids = vec![vec![0f64; d]; parameters.k];

        for i in 0..n {
            size[y[i]] += 1;
        }

        for i in 0..n {
            for j in 0..d {
                centroids[y[i]][j] += data.get((i, j)).to_f64().unwrap();
            }
        }

        for i in 0..parameters.k {
            for j in 0..d {
                centroids[i][j] /= size[i] as f64;
            }
        }

        let mut sums = vec![vec![0f64; d]; parameters.k];
        for _ in 1..=parameters.max_iter {
            let dist = bbd.clustering(&centroids, &mut sums, &mut size, &mut y);
            for i in 0..parameters.k {
                if size[i] > 0 {
                    for j in 0..d {
                        centroids[i][j] = sums[i][j] / size[i] as f64;
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
            _y: y,
            size,
            distortion,
            centroids,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        })
    }

    /// Predict clusters for `x`
    /// * `x` - matrix with new data to transform of size _KxM_ , where _K_ is number of new samples and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let (n, _) = x.shape();
        let mut result = Y::zeros(n);

        let mut row = vec![0f64; x.shape().1];

        for i in 0..n {
            let mut min_dist = std::f64::MAX;
            let mut best_cluster = 0;

            for j in 0..self.k {
                x.get_row(i)
                    .iterator(0)
                    .zip(row.iter_mut())
                    .for_each(|(&x, r)| *r = x.to_f64().unwrap());
                let dist = Euclidian::squared_distance(&row, &self.centroids[j]);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            result.set(i, TY::from_usize(best_cluster).unwrap());
        }

        Ok(result)
    }

    fn kmeans_plus_plus(data: &X, k: usize, seed: Option<u64>) -> Vec<usize> {
        let mut rng = get_rng_impl(seed);
        let (n, _) = data.shape();
        let mut y = vec![0; n];
        let mut centroid: Vec<TX> = data
            .get_row(rng.gen_range(0..n))
            .iterator(0)
            .cloned()
            .collect();

        let mut d = vec![std::f64::MAX; n];
        let mut row = vec![TX::zero(); data.shape().1];

        for j in 1..k {
            for i in 0..n {
                data.get_row(i)
                    .iterator(0)
                    .zip(row.iter_mut())
                    .for_each(|(&x, r)| *r = x);
                let dist = Euclidian::squared_distance(&row, &centroid);

                if dist < d[i] {
                    d[i] = dist;
                    y[i] = j - 1;
                }
            }

            let mut sum = 0f64;
            for i in d.iter() {
                sum += *i;
            }
            let cutoff = rng.gen::<f64>() * sum;
            let mut cost = 0f64;
            let mut index = 0;
            while index < n {
                cost += d[index];
                if cost >= cutoff {
                    break;
                }
                index += 1;
            }

            centroid = data.get_row(index).iterator(0).cloned().collect();
        }

        for i in 0..n {
            data.get_row(i)
                .iterator(0)
                .zip(row.iter_mut())
                .for_each(|(&x, r)| *r = x);
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
    use crate::linalg::basic::matrix::DenseMatrix;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_k() {
        let x = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6]]);

        assert!(KMeans::<i32, i32, DenseMatrix<i32>, Vec<i32>>::fit(
            &x,
            KMeansParameters::default().with_k(0)
        )
        .is_err());
        assert_eq!(
            "Fit failed: invalid number of clusters: 1",
            KMeans::<i32, i32, DenseMatrix<i32>, Vec<i32>>::fit(
                &x,
                KMeansParameters::default().with_k(1)
            )
            .unwrap_err()
            .to_string()
        );
    }

    #[test]
    fn search_parameters() {
        let parameters = KMeansSearchParameters {
            k: vec![2, 4],
            max_iter: vec![10, 100],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.k, 2);
        assert_eq!(next.max_iter, 10);
        let next = iter.next().unwrap();
        assert_eq!(next.k, 4);
        assert_eq!(next.max_iter, 10);
        let next = iter.next().unwrap();
        assert_eq!(next.k, 2);
        assert_eq!(next.max_iter, 100);
        let next = iter.next().unwrap();
        assert_eq!(next.k, 4);
        assert_eq!(next.max_iter, 100);
        assert!(iter.next().is_none());
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

        let y: Vec<usize> = kmeans.predict(&x).unwrap();

        for i in 0..y.len() {
            assert_eq!(y[i] as usize, kmeans._y[i]);
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

        let kmeans: KMeans<f32, f32, DenseMatrix<f32>, Vec<f32>> =
            KMeans::fit(&x, Default::default()).unwrap();

        let deserialized_kmeans: KMeans<f32, f32, DenseMatrix<f32>, Vec<f32>> =
            serde_json::from_str(&serde_json::to_string(&kmeans).unwrap()).unwrap();

        assert_eq!(kmeans, deserialized_kmeans);
    }
}
