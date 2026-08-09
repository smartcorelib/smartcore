//! # Agglomerative Hierarchical Clustering
//!
//! Agglomerative clustering is a "bottom-up" hierarchical clustering method. It works by placing each data point in its own cluster and then successively merging the two most similar clusters until a stopping criterion is met. This process creates a tree-based hierarchy of clusters known as a dendrogram.
//!
//! The similarity of two clusters is determined by a **linkage criterion**. This implementation uses **single-linkage**, where the distance between two clusters is defined as the minimum distance between any single point in the first cluster and any single point in the second cluster. The distance between points is the standard Euclidean distance.
//!
//! The algorithm first builds the full hierarchy of `N-1` merges. To obtain a specific number of clusters, `n_clusters`, the algorithm then effectively "cuts" the dendrogram at the point where `n_clusters` remain.
//!
//! ## Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::cluster::agglomerative::{AgglomerativeClustering, AgglomerativeClusteringParameters};
//!
//! // A dataset with 2 distinct groups of points.
//! let x = DenseMatrix::from_2d_array(&[
//!         &[0.0, 0.0], &[1.0, 1.0], &[0.5, 0.5], // Cluster A
//!         &[10.0, 10.0], &[11.0, 11.0], &[10.5, 10.5], // Cluster B
//!     ]).unwrap();
//!
//! // Set parameters to find 2 clusters.
//! let parameters = AgglomerativeClusteringParameters::default().with_n_clusters(2);
//!
//! // Fit the model to the data.
//! let clustering = AgglomerativeClustering::<f64, usize, DenseMatrix<f64>, Vec<usize>>::fit(&x, parameters).unwrap();
//!
//! // Get the cluster assignments.
//! let labels = clustering.labels; // e.g., [0, 0, 0, 1, 1, 1]
//! ```
//!
//! ## References:
//!
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 10.3.2 Hierarchical Clustering](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["The Elements of Statistical Learning", Hastie T., Tibshirani R., Friedman J., 14.3.12 Hierarchical Clustering](https://hastie.su.domains/ElemStatLearn/)

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::api::UnsupervisedEstimator;
use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;

/// Parameters for the Agglomerative Clustering algorithm.
#[derive(Debug, Clone, Copy)]
pub struct AgglomerativeClusteringParameters {
    /// The number of clusters to find.
    pub n_clusters: usize,
}

impl AgglomerativeClusteringParameters {
    /// Sets the number of clusters.
    ///
    /// # Arguments
    /// * `n_clusters` - The desired number of clusters.
    pub fn with_n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }
}

impl Default for AgglomerativeClusteringParameters {
    fn default() -> Self {
        AgglomerativeClusteringParameters { n_clusters: 2 }
    }
}

/// Agglomerative Clustering model.
///
/// This implementation uses single-linkage clustering, which is mathematically
/// equivalent to finding the Minimum Spanning Tree (MST) of the data points.
/// The core logic is an efficient implementation of Kruskal's algorithm, which
/// processes all pairwise distances in increasing order and uses a Disjoint
/// Set Union (DSU) data structure to track cluster membership.
#[derive(Debug)]
pub struct AgglomerativeClustering<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    /// The cluster label assigned to each sample.
    pub labels: Vec<usize>,
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>> AgglomerativeClustering<TX, TY, X, Y> {
    /// Fits the agglomerative clustering model to the data.
    ///
    /// # Arguments
    /// * `data` - A reference to the input data matrix.
    /// * `parameters` - The parameters for the clustering algorithm, including `n_clusters`.
    ///
    /// # Returns
    /// A `Result` containing the fitted model with cluster labels, or an error if
    pub fn fit(data: &X, parameters: AgglomerativeClusteringParameters) -> Result<Self, Failed> {
        let (num_samples, _) = data.shape();
        let n_clusters = parameters.n_clusters;
        if n_clusters > num_samples {
            return Err(Failed::because(
                FailedError::ParametersError,
                &format!(
                    "n_clusters: {n_clusters} cannot be greater than n_samples: {num_samples}"
                ),
            ));
        }

        let mut distance_pairs = Vec::new();
        for i in 0..num_samples {
            for j in (i + 1)..num_samples {
                let distance: f64 = data
                    .get_row(i)
                    .iterator(0)
                    .zip(data.get_row(j).iterator(0))
                    .map(|(&a, &b)| (a.to_f64().unwrap() - b.to_f64().unwrap()).powi(2))
                    .sum::<f64>();

                distance_pairs.push((distance, i, j));
            }
        }
        distance_pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut parent = HashMap::new();
        let mut children = HashMap::new();
        for i in 0..num_samples {
            parent.insert(i, i);
            children.insert(i, vec![i]);
        }

        let mut merge_history = Vec::new();
        let num_merges_needed = num_samples - 1;

        while merge_history.len() < num_merges_needed {
            let (_, p1, p2) = distance_pairs.pop().unwrap();

            let root1 = parent[&p1];
            let root2 = parent[&p2];

            if root1 != root2 {
                let root2_children = children.remove(&root2).unwrap();
                for child in root2_children.iter() {
                    parent.insert(*child, root1);
                }
                let root1_children = children.get_mut(&root1).unwrap();
                root1_children.extend(root2_children);
                merge_history.push((root1, root2));
            }
        }

        let mut clusters = HashMap::new();
        let mut assignments = HashMap::new();

        for i in 0..num_samples {
            clusters.insert(i, vec![i]);
            assignments.insert(i, i);
        }

        let merges_to_apply = num_samples - n_clusters;

        for (root1, root2) in merge_history[0..merges_to_apply].iter() {
            let root1_cluster = assignments[root1];
            let root2_cluster = assignments[root2];

            let root2_assignments = clusters.remove(&root2_cluster).unwrap();
            for assignment in root2_assignments.iter() {
                assignments.insert(*assignment, root1_cluster);
            }
            let root1_assignments = clusters.get_mut(&root1_cluster).unwrap();
            root1_assignments.extend(root2_assignments);
        }

        let mut labels: Vec<usize> = (0..num_samples).map(|_| 0).collect();
        let mut cluster_keys: Vec<&usize> = clusters.keys().collect();
        cluster_keys.sort();
        for (i, key) in cluster_keys.into_iter().enumerate() {
            for index in clusters[key].iter() {
                labels[*index] = i;
            }
        }
        Ok(AgglomerativeClustering {
            labels,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        })
    }
}

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    UnsupervisedEstimator<X, AgglomerativeClusteringParameters>
    for AgglomerativeClustering<TX, TY, X, Y>
{
    fn fit(x: &X, parameters: AgglomerativeClusteringParameters) -> Result<Self, Failed> {
        AgglomerativeClustering::fit(x, parameters)
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::basic::matrix::DenseMatrix;
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_simple_clustering() {
        // Two distinct clusters, far apart.
        let data = vec![
            0.0, 0.0, 1.0, 1.0, 0.5, 0.5, // Cluster A
            10.0, 10.0, 11.0, 11.0, 10.5, 10.5, // Cluster B
        ];
        let matrix = DenseMatrix::new(6, 2, data, false).unwrap();
        let parameters = AgglomerativeClusteringParameters::default().with_n_clusters(2);
        // Using f64 for TY as usize doesn't satisfy the Number trait bound.
        let clustering = AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(
            &matrix, parameters,
        )
        .unwrap();

        let labels = clustering.labels;

        // Check that all points in the first group have the same label.
        let first_group_label = labels[0];
        assert!(labels[0..3].iter().all(|&l| l == first_group_label));

        // Check that all points in the second group have the same label.
        let second_group_label = labels[3];
        assert!(labels[3..6].iter().all(|&l| l == second_group_label));

        // Check that the two groups have different labels.
        assert_ne!(first_group_label, second_group_label);
    }

    #[test]
    fn test_four_clusters() {
        // Four distinct clusters in the corners of a square.
        let data = vec![
            0.0, 0.0, 1.0, 1.0, // Cluster A
            100.0, 100.0, 101.0, 101.0, // Cluster B
            0.0, 100.0, 1.0, 101.0, // Cluster C
            100.0, 0.0, 101.0, 1.0, // Cluster D
        ];
        let matrix = DenseMatrix::new(8, 2, data, false).unwrap();
        let parameters = AgglomerativeClusteringParameters::default().with_n_clusters(4);
        let clustering = AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(
            &matrix, parameters,
        )
        .unwrap();

        let labels = clustering.labels;

        // Verify that there are exactly 4 unique labels produced.
        let unique_labels: HashSet<usize> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 4);

        // Verify that points within each original group were assigned the same cluster label.
        let label_a = labels[0];
        assert_eq!(label_a, labels[1]);

        let label_b = labels[2];
        assert_eq!(label_b, labels[3]);

        let label_c = labels[4];
        assert_eq!(label_c, labels[5]);

        let label_d = labels[6];
        assert_eq!(label_d, labels[7]);

        // Verify that all four groups received different labels.
        assert_ne!(label_a, label_b);
        assert_ne!(label_a, label_c);
        assert_ne!(label_a, label_d);
        assert_ne!(label_b, label_c);
        assert_ne!(label_b, label_d);
        assert_ne!(label_c, label_d);
    }

    #[test]
    fn test_n_clusters_equal_to_samples() {
        let data = vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0];
        let matrix = DenseMatrix::new(3, 2, data, false).unwrap();
        let parameters = AgglomerativeClusteringParameters::default().with_n_clusters(3);
        let clustering = AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(
            &matrix, parameters,
        )
        .unwrap();

        // Each point should be its own cluster. Sorting makes the test deterministic.
        let mut labels = clustering.labels;
        labels.sort();
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn test_one_cluster() {
        let data = vec![0.0, 0.0, 5.0, 5.0, 10.0, 10.0];
        let matrix = DenseMatrix::new(3, 2, data, false).unwrap();
        let parameters = AgglomerativeClusteringParameters::default().with_n_clusters(1);
        let clustering = AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(
            &matrix, parameters,
        )
        .unwrap();

        // All points should be in the same cluster.
        assert_eq!(clustering.labels, vec![0, 0, 0]);
    }

    #[test]
    fn test_error_on_too_many_clusters() {
        let data = vec![0.0, 0.0, 5.0, 5.0];
        let matrix = DenseMatrix::new(2, 2, data, false).unwrap();
        let parameters = AgglomerativeClusteringParameters::default().with_n_clusters(3);
        let result = AgglomerativeClustering::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(
            &matrix, parameters,
        );

        assert!(result.is_err());
    }
}
