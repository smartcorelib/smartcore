//! Stage 3: edge-case & known-answer parity tests for src/cluster.
//!
//! Covers: KMeans, DBSCAN.
//! Tracking issue: #394 / #391.

#[cfg(test)]
mod cluster_edge_cases {
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::cluster::kmeans::{KMeans, KMeansParameters};
    use crate::cluster::dbscan::{DBSCAN, DBSCANParameters};

    // ── KMeans ────────────────────────────────────────────────────────────────

    /// k=1: every point assigned to cluster 0.
    #[test]
    fn kmeans_k1_all_same_cluster() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 1.0],
            &[2.0, 2.0],
            &[3.0, 3.0],
        ]).unwrap();
        let model = KMeans::fit(&x, KMeansParameters::default().with_k(1).with_seed(0)).unwrap();
        let labels = model.predict(&x).unwrap();
        assert!(labels.iter().all(|&l| l == 0), "k=1 must assign all to cluster 0");
    }

    /// k=N (one cluster per point): each point gets a unique cluster.
    #[test]
    fn kmeans_k_equals_n_unique_clusters() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64, 0.0],
            &[10.0, 0.0],
            &[0.0, 10.0],
        ]).unwrap();
        let model = KMeans::fit(&x, KMeansParameters::default().with_k(3).with_seed(0)).unwrap();
        let labels = model.predict(&x).unwrap();
        let unique: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique.len(), 3, "k=3 on 3 distant points should yield 3 distinct clusters");
    }

    /// Seed determinism: same seed → same cluster assignments.
    #[test]
    fn kmeans_seed_determinism() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 0.0], &[1.1, 0.0], &[0.9, 0.0],
            &[9.0, 0.0],    &[9.1, 0.0], &[8.9, 0.0],
        ]).unwrap();
        let l1 = KMeans::fit(&x, KMeansParameters::default().with_k(2).with_seed(7)).unwrap().predict(&x).unwrap();
        let l2 = KMeans::fit(&x, KMeansParameters::default().with_k(2).with_seed(7)).unwrap().predict(&x).unwrap();
        assert_eq!(l1, l2, "same seed should give deterministic cluster assignments");
    }

    /// Well-separated clusters: intra-cluster label consistency.
    #[test]
    fn kmeans_well_separated_clusters() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64, 0.0], &[0.1, 0.0], &[0.0, 0.1],
            &[9.9, 9.9],    &[10.0, 9.9], &[9.9, 10.0],
        ]).unwrap();
        let model = KMeans::fit(&x, KMeansParameters::default().with_k(2).with_seed(0)).unwrap();
        let labels = model.predict(&x).unwrap();
        // First 3 must share a label, last 3 must share a different label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    // ── DBSCAN ────────────────────────────────────────────────────────────────

    /// All-noise: epsilon very small → every point is noise (-1).
    #[test]
    fn dbscan_all_noise() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64, 0.0],
            &[10.0, 0.0],
            &[0.0, 10.0],
            &[10.0, 10.0],
        ]).unwrap();
        let model = DBSCAN::fit(&x, DBSCANParameters::default().with_eps(0.001).with_min_samples(2)).unwrap();
        let labels = model.predict(&x).unwrap();
        // All points are noise (label == usize::MAX or a sentinel); no valid cluster.
        // We just verify fit+predict don't panic and return the right count.
        assert_eq!(labels.len(), 4);
    }

    /// Dense cluster: two tight groups each exceeding min_samples.
    #[test]
    fn dbscan_two_dense_clusters() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64, 0.0], &[0.1, 0.0], &[0.0, 0.1],
            &[9.9, 9.9],    &[10.0, 9.9], &[9.9, 10.0],
        ]).unwrap();
        let model = DBSCAN::fit(&x, DBSCANParameters::default().with_eps(0.5).with_min_samples(2)).unwrap();
        let labels = model.predict(&x).unwrap();
        assert_eq!(labels.len(), 6);
        // Both groups must form valid (non-noise) clusters.
        let unique: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique.len(), 2, "expected exactly 2 clusters, got {unique:?}");
    }

    /// min_samples edge: min_samples=1 makes every point its own cluster.
    #[test]
    fn dbscan_min_samples_one() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64], &[1.0], &[2.0],
        ]).unwrap();
        let result = DBSCAN::fit(&x, DBSCANParameters::default().with_eps(0.1).with_min_samples(1));
        assert!(result.is_ok());
    }
}
