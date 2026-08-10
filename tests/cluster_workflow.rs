//! Integration test: clustering models end-to-end workflow.
//!
//! `KMeans` / `DBSCAN`.
//! Tracking issue: #397 / #391.

use smartcore::linalg::basic::matrix::DenseMatrix;

// ---------------------------------------------------------------------------
// KMeans — inline blobs fixture
// ---------------------------------------------------------------------------

#[test]
fn kmeans_inline_workflow() {
    use smartcore::cluster::kmeans::{KMeans, KMeansParameters};

    // Two well-separated clusters
    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 1.0],
        &[1.1, 1.2],
        &[0.9, 1.1],
        &[1.2, 0.9],
        &[9.0, 9.0],
        &[9.1, 9.2],
        &[8.9, 9.1],
        &[9.2, 8.9],
    ])
    .unwrap();

    let params = KMeansParameters::default().with_k(2);
    let model = KMeans::fit(&x, params).expect("KMeans::fit");
    let labels: Vec<usize> = model.predict(&x).expect("predict");

    // Both clusters should be pure: all first-half the same label,
    // all second-half a different label.
    let cluster_a: std::collections::HashSet<usize> = labels[..4].iter().cloned().collect();
    let cluster_b: std::collections::HashSet<usize> = labels[4..].iter().cloned().collect();
    assert_eq!(cluster_a.len(), 1, "first cluster not pure");
    assert_eq!(cluster_b.len(), 1, "second cluster not pure");
    assert_ne!(
        labels[0], labels[4],
        "both halves assigned to the same cluster"
    );
}

// ---------------------------------------------------------------------------
// DBSCAN — inline fixture
// ---------------------------------------------------------------------------

#[test]
fn dbscan_inline_workflow() {
    use smartcore::cluster::dbscan::{DBSCAN, DBSCANParameters};

    // Two dense clusters + one noise point far away
    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 1.0],
        &[1.1, 1.0],
        &[1.0, 1.1], // cluster A
        &[8.0, 8.0],
        &[8.1, 8.0],
        &[8.0, 8.1],   // cluster B
        &[50.0, 50.0], // noise
    ])
    .unwrap();

    // DBSCANParameters has no ::new(); use the builder pattern.
    // Default distance is Euclidean, so only eps and min_samples need setting.
    let params = DBSCANParameters::default()
        .with_min_samples(2)
        .with_eps(0.5);

    // No fit_predict(); chain fit then predict.
    let labels: Vec<i32> = DBSCAN::fit(&x, params)
        .and_then(|m| m.predict(&x))
        .expect("DBSCAN fit+predict");

    // DBSCAN assigns noise = 0; real clusters start at 1.
    // The two clusters should produce exactly 2 distinct non-zero labels.
    let noise_label = 0i32;
    let non_noise: std::collections::HashSet<i32> = labels[..6]
        .iter()
        .cloned()
        .filter(|&l| l != noise_label)
        .collect();
    assert_eq!(
        non_noise.len(),
        2,
        "expected 2 DBSCAN clusters, got: {non_noise:?}"
    );

    // The far-away point should be noise (label 0)
    assert_eq!(labels[6], noise_label, "expected noise point at index 6");
}

// ---------------------------------------------------------------------------
// KMeans — make_blobs generator (datasets feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[test]
fn kmeans_generated_blobs_workflow() {
    use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
    use smartcore::dataset::generator::make_blobs;

    let ds = make_blobs(120, 2, 3);
    let x = DenseMatrix::from_iterator(ds.data.iter().copied(), ds.num_samples, ds.num_features, 0);

    let params = KMeansParameters::default().with_k(3);
    let model = KMeans::fit(&x, params).expect("KMeans fit on blobs");
    let labels: Vec<usize> = model.predict(&x).expect("predict");

    let unique: std::collections::HashSet<usize> = labels.iter().cloned().collect();
    assert_eq!(unique.len(), 3, "expected 3 KMeans clusters on 3-blob data");
}
