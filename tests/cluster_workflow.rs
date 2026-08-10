//! Integration test: clustering end-to-end workflow.
//!
//! `KMeans` and `DBSCAN` — inline fixtures and (optionally) the iris dataset.
//! Tracking issue: #397 / #391.
//!
//! API notes:
//!   - `from_iterator()` comes from `Array2` trait; import it where used

use smartcore::linalg::basic::matrix::DenseMatrix;

// ---------------------------------------------------------------------------
// KMeans — inline fixture
// ---------------------------------------------------------------------------

#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn kmeans_inline_workflow() {
    use smartcore::cluster::kmeans::{KMeans, KMeansParameters};

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 1.0],
        &[1.1, 1.2],
        &[0.9, 0.8],
        &[1.0, 1.1],
        &[10.0, 10.0],
        &[10.1, 9.9],
        &[9.9, 10.1],
        &[10.0, 9.8],
    ])
    .unwrap();

    let params = KMeansParameters::default().with_k(2);
    let model = KMeans::fit(&x, params).expect("KMeans::fit");
    let labels: Vec<usize> = model.predict(&x).expect("predict");

    let cluster_a: std::collections::HashSet<usize> = labels[..4].iter().cloned().collect();
    let cluster_b: std::collections::HashSet<usize> = labels[4..].iter().cloned().collect();
    assert_eq!(cluster_a.len(), 1, "first 4 points should share a cluster");
    assert_eq!(cluster_b.len(), 1, "last 4 points should share a cluster");
    assert_ne!(
        cluster_a.iter().next(),
        cluster_b.iter().next(),
        "the two groups should be in different clusters"
    );
}

// ---------------------------------------------------------------------------
// DBSCAN — inline fixture
// ---------------------------------------------------------------------------

#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn dbscan_inline_workflow() {
    use smartcore::cluster::dbscan::{DBSCAN, DBSCANParameters};
    use smartcore::metrics::distance::Distances;

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 1.0],
        &[1.1, 1.2],
        &[0.9, 0.8],
        &[1.0, 1.1],
        &[10.0, 10.0],
        &[10.1, 9.9],
        &[9.9, 10.1],
        &[10.0, 9.8],
    ])
    .unwrap();

    let params = DBSCANParameters::default()
        .with_min_samples(2)
        .with_eps(0.5)
        .with_distance(Distances::euclidian());
    let labels: Vec<i32> = DBSCAN::fit(&x, params)
        .and_then(|m| m.predict(&x))
        .expect("DBSCAN::fit_predict");

    assert!(
        labels.iter().all(|&l| l > 0),
        "unexpected noise points: {labels:?}"
    );
    let cluster_a: std::collections::HashSet<i32> = labels[..4].iter().cloned().collect();
    let cluster_b: std::collections::HashSet<i32> = labels[4..].iter().cloned().collect();
    assert_eq!(cluster_a.len(), 1);
    assert_eq!(cluster_b.len(), 1);
    assert_ne!(cluster_a.iter().next(), cluster_b.iter().next());
}

// ---------------------------------------------------------------------------
// KMeans on iris dataset (datasets feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn kmeans_iris_workflow() {
    use smartcore::cluster::kmeans::{KMeans, KMeansParameters};
    use smartcore::dataset::iris::load_dataset;
    use smartcore::linalg::basic::arrays::Array2;

    let ds = load_dataset();
    let x = DenseMatrix::from_iterator(ds.data.iter().copied(), ds.num_samples, ds.num_features, 0);

    let params = KMeansParameters::default().with_k(3);
    let model = KMeans::fit(&x, params).expect("KMeans::fit on iris");
    let labels: Vec<usize> = model.predict(&x).expect("predict");

    let unique: std::collections::HashSet<usize> = labels.iter().cloned().collect();
    assert_eq!(unique.len(), 3, "expected 3 clusters, got {}", unique.len());
}
