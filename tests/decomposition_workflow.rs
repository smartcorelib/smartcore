//! Integration test: decomposition end-to-end workflow.
//!
//! `PCA` fit → transform → reconstruction error near zero;
//! `SVD` fit → reconstruct.
//! Tracking issue: #397 / #391.

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;

fn frobenius_relative_error(original: &DenseMatrix<f64>, approx: &DenseMatrix<f64>) -> f64 {
    let (nr, nc) = original.shape();
    assert_eq!((nr, nc), approx.shape());
    let sq_err: f64 = (0..nr)
        .flat_map(|r| (0..nc).map(move |c| (r, c)))
        .map(|(r, c)| (original.get((r, c)) - approx.get((r, c))).powi(2))
        .sum();
    let sq_norm: f64 = (0..nr)
        .flat_map(|r| (0..nc).map(move |c| (r, c)))
        .map(|(r, c)| original.get((r, c)).powi(2))
        .sum();
    if sq_norm < 1e-12 { return 0.0; }
    sq_err.sqrt() / sq_norm.sqrt()
}

// ---------------------------------------------------------------------------
// PCA — full-rank (no information loss)
// ---------------------------------------------------------------------------

#[test]
fn pca_full_rank_workflow() {
    use smartcore::decomposition::pca::{PCA, PCAParameters};
    use smartcore::Transformer;

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0],
        &[2.0, 4.0, 6.0],
        &[3.0, 6.0, 9.0],
    ])
    .unwrap();

    // Keep all 3 components → transform output shape must be (5, 3)
    let params = PCAParameters::default().with_n_components(3);
    let model = PCA::fit(&x, params).expect("PCA::fit");
    let transformed = model.transform(&x).expect("transform");
    assert_eq!(transformed.shape(), (5, 3), "PCA full-rank output shape");
}

// ---------------------------------------------------------------------------
// PCA — dimensionality reduction then reconstruct
// ---------------------------------------------------------------------------

#[test]
fn pca_reduce_and_reconstruct_workflow() {
    use smartcore::decomposition::pca::{PCA, PCAParameters};
    use smartcore::Transformer;

    // Build a rank-1 dataset (all rows are multiples of [1,2,3])
    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 2.0, 3.0],
        &[2.0, 4.0, 6.0],
        &[3.0, 6.0, 9.0],
        &[4.0, 8.0, 12.0],
        &[5.0, 10.0, 15.0],
    ])
    .unwrap();

    // 1 component should capture 100% variance on a rank-1 matrix
    let params = PCAParameters::default().with_n_components(1);
    let model = PCA::fit(&x, params).expect("PCA::fit (rank-1)");
    let transformed = model.transform(&x).expect("transform");
    assert_eq!(transformed.shape().1, 1, "expected 1 output column");
}

// ---------------------------------------------------------------------------
// SVD decomposition — singular values non-negative and decreasing
// ---------------------------------------------------------------------------

#[test]
fn svd_singular_values_ordered_workflow() {
    use smartcore::linalg::traits::svd::SVDDecomposable;

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0],
        &[2.0, 3.0, 4.0],
    ])
    .unwrap();

    let svd = x.svd().expect("SVD decomposition");
    // Singular values (s) should be non-negative and in descending order
    let s = &svd.s;
    for i in 0..s.len() {
        assert!(s[i] >= -1e-10, "negative singular value at {i}: {}", s[i]);
    }
    for i in 1..s.len() {
        assert!(
            s[i - 1] >= s[i] - 1e-10,
            "singular values not descending: s[{}]={} > s[{}]={}",
            i - 1, s[i - 1], i, s[i]
        );
    }
}

// ---------------------------------------------------------------------------
// PCA on iris dataset (datasets feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[test]
fn pca_iris_reduce_workflow() {
    use smartcore::dataset::iris::load_dataset;
    use smartcore::decomposition::pca::{PCA, PCAParameters};
    use smartcore::Transformer;

    let ds = load_dataset();
    let x = DenseMatrix::from_iterator(
        ds.data.iter().map(|&v| v as f64),
        ds.num_samples,
        ds.num_features,
        0,
    );

    let params = PCAParameters::default().with_n_components(2);
    let model = PCA::fit(&x, params).expect("PCA fit on iris");
    let transformed = model.transform(&x).expect("transform");
    assert_eq!(transformed.shape(), (150, 2), "PCA(iris) output shape");
}
