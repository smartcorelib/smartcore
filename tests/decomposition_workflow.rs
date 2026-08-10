//! Integration test: decomposition end-to-end workflow.
//!
//! `PCA` fit → transform → shape/column checks;
//! `SVD` singular-value ordering.
//! Tracking issue: #397 / #391.
//!
//! API notes:
//!   - `shape()` comes from `smartcore::linalg::basic::arrays::Array` which must
//!     be in scope wherever `.shape()` is called on a `DenseMatrix`
//!   - `from_iterator()` comes from `smartcore::linalg::basic::arrays::Array2`
//!   - PCA::fit / PCA::transform are inherent methods; no trait import needed

use smartcore::linalg::basic::matrix::DenseMatrix;

// ---------------------------------------------------------------------------
// PCA — full-rank (no information loss)
// ---------------------------------------------------------------------------

#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn pca_full_rank_workflow() {
    use smartcore::decomposition::pca::{PCA, PCAParameters};
    use smartcore::linalg::basic::arrays::Array;

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
        &[7.0, 8.0, 9.0],
        &[2.0, 4.0, 6.0],
        &[3.0, 6.0, 9.0],
    ])
    .unwrap();

    let params = PCAParameters::default().with_n_components(3);
    let model = PCA::fit(&x, params).expect("PCA::fit");
    let transformed = model.transform(&x).expect("transform");
    assert_eq!(transformed.shape(), (5, 3), "PCA full-rank output shape");
}

// ---------------------------------------------------------------------------
// PCA — dimensionality reduction
// ---------------------------------------------------------------------------

#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn pca_reduce_and_reconstruct_workflow() {
    use smartcore::decomposition::pca::{PCA, PCAParameters};
    use smartcore::linalg::basic::arrays::Array;

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 2.0, 3.0],
        &[2.0, 4.0, 6.0],
        &[3.0, 6.0, 9.0],
        &[4.0, 8.0, 12.0],
        &[5.0, 10.0, 15.0],
    ])
    .unwrap();

    let params = PCAParameters::default().with_n_components(1);
    let model = PCA::fit(&x, params).expect("PCA::fit (rank-1)");
    let transformed = model.transform(&x).expect("transform");
    assert_eq!(transformed.shape().1, 1, "expected 1 output column");
}

// ---------------------------------------------------------------------------
// SVD decomposition — singular values non-negative and decreasing
// (SVD is not available on wasm32)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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
    let s = &svd.s;
    for i in 0..s.len() {
        assert!(s[i] >= -1e-10, "negative singular value at {i}: {}", s[i]);
    }
    for i in 1..s.len() {
        assert!(
            s[i - 1] >= s[i] - 1e-10,
            "singular values not descending: s[{}]={} > s[{}]={}",
            i - 1,
            s[i - 1],
            i,
            s[i]
        );
    }
}

// ---------------------------------------------------------------------------
// PCA on iris dataset (datasets feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn pca_iris_reduce_workflow() {
    use smartcore::dataset::iris::load_dataset;
    use smartcore::decomposition::pca::{PCA, PCAParameters};
    use smartcore::linalg::basic::arrays::Array;
    use smartcore::linalg::basic::arrays::Array2;

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
