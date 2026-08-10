//! Integration test: linear models end-to-end workflow.
//!
//! `LinearRegression`, `RidgeRegression`, `LogisticRegression` —
//! load (or inline) data → train → predict → evaluate with non-trivial thresholds.
//!
//! Dataset-dependent paths are gated on `#[cfg(feature = "datasets")]`;
//! a tiny inline fixture is used for the no-feature path.
//!
//! Tracking issue: #397 / #391.
//!
//! API notes:
//!   - `from_iterator()` comes from `Array2` trait; import it where used

use smartcore::linalg::basic::matrix::DenseMatrix;

fn accuracy(predicted: &[u32], actual: &[u32]) -> f64 {
    assert_eq!(predicted.len(), actual.len());
    let correct = predicted
        .iter()
        .zip(actual.iter())
        .filter(|(p, a)| p == a)
        .count();
    correct as f64 / actual.len() as f64
}

fn mae(predicted: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(predicted.len(), actual.len());
    predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / actual.len() as f64
}

// ---------------------------------------------------------------------------
// LinearRegression — inline fixture
// ---------------------------------------------------------------------------

#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn linear_regression_inline_workflow() {
    use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters};

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64],
        &[2.0],
        &[3.0],
        &[4.0],
        &[5.0],
        &[6.0],
        &[7.0],
        &[8.0],
        &[9.0],
        &[10.0],
    ])
    .unwrap();
    let y: Vec<f64> = (1..=10).map(|i| 3.0 * i as f64 + 1.0).collect();

    let model = LinearRegression::fit(&x, &y, LinearRegressionParameters::default())
        .expect("LinearRegression::fit");
    let preds = model.predict(&x).expect("LinearRegression::predict");

    let err = mae(&preds, &y);
    assert!(err < 0.5, "LinearRegression MAE too high: {err:.4}");
}

// ---------------------------------------------------------------------------
// RidgeRegression — inline fixture
// ---------------------------------------------------------------------------

#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn ridge_regression_inline_workflow() {
    use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 1.0],
        &[2.0, 4.0],
        &[3.0, 9.0],
        &[4.0, 16.0],
        &[5.0, 25.0],
        &[6.0, 36.0],
    ])
    .unwrap();
    let y: Vec<f64> = vec![2.0, 5.0, 10.0, 17.0, 26.0, 37.0];

    let params = RidgeRegressionParameters::default().with_alpha(0.1);
    let model = RidgeRegression::fit(&x, &y, params).expect("RidgeRegression::fit");
    let preds = model.predict(&x).expect("RidgeRegression::predict");

    let err = mae(&preds, &y);
    assert!(err < 2.0, "RidgeRegression MAE too high: {err:.4}");
}

// ---------------------------------------------------------------------------
// LogisticRegression — inline binary fixture
// ---------------------------------------------------------------------------

#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn logistic_regression_inline_workflow() {
    use smartcore::linear::logistic_regression::{
        LogisticRegression, LogisticRegressionParameters,
    };

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 1.0],
        &[2.0, 1.5],
        &[1.5, 2.0],
        &[2.5, 2.5],
        &[8.0, 8.0],
        &[9.0, 8.5],
        &[8.5, 9.0],
        &[9.5, 9.5],
    ])
    .unwrap();
    let y: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1];

    let model = LogisticRegression::fit(&x, &y, LogisticRegressionParameters::default())
        .expect("LogisticRegression::fit");
    let preds = model.predict(&x).expect("LogisticRegression::predict");

    let acc = accuracy(&preds, &y);
    assert!(
        acc >= 0.875,
        "LogisticRegression accuracy too low: {acc:.3}"
    );
}

// ---------------------------------------------------------------------------
// LinearRegression on iris dataset (dataset feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[cfg_attr(
    all(target_arch = "wasm32", not(target_os = "wasi")),
    wasm_bindgen_test::wasm_bindgen_test
)]
#[test]
fn linear_regression_iris_sepal_workflow() {
    use smartcore::dataset::iris::load_dataset;
    use smartcore::linalg::basic::arrays::Array2;
    use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters};

    let ds = load_dataset();
    let x_f64: DenseMatrix<f64> = DenseMatrix::from_iterator(
        ds.data
            .chunks(ds.num_features)
            .flat_map(|row| row[..2].iter().map(|&v| v as f64)),
        ds.num_samples,
        2,
        0,
    );
    let petal_len: Vec<f64> = ds
        .data
        .chunks(ds.num_features)
        .map(|row| row[2] as f64)
        .collect();

    let model = LinearRegression::fit(&x_f64, &petal_len, LinearRegressionParameters::default())
        .expect("fit on iris");
    let preds = model.predict(&x_f64).expect("predict on iris");
    let err = mae(&preds, &petal_len);
    assert!(err < 0.7, "LinearRegression (iris) MAE too high: {err:.4}");
}
