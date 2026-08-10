//! Integration test: ensemble models end-to-end workflow.
//!
//! `RandomForestClassifier` / `RandomForestRegressor`.
//! Tracking issue: #397 / #391.

use smartcore::linalg::basic::matrix::DenseMatrix;

fn accuracy(predicted: &[u32], actual: &[u32]) -> f64 {
    predicted.iter().zip(actual.iter()).filter(|(p, a)| p == a).count() as f64
        / actual.len() as f64
}

fn mae(predicted: &[f64], actual: &[f64]) -> f64 {
    predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / actual.len() as f64
}

// ---------------------------------------------------------------------------
// RandomForestClassifier — inline fixture
// ---------------------------------------------------------------------------

#[test]
fn random_forest_classifier_inline_workflow() {
    use smartcore::ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    };

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 2.0], &[2.0, 3.0], &[3.0, 4.0], &[4.0, 5.0],
        &[10.0, 11.0], &[11.0, 12.0], &[12.0, 13.0], &[13.0, 14.0],
    ])
    .unwrap();
    let y: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1];

    let params = RandomForestClassifierParameters::default()
        .with_n_trees(10)
        .with_max_depth(3);
    let model = RandomForestClassifier::fit(&x, &y, params)
        .expect("RandomForestClassifier::fit");
    let preds = model.predict(&x).expect("predict");

    let acc = accuracy(&preds, &y);
    assert!(acc >= 0.875, "RandomForestClassifier accuracy: {acc:.3}");
}

// ---------------------------------------------------------------------------
// RandomForestRegressor — inline fixture
// ---------------------------------------------------------------------------

#[test]
fn random_forest_regressor_inline_workflow() {
    use smartcore::ensemble::random_forest_regressor::{
        RandomForestRegressor, RandomForestRegressorParameters,
    };

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0],
        &[6.0], &[7.0], &[8.0],
    ])
    .unwrap();
    let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

    let params = RandomForestRegressorParameters::default()
        .with_n_trees(10)
        .with_max_depth(4);
    let model = RandomForestRegressor::fit(&x, &y, params)
        .expect("RandomForestRegressor::fit");
    let preds = model.predict(&x).expect("predict");

    let err = mae(&preds, &y);
    assert!(err < 2.0, "RandomForestRegressor MAE too high: {err:.4}");
}

// ---------------------------------------------------------------------------
// RandomForestClassifier — iris dataset
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[test]
fn random_forest_classifier_iris_workflow() {
    use smartcore::dataset::iris::load_dataset;
    use smartcore::ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    };

    let ds = load_dataset();
    let x = DenseMatrix::from_iterator(
        ds.data.iter().map(|&v| v as f64),
        ds.num_samples,
        ds.num_features,
        0,
    );
    let y: Vec<u32> = ds.target.clone();

    let params = RandomForestClassifierParameters::default()
        .with_n_trees(20)
        .with_max_depth(5);
    let model = RandomForestClassifier::fit(&x, &y, params).expect("fit on iris");
    let preds = model.predict(&x).expect("predict");

    let acc = accuracy(&preds, &y);
    assert!(acc >= 0.90, "RandomForestClassifier (iris) accuracy: {acc:.3}");
}
