//! Integration test: decision tree models end-to-end workflow.
//!
//! `DecisionTreeClassifier` / `DecisionTreeRegressor`.
//! Tracking issue: #397 / #391.

use smartcore::linalg::basic::matrix::DenseMatrix;

fn accuracy(predicted: &[u32], actual: &[u32]) -> f64 {
    predicted
        .iter()
        .zip(actual.iter())
        .filter(|(p, a)| p == a)
        .count() as f64
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
// DecisionTreeClassifier — inline XOR fixture
// ---------------------------------------------------------------------------

#[test]
fn decision_tree_classifier_inline_workflow() {
    use smartcore::tree::decision_tree_classifier::{
        DecisionTreeClassifier, DecisionTreeClassifierParameters,
    };

    let x = DenseMatrix::from_2d_array(&[
        &[0.0_f64, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 0.0],
        &[1.0, 1.0],
    ])
    .unwrap();
    // XOR: class 1 when exactly one feature is 1
    let y: Vec<u32> = vec![0, 1, 1, 0, 0, 1, 1, 0];

    let params = DecisionTreeClassifierParameters::default().with_max_depth(4);
    let model = DecisionTreeClassifier::fit(&x, &y, params).expect("DecisionTreeClassifier::fit");
    let preds = model.predict(&x).expect("predict");

    let acc = accuracy(&preds, &y);
    assert!(
        acc >= 0.875,
        "DecisionTreeClassifier XOR accuracy: {acc:.3}"
    );
}

// ---------------------------------------------------------------------------
// DecisionTreeRegressor — inline quadratic fixture
// ---------------------------------------------------------------------------

#[test]
fn decision_tree_regressor_inline_workflow() {
    use smartcore::tree::decision_tree_regressor::{
        DecisionTreeRegressor, DecisionTreeRegressorParameters,
    };

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64],
        &[2.0],
        &[3.0],
        &[4.0],
        &[5.0],
        &[6.0],
        &[7.0],
        &[8.0],
    ])
    .unwrap();
    let y: Vec<f64> = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]; // x^2

    let params = DecisionTreeRegressorParameters::default().with_max_depth(4);
    let model = DecisionTreeRegressor::fit(&x, &y, params).expect("DecisionTreeRegressor::fit");
    let preds = model.predict(&x).expect("predict");

    let err = mae(&preds, &y);
    assert!(err < 5.0, "DecisionTreeRegressor MAE too high: {err:.4}");
}

// ---------------------------------------------------------------------------
// DecisionTreeClassifier — iris dataset
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[test]
fn decision_tree_classifier_iris_workflow() {
    use smartcore::dataset::iris::load_dataset;
    use smartcore::tree::decision_tree_classifier::{
        DecisionTreeClassifier, DecisionTreeClassifierParameters,
    };

    let ds = load_dataset();
    let x = DenseMatrix::from_iterator(
        ds.data.iter().map(|&v| v as f64),
        ds.num_samples,
        ds.num_features,
        0,
    );
    let y: Vec<u32> = ds.target.clone();

    let params = DecisionTreeClassifierParameters::default().with_max_depth(5);
    let model = DecisionTreeClassifier::fit(&x, &y, params).expect("fit on iris");
    let preds = model.predict(&x).expect("predict");

    let acc = accuracy(&preds, &y);
    // A depth-5 tree on 150 iris samples should easily exceed 90 %
    assert!(
        acc >= 0.90,
        "DecisionTreeClassifier (iris) accuracy: {acc:.3}"
    );
}
