//! Integration test: KNN models end-to-end workflow.
//!
//! `KNNClassifier` / `KNNRegressor`.
//! Tracking issue: #397 / #391.
//!
//! API notes:
//!   - `from_iterator()` comes from `Array2` trait; import it where used

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
// KNNClassifier — inline fixture
// ---------------------------------------------------------------------------

#[test]
fn knn_classifier_inline_workflow() {
    use smartcore::algorithm::neighbour::KNNAlgorithmName;
    use smartcore::neighbors::KNNWeightFunction;
    use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 2.0],
        &[1.5, 2.5],
        &[2.0, 3.0],
        &[8.0, 8.0],
        &[8.5, 8.5],
        &[9.0, 9.0],
    ])
    .unwrap();
    let y: Vec<u32> = vec![0, 0, 0, 1, 1, 1];

    let params = KNNClassifierParameters::default()
        .with_k(3)
        .with_algorithm(KNNAlgorithmName::LinearSearch)
        .with_weight(KNNWeightFunction::Uniform);
    let model = KNNClassifier::fit(&x, &y, params).expect("KNNClassifier::fit");
    let preds = model.predict(&x).expect("predict");

    let acc = accuracy(&preds, &y);
    assert!(acc >= 0.833, "KNNClassifier accuracy: {acc:.3}");
}

// ---------------------------------------------------------------------------
// KNNRegressor — inline fixture
// ---------------------------------------------------------------------------

#[test]
fn knn_regressor_inline_workflow() {
    use smartcore::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};

    let x = DenseMatrix::from_2d_array(&[&[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0]]).unwrap();
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = KNNRegressorParameters::default().with_k(2);
    let model = KNNRegressor::fit(&x, &y, params).expect("KNNRegressor::fit");
    let preds = model.predict(&x).expect("predict");

    let err = mae(&preds, &y);
    assert!(err < 1.0, "KNNRegressor MAE too high: {err:.4}");
}

// ---------------------------------------------------------------------------
// KNNClassifier — iris dataset
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[test]
fn knn_classifier_iris_workflow() {
    use smartcore::algorithm::neighbour::KNNAlgorithmName;
    use smartcore::dataset::iris::load_dataset;
    use smartcore::linalg::basic::arrays::Array2;
    use smartcore::neighbors::KNNWeightFunction;
    use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};

    let ds = load_dataset();
    // from_iterator is provided by the Array2 trait
    let x = DenseMatrix::from_iterator(
        ds.data.iter().map(|&v| v as f64),
        ds.num_samples,
        ds.num_features,
        0,
    );
    let y: Vec<u32> = ds.target.clone();

    let params = KNNClassifierParameters::default()
        .with_k(5)
        .with_algorithm(KNNAlgorithmName::LinearSearch)
        .with_weight(KNNWeightFunction::Uniform);
    let model = KNNClassifier::fit(&x, &y, params).expect("fit on iris");
    let preds = model.predict(&x).expect("predict");

    let acc = accuracy(&preds, &y);
    assert!(acc >= 0.90, "KNNClassifier (iris) accuracy: {acc:.3}");
}
