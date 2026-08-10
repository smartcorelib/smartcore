//! Integration test: model selection end-to-end workflow.
//!
//! `train_test_split` → fit → evaluate; `cross_validate`.
//! Tracking issue: #397 / #391.

use smartcore::linalg::basic::matrix::DenseMatrix;

fn accuracy(predicted: &[u32], actual: &[u32]) -> f64 {
    predicted.iter().zip(actual.iter()).filter(|(p, a)| p == a).count() as f64
        / actual.len() as f64
}

// ---------------------------------------------------------------------------
// train_test_split — inline fixture
// ---------------------------------------------------------------------------

#[test]
fn train_test_split_workflow() {
    use smartcore::algorithm::neighbour::KNNAlgorithmName;
    use smartcore::model_selection::train_test_split;
    use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
    use smartcore::neighbors::KNNWeightFunction;

    // 20 well-separated samples, 2 classes
    let n = 20usize;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            if i < n / 2 {
                vec![i as f64, i as f64]
            } else {
                vec![i as f64 + 100.0, i as f64 + 100.0]
            }
        })
        .collect();
    let refs: Vec<&[f64]> = data.iter().map(|r| r.as_slice()).collect();
    let x = DenseMatrix::from_2d_array(&refs).unwrap();
    let y: Vec<u32> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, 0.3, true, Some(42));

    assert!(!x_train.is_empty(), "train set empty");
    assert!(!x_test.is_empty(), "test set empty");
    assert_eq!(y_train.len() + y_test.len(), n);

    let params = KNNClassifierParameters::default()
        .with_k(3)
        .with_algorithm(KNNAlgorithmName::LinearSearch)
        .with_weight(KNNWeightFunction::Uniform);
    let model = KNNClassifier::fit(&x_train, &y_train, params).expect("fit");
    let preds = model.predict(&x_test).expect("predict");

    let acc = accuracy(&preds, &y_test);
    assert!(acc >= 0.8, "train_test_split KNN accuracy: {acc:.3}");
}

// ---------------------------------------------------------------------------
// cross_validate — inline fixture
// ---------------------------------------------------------------------------

#[test]
fn cross_validate_knn_workflow() {
    use smartcore::algorithm::neighbour::KNNAlgorithmName;
    use smartcore::model_selection::{cross_validate, CrossValidationParameters};
    use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
    use smartcore::neighbors::KNNWeightFunction;

    let n = 30usize;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            if i < n / 2 {
                vec![i as f64, 0.0]
            } else {
                vec![i as f64, 100.0]
            }
        })
        .collect();
    let refs: Vec<&[f64]> = data.iter().map(|r| r.as_slice()).collect();
    let x = DenseMatrix::from_2d_array(&refs).unwrap();
    let y: Vec<u32> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();

    let cv_params = CrossValidationParameters::default().with_n_splits(5);
    let params = KNNClassifierParameters::default()
        .with_k(3)
        .with_algorithm(KNNAlgorithmName::LinearSearch)
        .with_weight(KNNWeightFunction::Uniform);

    // Scoring function: proportion of matching labels
    let score_fn = |y_true: &Vec<u32>, y_pred: &Vec<u32>| -> f64 {
        y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(a, b)| a == b)
            .count() as f64
            / y_true.len() as f64
    };

    let result = cross_validate(
        KNNClassifier::fit,
        &x,
        &y,
        params,
        cv_params,
        score_fn,
    )
    .expect("cross_validate");

    let mean_score = result.test_score.iter().sum::<f64>() / result.test_score.len() as f64;
    assert!(mean_score >= 0.7, "cross_validate mean accuracy: {mean_score:.3}");
}

// ---------------------------------------------------------------------------
// train_test_split on iris dataset (datasets feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[test]
fn train_test_split_iris_workflow() {
    use smartcore::dataset::iris::load_dataset;
    use smartcore::model_selection::train_test_split;
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

    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, 0.2, true, Some(0));

    let params = DecisionTreeClassifierParameters::default().with_max_depth(5);
    let model = DecisionTreeClassifier::fit(&x_train, &y_train, params)
        .expect("fit iris train");
    let preds = model.predict(&x_test).expect("predict iris test");

    let acc = accuracy(&preds, &y_test);
    assert!(acc >= 0.85, "DecisionTree (iris test split) accuracy: {acc:.3}");
}
