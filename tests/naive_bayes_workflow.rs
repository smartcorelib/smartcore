//! Integration test: Naive Bayes end-to-end workflow.
//!
//! GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB.
//! Tracking issue: #397 / #391.
//!
//! Note: ComplementNB is not implemented in smartcore; MultinomialNB is used
//! as the equivalent count-based classifier.

use smartcore::linalg::basic::matrix::DenseMatrix;

fn accuracy_u32(predicted: &[u32], actual: &[u32]) -> f64 {
    predicted
        .iter()
        .zip(actual.iter())
        .filter(|(p, a)| p == a)
        .count() as f64
        / actual.len() as f64
}

// ---------------------------------------------------------------------------
// GaussianNB
// ---------------------------------------------------------------------------

#[test]
fn gaussian_nb_inline_workflow() {
    use smartcore::naive_bayes::gaussian::{GaussianNB, GaussianNBParameters};

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 0.0],
        &[1.1, 0.1],
        &[0.9, -0.1],
        &[1.2, 0.2],
        &[-1.0, 0.0],
        &[-1.1, 0.1],
        &[-0.9, -0.1],
        &[-1.2, 0.2],
    ])
    .unwrap();
    let y: Vec<u32> = vec![0, 0, 0, 0, 1, 1, 1, 1];

    let model = GaussianNB::fit(&x, &y, GaussianNBParameters::default()).expect("GaussianNB::fit");
    let preds = model.predict(&x).expect("predict");

    assert!(
        accuracy_u32(&preds, &y) >= 0.875,
        "GaussianNB accuracy too low"
    );
}

// ---------------------------------------------------------------------------
// BernoulliNB — binary feature matrix
// ---------------------------------------------------------------------------

#[test]
fn bernoulli_nb_inline_workflow() {
    use smartcore::naive_bayes::bernoulli::{BernoulliNB, BernoulliNBParameters};

    // Binary document-term style features
    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 1.0, 0.0, 0.0],
        &[1.0, 0.0, 1.0, 0.0],
        &[1.0, 1.0, 1.0, 0.0],
        &[0.0, 0.0, 1.0, 1.0],
        &[0.0, 0.0, 0.0, 1.0],
        &[0.0, 1.0, 0.0, 1.0],
    ])
    .unwrap();
    let y: Vec<u32> = vec![0, 0, 0, 1, 1, 1];

    let model =
        BernoulliNB::fit(&x, &y, BernoulliNBParameters::default()).expect("BernoulliNB::fit");
    let preds = model.predict(&x).expect("predict");

    assert!(
        accuracy_u32(&preds, &y) >= 0.666,
        "BernoulliNB accuracy too low"
    );
}

// ---------------------------------------------------------------------------
// CategoricalNB — small integer category matrix
// ---------------------------------------------------------------------------

#[test]
fn categorical_nb_inline_workflow() {
    use smartcore::naive_bayes::categorical::{CategoricalNB, CategoricalNBParameters};

    // 3-category features stored as f64 (smartcore categorical NB uses RealNumber)
    let x = DenseMatrix::from_2d_array(&[
        &[0.0_f64, 1.0, 0.0],
        &[0.0, 0.0, 1.0],
        &[1.0, 0.0, 0.0],
        &[2.0, 1.0, 0.0],
        &[2.0, 2.0, 1.0],
        &[1.0, 2.0, 2.0],
    ])
    .unwrap();
    let y: Vec<u32> = vec![0, 0, 0, 1, 1, 1];

    let model =
        CategoricalNB::fit(&x, &y, CategoricalNBParameters::default()).expect("CategoricalNB::fit");
    let preds = model.predict(&x).expect("predict");

    assert!(
        accuracy_u32(&preds, &y) >= 0.666,
        "CategoricalNB accuracy too low"
    );
}

// ---------------------------------------------------------------------------
// MultinomialNB — count / frequency feature matrix
// ---------------------------------------------------------------------------

#[test]
fn multinomial_nb_inline_workflow() {
    use smartcore::naive_bayes::multinomial::{MultinomialNB, MultinomialNBParameters};

    let x = DenseMatrix::from_2d_array(&[
        &[3.0_f64, 1.0, 0.0],
        &[4.0, 2.0, 0.0],
        &[5.0, 0.0, 1.0],
        &[0.0, 3.0, 4.0],
        &[0.0, 4.0, 5.0],
        &[1.0, 2.0, 6.0],
    ])
    .unwrap();
    let y: Vec<u32> = vec![0, 0, 0, 1, 1, 1];

    let model =
        MultinomialNB::fit(&x, &y, MultinomialNBParameters::default()).expect("MultinomialNB::fit");
    let preds = model.predict(&x).expect("predict");

    assert!(
        accuracy_u32(&preds, &y) >= 0.666,
        "MultinomialNB accuracy too low"
    );
}

// ---------------------------------------------------------------------------
// GaussianNB on iris (dataset feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "datasets")]
#[test]
fn gaussian_nb_iris_workflow() {
    use smartcore::dataset::iris::load_dataset;
    use smartcore::naive_bayes::gaussian::{GaussianNB, GaussianNBParameters};

    let ds = load_dataset();
    let x = DenseMatrix::from_iterator(
        ds.data.iter().map(|&v| v as f64),
        ds.num_samples,
        ds.num_features,
        0,
    );
    let y: Vec<u32> = ds.target.clone();

    let model = GaussianNB::fit(&x, &y, GaussianNBParameters::default()).expect("fit on iris");
    let preds = model.predict(&x).expect("predict");

    let acc = accuracy_u32(&preds, &y);
    assert!(acc >= 0.90, "GaussianNB (iris) accuracy: {acc:.3}");
}
