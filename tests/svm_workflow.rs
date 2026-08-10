//! Integration test: SVM models end-to-end workflow.
//!
//! `SVC` with RBF and linear kernels, `SVR` with RBF kernel.
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
// SVC with RBF kernel
// ---------------------------------------------------------------------------

#[test]
fn svc_rbf_inline_workflow() {
    use smartcore::svm::{
        Kernels,
        svc::{SVC, SVCParameters},
    };

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 1.0],
        &[2.0, 1.5],
        &[1.5, 2.0],
        &[8.0, 8.0],
        &[9.0, 8.5],
        &[8.5, 9.0],
    ])
    .unwrap();
    let y: Vec<i32> = vec![-1, -1, -1, 1, 1, 1];

    let params = SVCParameters::default()
        .with_c(1.0)
        .with_kernel(Kernels::rbf(0.5));
    let model = SVC::fit(&x, &y, params).expect("SVC (RBF)::fit");
    let preds = model.predict(&x).expect("predict");

    let acc = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count() as f64 / y.len() as f64;
    assert!(acc >= 0.833, "SVC (RBF) accuracy: {acc:.3}");
}

// ---------------------------------------------------------------------------
// SVC with linear kernel
// ---------------------------------------------------------------------------

#[test]
fn svc_linear_inline_workflow() {
    use smartcore::svm::{
        Kernels,
        svc::{SVC, SVCParameters},
    };

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 1.0],
        &[2.0, 1.5],
        &[1.5, 2.0],
        &[2.5, 2.0],
        &[8.0, 8.0],
        &[9.0, 8.5],
        &[8.5, 9.0],
        &[9.5, 9.0],
    ])
    .unwrap();
    let y: Vec<i32> = vec![-1, -1, -1, -1, 1, 1, 1, 1];

    let params = SVCParameters::default()
        .with_c(1.0)
        .with_kernel(Kernels::linear());
    let model = SVC::fit(&x, &y, params).expect("SVC (linear)::fit");
    let preds = model.predict(&x).expect("predict");

    let acc = preds.iter().zip(y.iter()).filter(|(p, a)| p == a).count() as f64 / y.len() as f64;
    assert!(acc >= 0.875, "SVC (linear) accuracy: {acc:.3}");
}

// ---------------------------------------------------------------------------
// SVR with RBF kernel
// ---------------------------------------------------------------------------

#[test]
fn svr_rbf_inline_workflow() {
    use smartcore::svm::{
        Kernels,
        svr::{SVR, SVRParameters},
    };

    let x = DenseMatrix::from_2d_array(&[&[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0]]).unwrap();
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let params = SVRParameters::default()
        .with_c(10.0)
        .with_eps(0.1)
        .with_kernel(Kernels::rbf(1.0));
    let model = SVR::fit(&x, &y, params).expect("SVR (RBF)::fit");
    let preds = model.predict(&x).expect("predict");

    let err = mae(&preds, &y);
    assert!(err < 1.0, "SVR (RBF) MAE too high: {err:.4}");
}
