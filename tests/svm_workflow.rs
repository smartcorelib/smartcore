//! Integration test: SVM end-to-end workflow.
//!
//! `SVC` (RBF kernel, linear kernel), `SVR` (RBF kernel).
//! Tracking issue: #397 / #391.
//!
//! API notes:
//!   - `Kernels::rbf()` takes 0 arguments (gamma is fixed internally)
//!   - `SVC::fit` / `SVR::fit` take params by reference (`&params`)
//!   - `SVC::predict` returns `Vec<f64>`, not `Vec<i32>`

use smartcore::linalg::basic::matrix::DenseMatrix;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn accuracy_f64(predicted: &[f64], actual: &[f64]) -> f64 {
    predicted
        .iter()
        .zip(actual.iter())
        .filter(|(p, a)| (*p - *a).abs() < 1e-9)
        .count() as f64
        / actual.len() as f64
}

// ---------------------------------------------------------------------------
// SVC — RBF kernel
// ---------------------------------------------------------------------------

#[test]
fn svc_rbf_inline_workflow() {
    use smartcore::svm::{
        Kernels,
        svc::{SVC, SVCParameters},
    };

    // Linearly separable 2-class problem
    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 0.0], &[1.1, 0.1], &[0.9, -0.1], &[1.2, 0.2],
        &[-1.0, 0.0], &[-1.1, 0.1], &[-0.9, -0.1], &[-1.2, 0.2],
    ])
    .unwrap();
    // SVC predict returns f64 labels
    let y: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];

    // Kernels::rbf() takes no arguments
    let params = SVCParameters::default()
        .with_c(1.0)
        .with_kernel(Kernels::rbf());
    // fit takes params by reference
    let model = SVC::fit(&x, &y, &params).expect("SVC (RBF)::fit");
    let preds: Vec<f64> = model.predict(&x).expect("predict");

    let acc = accuracy_f64(&preds, &y);
    assert!(acc >= 0.875, "SVC (RBF) accuracy: {acc:.3}");
}

// ---------------------------------------------------------------------------
// SVC — linear kernel
// ---------------------------------------------------------------------------

#[test]
fn svc_linear_inline_workflow() {
    use smartcore::svm::{
        Kernels,
        svc::{SVC, SVCParameters},
    };

    let x = DenseMatrix::from_2d_array(&[
        &[2.0_f64, 0.0], &[2.1, 0.1], &[1.9, -0.1], &[2.2, 0.2],
        &[-2.0, 0.0], &[-2.1, 0.1], &[-1.9, -0.1], &[-2.2, 0.2],
    ])
    .unwrap();
    let y: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];

    let params = SVCParameters::default()
        .with_c(1.0)
        .with_kernel(Kernels::linear());
    let model = SVC::fit(&x, &y, &params).expect("SVC (linear)::fit");
    let preds: Vec<f64> = model.predict(&x).expect("predict");

    let acc = accuracy_f64(&preds, &y);
    assert!(acc >= 0.875, "SVC (linear) accuracy: {acc:.3}");
}

// ---------------------------------------------------------------------------
// SVR — RBF kernel
// ---------------------------------------------------------------------------

#[test]
fn svr_rbf_inline_workflow() {
    use smartcore::svm::{
        Kernels,
        svr::{SVR, SVRParameters},
    };

    // y = x[0] + x[1]  (simple linear target, RBF should fit it well)
    let x = DenseMatrix::from_2d_array(&[
        &[0.0_f64, 0.0], &[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0],
        &[2.0, 0.0], &[0.0, 2.0], &[2.0, 2.0], &[3.0, 0.0],
    ])
    .unwrap();
    let y: Vec<f64> = vec![0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 3.0];

    // Kernels::rbf() takes no arguments
    let params = SVRParameters::default()
        .with_c(10.0)
        .with_eps(0.1)
        .with_kernel(Kernels::rbf());
    let model = SVR::fit(&x, &y, &params).expect("SVR (RBF)::fit");
    let preds: Vec<f64> = model.predict(&x).expect("predict");

    // Mean Absolute Error should be small on training data
    let mae: f64 = preds
        .iter()
        .zip(y.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / y.len() as f64;
    assert!(mae < 1.0, "SVR (RBF) MAE too high: {mae:.3}");
}
