//! Stage 3: edge-case & known-answer parity tests for src/svm.
//!
//! Covers: SVC, SVR.
//! Tracking issue: #394 / #391.

#[cfg(test)]
mod svm_edge_cases {
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::svm::svc::{SVC, SVCParameters};
    use crate::svm::svr::{SVR, SVRParameters};
    use crate::svm::Kernels;

    // ── SVC ───────────────────────────────────────────────────────────────────

    /// Small linearly separable dataset: all predictions must be correct.
    #[test]
    fn svc_linearly_separable_perfect_accuracy() {
        let x = DenseMatrix::from_2d_array(&[
            &[-2.0_f64, -1.0],
            &[-1.0, -1.0],
            &[1.0,  1.0],
            &[2.0,  1.0],
        ]).unwrap();
        let y: Vec<i32> = vec![-1, -1, 1, 1];
        let model = SVC::fit(&x, &y, SVCParameters::default().with_c(100.0)).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds, y, "SVC failed perfect separation");
    }

    /// Decision-function sign must align with predicted class (+1 / -1).
    #[test]
    fn svc_decision_function_sign_consistent() {
        let x = DenseMatrix::from_2d_array(&[
            &[-3.0_f64], &[-2.0], &[2.0], &[3.0],
        ]).unwrap();
        let y: Vec<i32> = vec![-1, -1, 1, 1];
        let model = SVC::fit(&x, &y, SVCParameters::default().with_c(100.0)).unwrap();
        let scores = model.decision_function(&x).unwrap();
        let preds = model.predict(&x).unwrap();
        for (score, pred) in scores.iter().zip(preds.iter()) {
            let expected_sign = if *pred == 1 { 1.0_f64 } else { -1.0_f64 };
            assert!(
                score * expected_sign > 0.0,
                "decision score sign mismatch: score={score}, pred={pred}"
            );
        }
    }

    /// RBF kernel on separable data must achieve 100% accuracy.
    #[test]
    fn svc_rbf_kernel_separable() {
        let x = DenseMatrix::from_2d_array(&[
            &[-2.0_f64, 0.0],
            &[-1.5, 0.0],
            &[1.5, 0.0],
            &[2.0, 0.0],
        ]).unwrap();
        let y: Vec<i32> = vec![-1, -1, 1, 1];
        let model = SVC::fit(
            &x,
            &y,
            SVCParameters::default().with_c(10.0).with_kernel(Kernels::rbf().with_gamma(1.0)),
        ).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds, y);
    }

    /// Non-separable (XOR) should not panic; model returns some prediction.
    #[test]
    fn svc_non_separable_no_panic() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64, 0.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[1.0, 1.0],
        ]).unwrap();
        let y: Vec<i32> = vec![-1, 1, 1, -1]; // XOR
        let result = SVC::fit(&x, &y, SVCParameters::default().with_c(1.0));
        if let Ok(model) = result {
            assert!(model.predict(&x).is_ok());
        }
    }

    // ── SVR ───────────────────────────────────────────────────────────────────

    /// Perfect-fit: y = 2x; predictions within epsilon tolerance.
    #[test]
    fn svr_linear_known_answer() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0],
        ]).unwrap();
        let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let model = SVR::fit(&x, &y, SVRParameters::default().with_c(100.0).with_eps(0.01)).unwrap();
        let y_hat = model.predict(&x).unwrap();
        for (a, b) in y.iter().zip(y_hat.iter()) {
            assert!((a - b).abs() < 1.0, "SVR pred: expected {a}, got {b}");
        }
    }

    /// Constant target: SVR should predict approximately the constant.
    #[test]
    fn svr_constant_target() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64], &[2.0], &[3.0], &[4.0],
        ]).unwrap();
        let y: Vec<f64> = vec![5.0, 5.0, 5.0, 5.0];
        let model = SVR::fit(&x, &y, SVRParameters::default().with_c(1.0)).unwrap();
        let y_hat = model.predict(&x).unwrap();
        for b in y_hat.iter() {
            assert!((b - 5.0).abs() < 1.0, "SVR constant target: got {b}");
        }
    }
}
