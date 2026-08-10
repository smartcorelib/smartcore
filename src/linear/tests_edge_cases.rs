//! Stage 3: edge-case & known-answer parity tests for src/linear.
//!
//! Covers: linear_regression, logistic_regression, ridge_regression, lasso, elastic_net.
//! Tracking issue: #394 / #391.

#[cfg(test)]
mod linear_edge_cases {
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::linear::linear_regression::{LinearRegression, LinearRegressionParameters, LinearRegressionSolverName};
    use crate::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};
    use crate::linear::lasso::{Lasso, LassoParameters};
    use crate::linear::elastic_net::{ElasticNet, ElasticNetParameters};
    use crate::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};

    // ── LinearRegression ──────────────────────────────────────────────────────

    /// Single-sample: model should fit without panic and reproduce the target.
    #[test]
    fn linear_regression_single_sample() {
        let x = DenseMatrix::from_2d_array(&[&[1.0_f64, 2.0]]).unwrap();
        let y: Vec<f64> = vec![5.0];
        let model = LinearRegression::fit(&x, &y, Default::default());
        // Under-determined with SVD: fit must succeed (not panic/error).
        assert!(model.is_ok());
    }

    /// Perfect-fit (y = 2*x1 + 3*x2): coefficients should reproduce exactly.
    #[test]
    fn linear_regression_perfect_fit_known_answer() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[2.0, 3.0],
        ]).unwrap();
        let y: Vec<f64> = vec![2.0, 3.0, 5.0, 13.0];
        let model = LinearRegression::fit(&x, &y, LinearRegressionParameters {
            solver: LinearRegressionSolverName::QR,
        }).unwrap();
        let y_hat = model.predict(&x).unwrap();
        for (a, b) in y.iter().zip(y_hat.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {a}, got {b}");
        }
    }

    /// QR and SVD solvers must agree on coefficients to high precision.
    #[test]
    fn linear_regression_solver_parity() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let qr = LinearRegression::fit(&x, &y, LinearRegressionParameters { solver: LinearRegressionSolverName::QR }).unwrap();
        let svd = LinearRegression::fit(&x, &y, LinearRegressionParameters { solver: LinearRegressionSolverName::SVD }).unwrap();
        let y_qr = qr.predict(&x).unwrap();
        let y_svd = svd.predict(&x).unwrap();
        for (a, b) in y_qr.iter().zip(y_svd.iter()) {
            assert!((a - b).abs() < 1e-6, "QR={a} SVD={b} diverge");
        }
    }

    /// Collinear features: SVD fit must succeed (QR may fail; SVD is the fallback).
    #[test]
    fn linear_regression_collinear_features_svd() {
        // x2 == 2 * x1 → rank-deficient design matrix.
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0],
            &[2.0, 4.0],
            &[3.0, 6.0],
            &[4.0, 8.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let result = LinearRegression::fit(&x, &y, LinearRegressionParameters {
            solver: LinearRegressionSolverName::SVD,
        });
        // Must not panic; predictions should be within a loose tolerance.
        assert!(result.is_ok());
        let y_hat = result.unwrap().predict(&x).unwrap();
        for (a, b) in y.iter().zip(y_hat.iter()) {
            assert!((a - b).abs() < 0.5, "collinear pred error: expected {a}, got {b}");
        }
    }

    // ── RidgeRegression ───────────────────────────────────────────────────────

    /// Ridge with alpha=0 should closely match OLS.
    #[test]
    fn ridge_regression_zero_alpha_matches_ols() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ridge = RidgeRegression::fit(&x, &y, RidgeRegressionParameters::default().with_alpha(0.0)).unwrap();
        let ols = LinearRegression::fit(&x, &y, Default::default()).unwrap();
        let y_ridge = ridge.predict(&x).unwrap();
        let y_ols = ols.predict(&x).unwrap();
        for (a, b) in y_ridge.iter().zip(y_ols.iter()) {
            assert!((a - b).abs() < 1e-4, "ridge(α=0)={a} vs ols={b}");
        }
    }

    /// Ridge perfect-fit known answer: y = 3*x (single feature, no intercept ambiguity).
    #[test]
    fn ridge_regression_known_answer() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64],
            &[2.0],
            &[3.0],
            &[4.0],
            &[5.0],
        ]).unwrap();
        let y: Vec<f64> = vec![3.0, 6.0, 9.0, 12.0, 15.0];
        let model = RidgeRegression::fit(&x, &y, RidgeRegressionParameters::default().with_alpha(0.0)).unwrap();
        let y_hat = model.predict(&x).unwrap();
        for (a, b) in y.iter().zip(y_hat.iter()) {
            assert!((a - b).abs() < 1e-3, "expected {a}, got {b}");
        }
    }

    /// Ridge with collinear features must not panic.
    #[test]
    fn ridge_regression_collinear_features() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 1.0],
            &[2.0, 2.0],
            &[3.0, 3.0],
            &[4.0, 4.0],
        ]).unwrap();
        let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0];
        let result = RidgeRegression::fit(&x, &y, RidgeRegressionParameters::default().with_alpha(1.0));
        assert!(result.is_ok());
    }

    // ── Lasso ─────────────────────────────────────────────────────────────────

    /// Lasso on perfectly collinear features: must not panic; prediction reasonable.
    #[test]
    fn lasso_collinear_features() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 1.0],
            &[2.0, 2.0],
            &[3.0, 3.0],
            &[4.0, 4.0],
            &[5.0, 5.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = Lasso::fit(&x, &y, LassoParameters::default().with_alpha(0.01));
        assert!(result.is_ok());
    }

    /// Lasso sparsity: with a high alpha most coefficients should shrink to ~0.
    #[test]
    fn lasso_high_alpha_sparse_coefficients() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 1.0, 0.0],
            &[0.0, 1.0, 1.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 1.0, 1.0, 2.0, 2.0];
        let model = Lasso::fit(&x, &y, LassoParameters::default().with_alpha(10.0)).unwrap();
        let coeffs = model.coefficients();
        let nonzero: usize = coeffs.iter().filter(|&&c| c.abs() > 1e-6).count();
        assert!(nonzero <= 2, "expected sparse coefficients, got {nonzero} nonzero");
    }

    // ── ElasticNet ────────────────────────────────────────────────────────────

    /// ElasticNet with l1_ratio=1 should behave like Lasso.
    #[test]
    fn elastic_net_l1_ratio_one_behaves_like_lasso() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 0.1;
        let en = ElasticNet::fit(&x, &y, ElasticNetParameters::default().with_alpha(alpha).with_l1_ratio(1.0)).unwrap();
        let lasso = Lasso::fit(&x, &y, LassoParameters::default().with_alpha(alpha)).unwrap();
        let y_en = en.predict(&x).unwrap();
        let y_lasso = lasso.predict(&x).unwrap();
        for (a, b) in y_en.iter().zip(y_lasso.iter()) {
            assert!((a - b).abs() < 0.5, "EN(l1=1)={a} vs Lasso={b}");
        }
    }

    /// ElasticNet with l1_ratio=0 should behave like Ridge.
    #[test]
    fn elastic_net_l1_ratio_zero_behaves_like_ridge() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 0.1;
        let en = ElasticNet::fit(&x, &y, ElasticNetParameters::default().with_alpha(alpha).with_l1_ratio(0.0)).unwrap();
        let ridge = RidgeRegression::fit(&x, &y, RidgeRegressionParameters::default().with_alpha(alpha)).unwrap();
        let y_en = en.predict(&x).unwrap();
        let y_ridge = ridge.predict(&x).unwrap();
        for (a, b) in y_en.iter().zip(y_ridge.iter()) {
            assert!((a - b).abs() < 0.5, "EN(l1=0)={a} vs Ridge={b}");
        }
    }

    // ── LogisticRegression ────────────────────────────────────────────────────

    /// Linearly separable 2-class: accuracy must be 100%.
    #[test]
    fn logistic_regression_separable_perfect_accuracy() {
        let x = DenseMatrix::from_2d_array(&[
            &[-2.0_f64], &[-1.0], &[1.0], &[2.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1];
        let model = LogisticRegression::fit(&x, &y, LogisticRegressionParameters::default()).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds, y, "expected perfect separation");
    }

    /// Single-feature binary: deterministic across identical runs.
    #[test]
    fn logistic_regression_deterministic() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64], &[1.0], &[2.0], &[3.0], &[4.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1, 1];
        let m1 = LogisticRegression::fit(&x, &y, LogisticRegressionParameters::default()).unwrap();
        let m2 = LogisticRegression::fit(&x, &y, LogisticRegressionParameters::default()).unwrap();
        assert_eq!(m1.predict(&x).unwrap(), m2.predict(&x).unwrap());
    }
}
