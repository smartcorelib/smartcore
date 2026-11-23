//! Stage 3: edge-case & known-answer parity tests for src/ensemble.
//!
//! Covers: RandomForestClassifier, RandomForestRegressor.
//! Tracking issue: #394 / #391.

#[cfg(test)]
mod ensemble_edge_cases {
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::ensemble::random_forest_classifier::{RandomForestClassifier, RandomForestClassifierParameters};
    use crate::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};

    // ── RandomForestClassifier ────────────────────────────────────────────────

    /// n_estimators=1 should behave like a single decision tree.
    #[test]
    fn rfc_one_estimator_no_panic() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1];
        let model = RandomForestClassifier::fit(
            &x, &y,
            RandomForestClassifierParameters::default().with_n_trees(1).with_seed(42),
        ).unwrap();
        assert!(model.predict(&x).is_ok());
    }

    /// Seed determinism: identical seeds produce identical predictions.
    #[test]
    fn rfc_seed_determinism() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
            &[11.0, 12.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let m1 = RandomForestClassifier::fit(
            &x, &y,
            RandomForestClassifierParameters::default().with_n_trees(10).with_seed(123),
        ).unwrap();
        let m2 = RandomForestClassifier::fit(
            &x, &y,
            RandomForestClassifierParameters::default().with_n_trees(10).with_seed(123),
        ).unwrap();
        assert_eq!(m1.predict(&x).unwrap(), m2.predict(&x).unwrap());
    }

    /// Different seeds should (usually) differ — at minimum must not panic.
    #[test]
    fn rfc_different_seeds_no_panic() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0], &[3.0, 4.0], &[5.0, 6.0],
            &[7.0, 8.0],    &[9.0, 10.0], &[11.0, 12.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        assert!(RandomForestClassifier::fit(&x, &y, RandomForestClassifierParameters::default().with_seed(1)).is_ok());
        assert!(RandomForestClassifier::fit(&x, &y, RandomForestClassifierParameters::default().with_seed(2)).is_ok());
    }

    // ── RandomForestRegressor ─────────────────────────────────────────────────

    /// n_estimators=1 should not panic and return finite predictions.
    #[test]
    fn rfr_one_estimator_finite_preds() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = RandomForestRegressor::fit(
            &x, &y,
            RandomForestRegressorParameters::default().with_n_trees(1).with_seed(0),
        ).unwrap();
        let preds = model.predict(&x).unwrap();
        assert!(preds.iter().all(|v| v.is_finite()));
    }

    /// Seed determinism for regressor.
    #[test]
    fn rfr_seed_determinism() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 0.0], &[2.0, 1.0], &[3.0, 2.0],
            &[4.0, 3.0],    &[5.0, 4.0], &[6.0, 5.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p1 = RandomForestRegressor::fit(&x, &y, RandomForestRegressorParameters::default().with_n_trees(10).with_seed(99)).unwrap().predict(&x).unwrap();
        let p2 = RandomForestRegressor::fit(&x, &y, RandomForestRegressorParameters::default().with_n_trees(10).with_seed(99)).unwrap().predict(&x).unwrap();
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < 1e-10, "non-deterministic: {a} vs {b}");
        }
    }
}
