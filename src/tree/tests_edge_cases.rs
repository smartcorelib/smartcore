//! Stage 3: edge-case & known-answer parity tests for src/tree.
//!
//! Covers: DecisionTreeClassifier, DecisionTreeRegressor.
//! Tracking issue: #394 / #391.

#[cfg(test)]
mod tree_edge_cases {
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::tree::decision_tree_classifier::{DecisionTreeClassifier, DecisionTreeClassifierParameters};
    use crate::tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters};

    // ── DecisionTreeClassifier ────────────────────────────────────────────────

    /// Depth=1 (stump): must partition into two majority groups.
    #[test]
    fn dtc_depth_limit_one() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64], &[1.0], &[2.0], &[3.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1];
        let model = DecisionTreeClassifier::fit(
            &x, &y,
            DecisionTreeClassifierParameters::default().with_max_depth(1),
        ).unwrap();
        let preds = model.predict(&x).unwrap();
        // With depth=1 there are exactly 2 leaves → both classes present.
        assert!(preds.contains(&0) && preds.contains(&1));
    }

    /// Pure-node early stopping: single-class input fits without error.
    #[test]
    fn dtc_pure_node_single_class() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
        ]).unwrap();
        let y: Vec<u32> = vec![1, 1, 1]; // all same class
        let model = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
        let preds = model.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 1));
    }

    /// Single-feature split: must perfectly classify linearly separable data.
    #[test]
    fn dtc_single_feature_perfect_split() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64], &[1.0], &[10.0], &[11.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1];
        let model = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds, y);
    }

    /// Determinism: same data, same seed → identical predictions.
    #[test]
    fn dtc_deterministic_with_seed() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 0.5],
            &[2.0, 1.5],
            &[3.0, 2.5],
            &[4.0, 3.5],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1];
        let m1 = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
        let m2 = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
        assert_eq!(m1.predict(&x).unwrap(), m2.predict(&x).unwrap());
    }

    // ── DecisionTreeRegressor ─────────────────────────────────────────────────

    /// Constant target: all predictions should equal the constant.
    #[test]
    fn dtr_constant_target() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64], &[2.0], &[3.0], &[4.0],
        ]).unwrap();
        let y: Vec<f64> = vec![7.0, 7.0, 7.0, 7.0];
        let model = DecisionTreeRegressor::fit(&x, &y, Default::default()).unwrap();
        let y_hat = model.predict(&x).unwrap();
        for b in y_hat.iter() {
            assert!((b - 7.0).abs() < 1e-6, "expected 7.0, got {b}");
        }
    }

    /// Known-answer: y = x (perfect staircase); deep tree should memorise.
    #[test]
    fn dtr_known_answer_identity() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = DecisionTreeRegressor::fit(&x, &y, Default::default()).unwrap();
        let y_hat = model.predict(&x).unwrap();
        for (a, b) in y.iter().zip(y_hat.iter()) {
            assert!((a - b).abs() < 1e-6, "expected {a}, got {b}");
        }
    }

    /// Depth limit: capped depth should not panic and predictions are finite.
    #[test]
    fn dtr_depth_limit_no_panic() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0],
        ]).unwrap();
        let y: Vec<f64> = vec![1.5, 3.5, 2.0, 4.5, 0.5];
        let model = DecisionTreeRegressor::fit(
            &x, &y,
            DecisionTreeRegressorParameters::default().with_max_depth(2),
        ).unwrap();
        let y_hat = model.predict(&x).unwrap();
        assert!(y_hat.iter().all(|v| v.is_finite()));
    }
}
