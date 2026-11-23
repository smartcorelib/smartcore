//! Stage 3: edge-case & known-answer parity tests for src/neighbors.
//!
//! Covers: KNNClassifier, KNNRegressor.
//! Tracking issue: #394 / #391.

#[cfg(test)]
mod neighbors_edge_cases {
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
    use crate::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};

    // ── KNNClassifier ─────────────────────────────────────────────────────────

    /// k=1: each training point should predict its own label.
    #[test]
    fn knn_classifier_k1_memorises() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 1, 2, 3];
        let model = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(1)).unwrap();
        assert_eq!(model.predict(&x).unwrap(), y);
    }

    /// k=N (all samples): prediction is the majority class everywhere.
    #[test]
    fn knn_classifier_k_equals_n() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64], &[1.0], &[2.0], &[3.0], &[10.0],
        ]).unwrap();
        // 4 class-0, 1 class-1 → majority is class 0 for k=5.
        let y: Vec<u32> = vec![0, 0, 0, 0, 1];
        let model = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(5)).unwrap();
        let preds = model.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 0), "majority-vote should return class 0 everywhere");
    }

    /// Uniform vs distance-weighted: both must succeed and produce valid classes.
    #[test]
    fn knn_classifier_weighted_vs_uniform() {
        use crate::neighbors::KNNWeightFunction;
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64], &[1.0], &[5.0], &[6.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1];
        let uniform = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(2).with_weight(KNNWeightFunction::Uniform)).unwrap();
        let weighted = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(2).with_weight(KNNWeightFunction::Distance)).unwrap();
        let p_u = uniform.predict(&x).unwrap();
        let p_w = weighted.predict(&x).unwrap();
        assert!(p_u.iter().all(|&p| p <= 1));
        assert!(p_w.iter().all(|&p| p <= 1));
    }

    // ── KNNRegressor ──────────────────────────────────────────────────────────

    /// k=1: predictions equal training targets exactly.
    #[test]
    fn knn_regressor_k1_exact() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64], &[1.0], &[2.0], &[3.0],
        ]).unwrap();
        let y: Vec<f64> = vec![0.0, 1.0, 4.0, 9.0];
        let model = KNNRegressor::fit(&x, &y, KNNRegressorParameters::default().with_k(1)).unwrap();
        let preds = model.predict(&x).unwrap();
        for (a, b) in y.iter().zip(preds.iter()) {
            assert!((a - b).abs() < 1e-10, "k=1 should memorise: expected {a}, got {b}");
        }
    }

    /// Uniform vs distance-weighted diverge on asymmetric neighbours.
    #[test]
    fn knn_regressor_uniform_vs_distance() {
        use crate::neighbors::KNNWeightFunction;
        // Query point at 1.1: neighbours are 1.0 (y=10) and 2.0 (y=20).
        // Distance weighting will bias toward 1.0 (closer).
        let x_train = DenseMatrix::from_2d_array(&[
            &[0.0_f64], &[1.0], &[2.0],
        ]).unwrap();
        let y_train: Vec<f64> = vec![0.0, 10.0, 20.0];
        let x_test = DenseMatrix::from_2d_array(&[&[1.1_f64]]).unwrap();

        let uniform = KNNRegressor::fit(&x_train, &y_train, KNNRegressorParameters::default().with_k(2).with_weight(KNNWeightFunction::Uniform)).unwrap();
        let weighted = KNNRegressor::fit(&x_train, &y_train, KNNRegressorParameters::default().with_k(2).with_weight(KNNWeightFunction::Distance)).unwrap();

        let p_u = uniform.predict(&x_test).unwrap()[0];
        let p_w = weighted.predict(&x_test).unwrap()[0];
        // Uniform = 15.0, distance-weighted < 15.0 (biased toward y=10).
        assert!((p_u - 15.0).abs() < 1e-6, "uniform expected 15.0, got {p_u}");
        assert!(p_w < p_u, "distance-weighted should be less than uniform: {p_w} vs {p_u}");
    }
}
