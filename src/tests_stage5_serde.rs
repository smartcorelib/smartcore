//! Stage 5: serde round-trip tests for every serializable type.
//!
//! Gated on `#[cfg(feature = "serde")]` so they only run under
//! `cargo test --features serde` (or `--all-features`).
//!
//! Tracking issue: #396 / #391.

#[cfg(all(test, feature = "serde"))]
mod serde_round_trips {
    use serde_json;

    // ── DenseMatrix ──────────────────────────────────────────────────────────

    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::linalg::basic::arrays::Array;

    #[test]
    fn dense_matrix_serde_json_round_trip() {
        let m = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
        ])
        .unwrap();
        let json = serde_json::to_string(&m).expect("serialize DenseMatrix");
        let m2: DenseMatrix<f64> = serde_json::from_str(&json).expect("deserialize DenseMatrix");
        assert_eq!(m.shape(), m2.shape());
        for r in 0..2 {
            for c in 0..3 {
                assert!(
                    (m.get((r, c)) - m2.get((r, c))).abs() < 1e-10,
                    "mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn dense_matrix_i32_serde_json_round_trip() {
        let m = DenseMatrix::from_2d_array(&[&[1_i32, 2, 3], &[4, 5, 6]]).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        let m2: DenseMatrix<i32> = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    // ── KNNClassifier ────────────────────────────────────────────────────────

    use crate::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
    use crate::metrics::distance::Distances;
    use crate::algorithm::neighbour::KNNAlgorithmName;
    use crate::neighbors::KNNWeightFunction;

    #[test]
    fn knn_classifier_serde_json_round_trip() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0], &[2.0, 3.0], &[3.0, 4.0],
            &[10.0, 11.0], &[11.0, 12.0], &[12.0, 13.0],
        ])
        .unwrap();
        let y: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let params = KNNClassifierParameters::default()
            .with_k(3)
            .with_algorithm(KNNAlgorithmName::LinearSearch)
            .with_weight(KNNWeightFunction::Uniform);
        let model = KNNClassifier::fit(&x, &y, params).unwrap();
        let json = serde_json::to_string(&model).expect("serialize KNNClassifier");
        let model2: KNNClassifier<f64, u32, DenseMatrix<f64>, _> =
            serde_json::from_str(&json).expect("deserialize KNNClassifier");
        let pred1 = model.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();
        assert_eq!(pred1, pred2, "KNNClassifier round-trip prediction mismatch");
    }

    // ── KNNRegressor ─────────────────────────────────────────────────────────

    use crate::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};

    #[test]
    fn knn_regressor_serde_json_round_trip() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0],
        ])
        .unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let params = KNNRegressorParameters::default().with_k(2);
        let model = KNNRegressor::fit(&x, &y, params).unwrap();
        let json = serde_json::to_string(&model).expect("serialize KNNRegressor");
        let model2: KNNRegressor<f64, f64, DenseMatrix<f64>, _> =
            serde_json::from_str(&json).expect("deserialize KNNRegressor");
        let pred1 = model.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();
        for (a, b) in pred1.iter().zip(pred2.iter()) {
            assert!((a - b).abs() < 1e-10, "KNNRegressor round-trip mismatch");
        }
    }

    // ── DecisionTreeClassifier ───────────────────────────────────────────────

    use crate::tree::decision_tree_classifier::{DecisionTreeClassifier, DecisionTreeClassifierParameters};

    #[test]
    fn decision_tree_classifier_serde_json_round_trip() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0_f64, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0],
            &[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0],
        ])
        .unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let params = DecisionTreeClassifierParameters::default().with_max_depth(3);
        let model = DecisionTreeClassifier::fit(&x, &y, params).unwrap();
        let json = serde_json::to_string(&model).expect("serialize DecisionTreeClassifier");
        let model2: DecisionTreeClassifier<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            serde_json::from_str(&json).expect("deserialize DecisionTreeClassifier");
        let pred1 = model.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();
        assert_eq!(pred1, pred2);
    }

    // ── DecisionTreeRegressor ────────────────────────────────────────────────

    use crate::tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters};

    #[test]
    fn decision_tree_regressor_serde_json_round_trip() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0], &[6.0],
        ])
        .unwrap();
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let params = DecisionTreeRegressorParameters::default().with_max_depth(3);
        let model = DecisionTreeRegressor::fit(&x, &y, params).unwrap();
        let json = serde_json::to_string(&model).expect("serialize DecisionTreeRegressor");
        let model2: DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            serde_json::from_str(&json).expect("deserialize DecisionTreeRegressor");
        let pred1 = model.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();
        for (a, b) in pred1.iter().zip(pred2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ── LinearRegression ─────────────────────────────────────────────────────

    use crate::linear::linear_regression::{LinearRegression, LinearRegressionParameters};
    use crate::SupervisedEstimator;

    #[test]
    fn linear_regression_serde_json_round_trip() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64], &[2.0], &[3.0], &[4.0], &[5.0],
        ])
        .unwrap();
        let y: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let model = LinearRegression::fit(&x, &y, LinearRegressionParameters::default()).unwrap();
        let json = serde_json::to_string(&model).expect("serialize LinearRegression");
        let model2: LinearRegression<f64, DenseMatrix<f64>, Vec<f64>> =
            serde_json::from_str(&json).expect("deserialize LinearRegression");
        let pred1 = model.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();
        for (a, b) in pred1.iter().zip(pred2.iter()) {
            assert!((a - b).abs() < 1e-6, "LinearRegression round-trip mismatch");
        }
    }

    // ── GaussianNB ───────────────────────────────────────────────────────────

    use crate::naive_bayes::gaussian::{GaussianNB, GaussianNBParameters};

    #[test]
    fn gaussian_nb_serde_json_round_trip() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 0.0], &[1.1, 0.1], &[0.9, -0.1],
            &[-1.0, 0.0], &[-1.1, 0.1], &[-0.9, -0.1],
        ])
        .unwrap();
        let y: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let model = GaussianNB::fit(&x, &y, GaussianNBParameters::default()).unwrap();
        let json = serde_json::to_string(&model).expect("serialize GaussianNB");
        let model2: GaussianNB<f64, u32, DenseMatrix<f64>, Vec<u32>> =
            serde_json::from_str(&json).expect("deserialize GaussianNB");
        let pred1 = model.predict(&x).unwrap();
        let pred2 = model2.predict(&x).unwrap();
        assert_eq!(pred1, pred2, "GaussianNB round-trip prediction mismatch");
    }

    // ── PCA ──────────────────────────────────────────────────────────────────

    use crate::decomposition::pca::{PCA, PCAParameters};
    use crate::Transformer;

    #[test]
    fn pca_serde_json_round_trip() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0],
        ])
        .unwrap();
        let params = PCAParameters::default().with_n_components(2);
        let model = PCA::fit(&x, params).unwrap();
        let json = serde_json::to_string(&model).expect("serialize PCA");
        let model2: PCA<f64, DenseMatrix<f64>> =
            serde_json::from_str(&json).expect("deserialize PCA");
        let t1 = model.transform(&x).unwrap();
        let t2 = model2.transform(&x).unwrap();
        assert_eq!(t1.shape(), t2.shape());
    }
}
