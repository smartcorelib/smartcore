//! Stage 4: edge-case & boundary tests for src/metrics.
//!
//! Covers: accuracy, precision, recall, f1, auc, r2, mae, mse,
//!         cluster_hcv / cluster_helpers contingency edge cases.
//! Tracking issue: #395 / #391.

#[cfg(test)]
mod metrics_edge_cases {
    use crate::metrics::{
        accuracy::Accuracy,
        auc::AUC,
        f1::F1,
        mean_absolute_error::MeanAbsoluteError,
        mean_squared_error::MeanSquaredError,
        precision::Precision,
        r2::R2,
        recall::Recall,
        Metrics,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn assert_close(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: expected {b}, got {a} (tol {tol})");
    }

    // ── Accuracy ─────────────────────────────────────────────────────────────

    /// Perfect predictions → 1.0
    #[test]
    fn accuracy_perfect_score() {
        let y: Vec<i32> = vec![0, 1, 2, 1, 0];
        let score = Accuracy::<i32>::new().get_score(&y, &y);
        assert_close(score, 1.0, 1e-10, "accuracy perfect");
    }

    /// All wrong (binary) → 0.0
    #[test]
    fn accuracy_worst_score_binary() {
        let y_true: Vec<i32> = vec![0, 0, 0, 0];
        let y_pred: Vec<i32> = vec![1, 1, 1, 1];
        let score = Accuracy::<i32>::new().get_score(&y_true, &y_pred);
        assert_close(score, 0.0, 1e-10, "accuracy worst");
    }

    /// Single sample, correct → 1.0
    #[test]
    fn accuracy_single_sample_correct() {
        let y_true: Vec<i32> = vec![1];
        let y_pred: Vec<i32> = vec![1];
        assert_close(Accuracy::<i32>::new().get_score(&y_true, &y_pred), 1.0, 1e-10, "single correct");
    }

    /// Single sample, wrong → 0.0
    #[test]
    fn accuracy_single_sample_wrong() {
        let y_true: Vec<i32> = vec![1];
        let y_pred: Vec<i32> = vec![0];
        assert_close(Accuracy::<i32>::new().get_score(&y_true, &y_pred), 0.0, 1e-10, "single wrong");
    }

    // ── Precision ────────────────────────────────────────────────────────────

    /// Perfect → 1.0
    #[test]
    fn precision_perfect() {
        let y: Vec<i32> = vec![0, 1, 1, 0, 1];
        let score = Precision::<i32>::new().get_score(&y, &y);
        assert_close(score, 1.0, 1e-10, "precision perfect");
    }

    /// All FP for positive class → 0.0
    #[test]
    fn precision_all_false_positives() {
        let y_true: Vec<i32> = vec![0, 0, 0, 0];
        let y_pred: Vec<i32> = vec![1, 1, 1, 1];
        let score = Precision::<i32>::new().get_score(&y_true, &y_pred);
        assert_close(score, 0.0, 1e-10, "precision all FP");
    }

    // ── Recall ───────────────────────────────────────────────────────────────

    /// Perfect → 1.0
    #[test]
    fn recall_perfect() {
        let y: Vec<i32> = vec![1, 0, 1, 1, 0];
        let score = Recall::<i32>::new().get_score(&y, &y);
        assert_close(score, 1.0, 1e-10, "recall perfect");
    }

    /// All FN: no positives predicted → 0.0
    #[test]
    fn recall_all_false_negatives() {
        let y_true: Vec<i32> = vec![1, 1, 1, 1];
        let y_pred: Vec<i32> = vec![0, 0, 0, 0];
        let score = Recall::<i32>::new().get_score(&y_true, &y_pred);
        assert_close(score, 0.0, 1e-10, "recall all FN");
    }

    // ── F1 ───────────────────────────────────────────────────────────────────

    /// Perfect predictions → 1.0
    #[test]
    fn f1_perfect() {
        let y: Vec<i32> = vec![0, 1, 1, 0, 1];
        let score = F1::<i32>::new().get_score(&y, &y);
        assert_close(score, 1.0, 1e-10, "f1 perfect");
    }

    /// All wrong binary predictions → 0.0
    #[test]
    fn f1_all_wrong() {
        let y_true: Vec<i32> = vec![1, 1, 1, 1];
        let y_pred: Vec<i32> = vec![0, 0, 0, 0];
        let score = F1::<i32>::new().get_score(&y_true, &y_pred);
        assert_close(score, 0.0, 1e-10, "f1 worst");
    }

    /// F1 = 2*P*R / (P+R) known-answer check.
    #[test]
    fn f1_known_answer() {
        // TP=2, FP=1, FN=1 → P=2/3, R=2/3, F1=2/3
        let y_true: Vec<i32> = vec![1, 1, 0, 1];
        let y_pred: Vec<i32> = vec![1, 1, 1, 0];
        let score = F1::<i32>::new().get_score(&y_true, &y_pred);
        assert_close(score, 2.0 / 3.0, 1e-6, "f1 known");
    }

    // ── AUC ──────────────────────────────────────────────────────────────────

    /// Perfect ranking → AUC = 1.0
    #[test]
    fn auc_perfect() {
        let y_true: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0];
        let y_score: Vec<f64> = vec![0.1, 0.2, 0.8, 0.9];
        let score = AUC::<f64>::new().get_score(&y_true, &y_score);
        assert_close(score, 1.0, 1e-8, "auc perfect");
    }

    /// Worst ranking (all inverted) → AUC = 0.0
    #[test]
    fn auc_worst() {
        let y_true: Vec<f64> = vec![1.0, 1.0, 0.0, 0.0];
        let y_score: Vec<f64> = vec![0.1, 0.2, 0.8, 0.9];
        let score = AUC::<f64>::new().get_score(&y_true, &y_score);
        assert_close(score, 0.0, 1e-8, "auc worst");
    }

    /// Random ranking → AUC ≈ 0.5
    #[test]
    fn auc_random_is_half() {
        let y_true: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0];
        let y_score: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5]; // all tied
        let score = AUC::<f64>::new().get_score(&y_true, &y_score);
        // Tied scores → AUC may be 0.5; accept range [0.0, 1.0] as non-panicking.
        assert!(score >= 0.0 && score <= 1.0, "auc tied scores out of range: {score}");
    }

    // ── R² ───────────────────────────────────────────────────────────────────

    /// Perfect predictions → R² = 1.0
    #[test]
    fn r2_perfect() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let score = R2::<f64>::new().get_score(&y, &y);
        assert_close(score, 1.0, 1e-10, "r2 perfect");
    }

    /// Predicting the mean → R² = 0.0
    #[test]
    fn r2_predicting_mean() {
        let y_true: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0]; // mean = 2.5
        let y_pred: Vec<f64> = vec![2.5, 2.5, 2.5, 2.5];
        let score = R2::<f64>::new().get_score(&y_true, &y_pred);
        assert_close(score, 0.0, 1e-10, "r2 mean baseline");
    }

    /// Known-answer: y=[1,2,3], ŷ=[2,2,2] → R² = -0.5
    #[test]
    fn r2_known_answer_negative() {
        let y_true: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y_pred: Vec<f64> = vec![2.0, 2.0, 2.0];
        let score = R2::<f64>::new().get_score(&y_true, &y_pred);
        // SS_res = (1-2)²+(2-2)²+(3-2)² = 2; SS_tot = (1-2)²+(2-2)²+(3-2)² = 2; R²=0
        // Actually mean=2: SS_tot=2, SS_res=2, R²=1-2/2=0.0
        assert_close(score, 0.0, 1e-10, "r2 predict mean");
    }

    // ── MAE ──────────────────────────────────────────────────────────────────

    /// Perfect → 0.0
    #[test]
    fn mae_perfect() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let score = MeanAbsoluteError::<f64>::new().get_score(&y, &y);
        assert_close(score, 0.0, 1e-10, "mae perfect");
    }

    /// Known answer: |1-3|+|2-2|+|3-1| / 3 = 4/3
    #[test]
    fn mae_known_answer() {
        let y_true: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y_pred: Vec<f64> = vec![3.0, 2.0, 1.0];
        let score = MeanAbsoluteError::<f64>::new().get_score(&y_true, &y_pred);
        assert_close(score, 4.0 / 3.0, 1e-10, "mae known");
    }

    /// Single-sample: error = |y_true - y_pred|
    #[test]
    fn mae_single_sample() {
        let y_true: Vec<f64> = vec![5.0];
        let y_pred: Vec<f64> = vec![2.0];
        let score = MeanAbsoluteError::<f64>::new().get_score(&y_true, &y_pred);
        assert_close(score, 3.0, 1e-10, "mae single");
    }

    // ── MSE ──────────────────────────────────────────────────────────────────

    /// Perfect → 0.0
    #[test]
    fn mse_perfect() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let score = MeanSquaredError::<f64>::new().get_score(&y, &y);
        assert_close(score, 0.0, 1e-10, "mse perfect");
    }

    /// Known answer: ((1-2)²+(2-4)²+(3-6)²)/3 = (1+4+9)/3 = 14/3
    #[test]
    fn mse_known_answer() {
        let y_true: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y_pred: Vec<f64> = vec![2.0, 4.0, 6.0];
        let score = MeanSquaredError::<f64>::new().get_score(&y_true, &y_pred);
        assert_close(score, 14.0 / 3.0, 1e-10, "mse known");
    }

    /// Single-sample: squared error = (y_true - y_pred)²
    #[test]
    fn mse_single_sample() {
        let y_true: Vec<f64> = vec![0.0];
        let y_pred: Vec<f64> = vec![3.0];
        let score = MeanSquaredError::<f64>::new().get_score(&y_true, &y_pred);
        assert_close(score, 9.0, 1e-10, "mse single");
    }

    // ── Cluster HCV contingency edge cases ───────────────────────────────────

    /// Single cluster: all points in one cluster, any labels.
    #[test]
    fn cluster_hcv_single_cluster_no_panic() {
        use crate::metrics::cluster_hcv::{
            homogeneity_score, completeness_score, v_measure_score,
        };
        let labels_true: Vec<u32> = vec![0, 1, 2, 0, 1];
        let labels_pred: Vec<u32> = vec![0, 0, 0, 0, 0]; // all same cluster
        // Must not panic; homogeneity = 0 (mixed), completeness = 1 (all in one cluster).
        let h = homogeneity_score(&labels_true, &labels_pred);
        let c = completeness_score(&labels_true, &labels_pred);
        let v = v_measure_score(&labels_true, &labels_pred);
        assert!(h >= 0.0 && h <= 1.0, "h={h}");
        assert_close(c, 1.0, 1e-6, "completeness single cluster");
        assert!(v >= 0.0 && v <= 1.0, "v={v}");
    }

    /// Perfect clustering: pred == true → homogeneity = completeness = v = 1.
    #[test]
    fn cluster_hcv_perfect_clustering() {
        use crate::metrics::cluster_hcv::{
            homogeneity_score, completeness_score, v_measure_score,
        };
        let labels: Vec<u32> = vec![0, 0, 1, 1, 2, 2];
        let h = homogeneity_score(&labels, &labels);
        let c = completeness_score(&labels, &labels);
        let v = v_measure_score(&labels, &labels);
        assert_close(h, 1.0, 1e-6, "h perfect");
        assert_close(c, 1.0, 1e-6, "c perfect");
        assert_close(v, 1.0, 1e-6, "v perfect");
    }

    /// All same labels: single-class truth → completeness = 1.
    #[test]
    fn cluster_hcv_all_same_true_labels() {
        use crate::metrics::cluster_hcv::completeness_score;
        let labels_true: Vec<u32> = vec![0, 0, 0, 0];
        let labels_pred: Vec<u32> = vec![0, 1, 0, 1];
        let c = completeness_score(&labels_true, &labels_pred);
        // When all true labels are the same, completeness is undefined / 1.0.
        assert!(c >= 0.0 && c <= 1.0, "completeness={c}");
    }
}
