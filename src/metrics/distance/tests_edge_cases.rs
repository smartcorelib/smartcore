//! Stage 4: edge-case & boundary tests for src/metrics/distance.
//!
//! Covers: Euclidian, Manhattan, Minkowski, Hamming, Mahalanobis, PairwiseDistance.
//! Tracking issue: #395 / #391.

#[cfg(test)]
mod distance_edge_cases {
    use crate::metrics::distance::{
        euclidian::Euclidian,
        hamming::Hamming,
        mahalanobis::Mahalanobis,
        manhattan::Manhattan,
        minkowski::Minkowski,
        Distance,
    };
    use crate::linalg::basic::matrix::DenseMatrix;

    fn assert_close(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: expected {b}, got {a} (tol {tol})");
    }

    // ── Euclidian ─────────────────────────────────────────────────────────────

    /// d(x, x) = 0 for zero vector.
    #[test]
    fn euclidian_zero_vector_self_distance() {
        let z: Vec<f64> = vec![0.0, 0.0, 0.0];
        assert_close(Euclidian::new().distance(&z, &z), 0.0, 1e-10, "euclidian zero self");
    }

    /// d(x, x) = 0 for any vector.
    #[test]
    fn euclidian_identical_points_zero() {
        let a: Vec<f64> = vec![3.0, -1.0, 4.0, 1.5];
        assert_close(Euclidian::new().distance(&a, &a), 0.0, 1e-10, "euclidian identical");
    }

    /// Known answer: d([0,0],[3,4]) = 5.
    #[test]
    fn euclidian_known_answer_3_4_5() {
        let a: Vec<f64> = vec![0.0, 0.0];
        let b: Vec<f64> = vec![3.0, 4.0];
        assert_close(Euclidian::new().distance(&a, &b), 5.0, 1e-10, "euclidian 3-4-5");
    }

    /// Symmetry: d(a,b) == d(b,a).
    #[test]
    fn euclidian_symmetric() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 5.0, 6.0];
        let d_ab = Euclidian::new().distance(&a, &b);
        let d_ba = Euclidian::new().distance(&b, &a);
        assert_close(d_ab, d_ba, 1e-10, "euclidian symmetric");
    }

    // ── Manhattan ────────────────────────────────────────────────────────────

    /// d(x, x) = 0.
    #[test]
    fn manhattan_identical_zero() {
        let a: Vec<f64> = vec![1.0, -2.0, 3.0];
        assert_close(Manhattan::new().distance(&a, &a), 0.0, 1e-10, "manhattan identical");
    }

    /// Known answer: |1-4| + |2-5| + |3-6| = 9.
    #[test]
    fn manhattan_known_answer() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 5.0, 6.0];
        assert_close(Manhattan::new().distance(&a, &b), 9.0, 1e-10, "manhattan known");
    }

    /// Zero vector to any point = sum of absolute values.
    #[test]
    fn manhattan_zero_vector() {
        let z: Vec<f64> = vec![0.0, 0.0];
        let a: Vec<f64> = vec![3.0, 4.0];
        assert_close(Manhattan::new().distance(&z, &a), 7.0, 1e-10, "manhattan from zero");
    }

    /// Symmetry.
    #[test]
    fn manhattan_symmetric() {
        let a: Vec<f64> = vec![1.0, 5.0];
        let b: Vec<f64> = vec![4.0, 1.0];
        assert_close(
            Manhattan::new().distance(&a, &b),
            Manhattan::new().distance(&b, &a),
            1e-10,
            "manhattan symmetric",
        );
    }

    // ── Minkowski ────────────────────────────────────────────────────────────

    /// p=1 must equal Manhattan.
    #[test]
    fn minkowski_p1_equals_manhattan() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b: Vec<f64> = vec![4.0, 6.0, 8.0];
        let mink = Minkowski::new(1).distance(&a, &b);
        let manh = Manhattan::new().distance(&a, &b);
        assert_close(mink, manh, 1e-8, "minkowski p=1 vs manhattan");
    }

    /// p=2 must equal Euclidean.
    #[test]
    fn minkowski_p2_equals_euclidean() {
        let a: Vec<f64> = vec![0.0, 0.0];
        let b: Vec<f64> = vec![3.0, 4.0];
        let mink = Minkowski::new(2).distance(&a, &b);
        let eucl = Euclidian::new().distance(&a, &b);
        assert_close(mink, eucl, 1e-8, "minkowski p=2 vs euclidean");
    }

    /// d(x, x) = 0 for any p.
    #[test]
    fn minkowski_identical_zero() {
        let a: Vec<f64> = vec![2.0, -3.0, 5.0];
        for p in [1u16, 2, 3, 5, 10] {
            let d = Minkowski::new(p).distance(&a, &a);
            assert_close(d, 0.0, 1e-10, &format!("minkowski p={p} identical"));
        }
    }

    /// Large p approximates Chebyshev (max-norm): d → max|aᵢ - bᵢ|.
    #[test]
    fn minkowski_large_p_approaches_chebyshev() {
        let a: Vec<f64> = vec![0.0, 0.0, 0.0];
        let b: Vec<f64> = vec![1.0, 2.0, 3.0]; // max component diff = 3
        let d_large_p = Minkowski::new(50).distance(&a, &b);
        assert!((d_large_p - 3.0).abs() < 0.05, "minkowski p=50 ≈ 3.0, got {d_large_p}");
    }

    // ── Hamming ──────────────────────────────────────────────────────────────

    /// Identical vectors → 0.
    #[test]
    fn hamming_identical_zero() {
        let a: Vec<i32> = vec![1, 0, 1, 1, 0];
        assert_close(Hamming::new().distance(&a, &a), 0.0, 1e-10, "hamming identical");
    }

    /// All different → 1.0 (normalised).
    #[test]
    fn hamming_all_different_one() {
        let a: Vec<i32> = vec![0, 0, 0, 0];
        let b: Vec<i32> = vec![1, 1, 1, 1];
        assert_close(Hamming::new().distance(&a, &b), 1.0, 1e-10, "hamming all different");
    }

    /// Known answer: [1,0,1,0] vs [0,0,1,1] → 2 positions differ → 0.5.
    #[test]
    fn hamming_known_answer_half() {
        let a: Vec<i32> = vec![1, 0, 1, 0];
        let b: Vec<i32> = vec![0, 0, 1, 1];
        assert_close(Hamming::new().distance(&a, &b), 0.5, 1e-10, "hamming half");
    }

    // ── Mahalanobis ──────────────────────────────────────────────────────────

    /// Identity covariance → Mahalanobis equals Euclidean.
    #[test]
    fn mahalanobis_identity_cov_equals_euclidean() {
        use crate::linalg::basic::arrays::ArrayView2;
        // 4 points that span 2D well enough for non-singular cov.
        let data = DenseMatrix::from_2d_array(&[
            &[0.0_f64, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[2.0, 2.0],
        ]).unwrap();

        // Build Mahalanobis from identity covariance explicitly.
        let identity = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 0.0],
            &[0.0, 1.0],
        ]).unwrap();

        let mah: Mahalanobis<f64, DenseMatrix<f64>> =
            Mahalanobis::new_from_covariance(&identity);

        let a = vec![0.0_f64, 0.0];
        let b = vec![3.0_f64, 4.0];

        let mah_d = mah.distance(&a, &b);
        let euc_d = Euclidian::new().distance(&a, &b);
        assert_close(mah_d, euc_d, 1e-6, "mahalanobis(I) == euclidean");
    }

    /// Known answer from doctest: distance ≈ 5.33.
    #[test]
    fn mahalanobis_known_answer_doctest() {
        use crate::linalg::basic::arrays::ArrayView2;
        let data = DenseMatrix::from_2d_array(&[
            &[64.0_f64, 580.0, 29.0],
            &[66.0, 570.0, 33.0],
            &[68.0, 590.0, 37.0],
            &[69.0, 660.0, 46.0],
            &[73.0, 600.0, 55.0],
        ]).unwrap();
        let a = data.mean_by(0);
        let b = vec![66.0, 640.0, 44.0];
        let mah: Mahalanobis<f64, DenseMatrix<f64>> = Mahalanobis::new(&data);
        let d = mah.distance(&a, &b);
        assert_close(d, 5.33, 0.05, "mahalanobis doctest");
    }

    // ── PairwiseDistance ─────────────────────────────────────────────────────

    use crate::metrics::distance::PairwiseDistance;

    /// Struct fields are correctly stored and retrieved.
    #[test]
    fn pairwise_distance_fields() {
        let pd: PairwiseDistance<f64> = PairwiseDistance {
            node: 3,
            neighbour: Some(7),
            distance: Some(1.414),
        };
        assert_eq!(pd.node, 3);
        assert_eq!(pd.neighbour, Some(7));
        assert!((pd.distance.unwrap() - 1.414).abs() < 1e-10);
    }

    /// None-distance sentinel is handled (used to signal "infinite" distance).
    #[test]
    fn pairwise_distance_none_distance() {
        let pd: PairwiseDistance<f64> = PairwiseDistance {
            node: 0,
            neighbour: None,
            distance: None,
        };
        assert!(pd.distance.is_none());
        assert!(pd.neighbour.is_none());
    }

    /// PartialOrd: node with smaller distance is "less than" one with larger.
    #[test]
    fn pairwise_distance_ordering() {
        let close: PairwiseDistance<f64> = PairwiseDistance { node: 0, neighbour: Some(1), distance: Some(1.0) };
        let far:   PairwiseDistance<f64> = PairwiseDistance { node: 0, neighbour: Some(2), distance: Some(9.0) };
        assert!(close < far);
    }

    /// Symmetry of pairwise distances (distance fn itself is symmetric, so two
    /// PairwiseDistance entries for (i→j) and (j→i) should carry equal distances).
    #[test]
    fn pairwise_distance_symmetry_via_euclidean() {
        let a: Vec<f64> = vec![1.0, 2.0];
        let b: Vec<f64> = vec![4.0, 6.0];
        let d_ab = Euclidian::new().distance(&a, &b);
        let d_ba = Euclidian::new().distance(&b, &a);
        let pd_ab: PairwiseDistance<f64> = PairwiseDistance { node: 0, neighbour: Some(1), distance: Some(d_ab) };
        let pd_ba: PairwiseDistance<f64> = PairwiseDistance { node: 1, neighbour: Some(0), distance: Some(d_ba) };
        assert_close(pd_ab.distance.unwrap(), pd_ba.distance.unwrap(), 1e-10, "pairwise symmetric");
    }
}
