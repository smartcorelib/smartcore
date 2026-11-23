//! Stage 3: edge-case & known-answer parity tests for src/decomposition.
//!
//! Covers: PCA, SVD.
//! Tracking issue: #394 / #391.

#[cfg(test)]
mod decomposition_edge_cases {
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::decomposition::pca::{PCA, PCAParameters};
    use crate::decomposition::svd::{SVD, SVDParameters};

    // ── PCA ───────────────────────────────────────────────────────────────────

    /// n_components=1 on 2D data: output must be 1D.
    #[test]
    fn pca_n_components_one() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
        ]).unwrap();
        let model = PCA::fit(&x, PCAParameters::default().with_n_components(1)).unwrap();
        let transformed = model.transform(&x).unwrap();
        let (_, ncols) = transformed.shape();
        assert_eq!(ncols, 1, "n_components=1 should produce 1-column output");
    }

    /// n_components = rank: reconstruction error should be near zero.
    #[test]
    fn pca_full_rank_reconstruction() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[2.0, 1.0],
        ]).unwrap();
        let model = PCA::fit(&x, PCAParameters::default().with_n_components(2)).unwrap();
        let t = model.transform(&x).unwrap();
        let reconstructed = model.inverse_transform(&t).unwrap();
        let (nrows, ncols) = x.shape();
        let mut mse = 0.0_f64;
        for i in 0..nrows {
            for j in 0..ncols {
                let diff = x.get((i, j)) - reconstructed.get((i, j));
                mse += diff * diff;
            }
        }
        mse /= (nrows * ncols) as f64;
        assert!(mse < 1e-6, "full-rank PCA reconstruction MSE={mse} should be ~0");
    }

    /// Rank-deficient input: PCA must not panic.
    #[test]
    fn pca_rank_deficient_input() {
        // Column 2 = column 1: rank = 1.
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 1.0],
            &[2.0, 2.0],
            &[3.0, 3.0],
            &[4.0, 4.0],
        ]).unwrap();
        let result = PCA::fit(&x, PCAParameters::default().with_n_components(1));
        assert!(result.is_ok(), "PCA must not fail on rank-deficient input");
    }

    /// Determinism: same data always gives same projection.
    #[test]
    fn pca_deterministic() {
        let x = DenseMatrix::from_2d_array(&[
            &[2.0_f64, 3.0],
            &[1.0, 5.0],
            &[4.0, 2.0],
            &[6.0, 1.0],
        ]).unwrap();
        let t1 = PCA::fit(&x, PCAParameters::default().with_n_components(1)).unwrap().transform(&x).unwrap();
        let t2 = PCA::fit(&x, PCAParameters::default().with_n_components(1)).unwrap().transform(&x).unwrap();
        let (nrows, _) = t1.shape();
        for i in 0..nrows {
            // Components may flip sign; compare abs values.
            assert!((t1.get((i, 0)).abs() - t2.get((i, 0)).abs()).abs() < 1e-6);
        }
    }

    // ── SVD ───────────────────────────────────────────────────────────────────

    /// n_components=1 on 3-column data: output must be 1-column.
    #[test]
    fn svd_decomp_n_components_one() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[10.0, 11.0, 12.0],
        ]).unwrap();
        let model = SVD::fit(&x, SVDParameters::default().with_n_components(1)).unwrap();
        let transformed = model.transform(&x).unwrap();
        let (_, ncols) = transformed.shape();
        assert_eq!(ncols, 1);
    }

    /// Reconstruction error for full-rank SVD should be near zero.
    #[test]
    fn svd_decomp_full_reconstruction() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 0.0],
            &[0.0, 2.0],
            &[1.0, 2.0],
            &[3.0, 1.0],
        ]).unwrap();
        let model = SVD::fit(&x, SVDParameters::default().with_n_components(2)).unwrap();
        let t = model.transform(&x).unwrap();
        let reconstructed = model.inverse_transform(&t).unwrap();
        let (nrows, ncols) = x.shape();
        let mut mse = 0.0_f64;
        for i in 0..nrows {
            for j in 0..ncols {
                let diff = x.get((i, j)) - reconstructed.get((i, j));
                mse += diff * diff;
            }
        }
        mse /= (nrows * ncols) as f64;
        assert!(mse < 1e-6, "full-rank SVD reconstruction MSE={mse} should be ~0");
    }

    /// Rank-deficient input must not panic.
    #[test]
    fn svd_decomp_rank_deficient_no_panic() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 1.0],
            &[2.0, 2.0],
            &[3.0, 3.0],
        ]).unwrap();
        assert!(SVD::fit(&x, SVDParameters::default().with_n_components(1)).is_ok());
    }
}
