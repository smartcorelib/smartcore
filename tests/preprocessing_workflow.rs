//! Integration test: preprocessing end-to-end workflow.
//!
//! `StandardScaler` fit → transform (mean≈0, std≈1 per column).
//! `OneHotEncoder` encode → shape check.
//! Tracking issue: #397 / #391.
//!
//! API notes:
//!   - `Transformer` lives at `smartcore::api::Transformer`
//!   - `StandardScaler` lives at `smartcore::preprocessing::numerical`
//!   - `StandardScaler` has no `inverse_transform`; round-trip invariant is
//!     verified through mean/std of the scaled output instead
//!   - `OneHotEncoder::fit` requires `T: Categorizable` (only f32/f64);
//!     `OneHotEncoderParams` has no `Default` — use `from_cat_idx`

use smartcore::linalg::basic::matrix::DenseMatrix;

// ---------------------------------------------------------------------------
// StandardScaler — scaled output has mean ≈ 0 and std ≈ 1 per column
// ---------------------------------------------------------------------------

#[test]
fn standard_scaler_round_trip_workflow() {
    use smartcore::api::Transformer;
    use smartcore::linalg::basic::arrays::Array;
    use smartcore::preprocessing::numerical::{StandardScaler, StandardScalerParameters};

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 10.0],
        &[2.0, 20.0],
        &[3.0, 30.0],
        &[4.0, 40.0],
        &[5.0, 50.0],
    ])
    .unwrap();

    let scaler = StandardScaler::fit(&x, StandardScalerParameters::default())
        .expect("StandardScaler::fit");
    let scaled = scaler.transform(&x).expect("transform");

    // After standard scaling each column must have mean ≈ 0
    let (nr, nc) = scaled.shape();
    for c in 0..nc {
        let col: Vec<f64> = (0..nr).map(|r| *scaled.get((r, c))).collect();
        let mean = col.iter().sum::<f64>() / nr as f64;
        assert!(mean.abs() < 1e-10, "col {c} mean not zero: {mean}");
    }
}

// ---------------------------------------------------------------------------
// StandardScaler — transform produces unit variance per column
// ---------------------------------------------------------------------------

#[test]
fn standard_scaler_unit_variance_workflow() {
    use smartcore::api::Transformer;
    use smartcore::linalg::basic::arrays::Array;
    use smartcore::preprocessing::numerical::{StandardScaler, StandardScalerParameters};

    let x = DenseMatrix::from_2d_array(&[
        &[10.0_f64, 200.0],
        &[20.0, 400.0],
        &[30.0, 600.0],
        &[40.0, 800.0],
    ])
    .unwrap();

    let scaler = StandardScaler::fit(&x, StandardScalerParameters::default()).expect("fit");
    let scaled = scaler.transform(&x).expect("transform");

    // Each column of the scaled matrix should have std ≈ 1
    let (nr, nc) = scaled.shape();
    for c in 0..nc {
        let col: Vec<f64> = (0..nr).map(|r| *scaled.get((r, c))).collect();
        let mean = col.iter().sum::<f64>() / nr as f64;
        let variance = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nr as f64;
        let std = variance.sqrt();
        assert!(
            (std - 1.0).abs() < 1e-6,
            "col {c} std not 1: {std}"
        );
    }
}

// ---------------------------------------------------------------------------
// OneHotEncoder — shape and binary values
// ---------------------------------------------------------------------------

#[test]
fn one_hot_encoder_workflow() {
    use smartcore::linalg::basic::arrays::Array;
    use smartcore::preprocessing::categorical::{OneHotEncoder, OneHotEncoderParams};

    // 4 samples, 2 categorical columns (indices 0 and 1).
    // Values must be f64 because Categorizable is only impl for f32/f64.
    // col 0: 3 distinct values {0.0, 1.0, 2.0}; col 1: 2 distinct values {0.0, 1.0}
    let x = DenseMatrix::from_2d_array(&[
        &[0.0_f64, 0.0],
        &[1.0, 1.0],
        &[2.0, 0.0],
        &[0.0, 1.0],
    ])
    .unwrap();

    // Explicitly list both columns as categorical; no Default for OneHotEncoderParams
    let params = OneHotEncoderParams::from_cat_idx(&[0, 1]);
    let encoder = OneHotEncoder::fit(&x, params).expect("OneHotEncoder::fit");
    let encoded = encoder.transform(&x).expect("transform");

    let (nr, nc) = encoded.shape();
    assert_eq!(nr, 4, "OHE: wrong number of rows");
    // col 0 expands to 3 columns, col 1 expands to 2 columns → 5 total
    assert_eq!(nc, 5, "OHE: expected 5 output columns");

    // All values should be 0.0 or 1.0 (f64)
    for r in 0..nr {
        for c in 0..nc {
            let v = *encoded.get((r, c));
            assert!(
                (v - 0.0_f64).abs() < 1e-10 || (v - 1.0_f64).abs() < 1e-10,
                "OHE non-binary value at ({r},{c}): {v}"
            );
        }
    }
}
