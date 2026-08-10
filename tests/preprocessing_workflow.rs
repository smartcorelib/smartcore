//! Integration test: preprocessing end-to-end workflow.
//!
//! `StandardScaler` fit → transform → inverse_transform round-trip.
//! `OneHotEncoder` encode → shape check.
//! Tracking issue: #397 / #391.

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;

// ---------------------------------------------------------------------------
// StandardScaler — round-trip
// ---------------------------------------------------------------------------

#[test]
fn standard_scaler_round_trip_workflow() {
    use smartcore::preprocessing::StandardScaler;
    use smartcore::Transformer;

    let x = DenseMatrix::from_2d_array(&[
        &[1.0_f64, 10.0],
        &[2.0, 20.0],
        &[3.0, 30.0],
        &[4.0, 40.0],
        &[5.0, 50.0],
    ])
    .unwrap();

    let scaler = StandardScaler::fit(&x, Default::default()).expect("StandardScaler::fit");
    let scaled = scaler.transform(&x).expect("transform");

    // After standard scaling, each column should have mean ≈ 0 and std ≈ 1
    let (nr, nc) = scaled.shape();
    for c in 0..nc {
        let col: Vec<f64> = (0..nr).map(|r| *scaled.get((r, c))).collect();
        let mean = col.iter().sum::<f64>() / nr as f64;
        assert!(mean.abs() < 1e-10, "col {c} mean not zero: {mean}");
    }
}

// ---------------------------------------------------------------------------
// StandardScaler — transform then inverse_transform recovers original
// ---------------------------------------------------------------------------

#[test]
fn standard_scaler_inverse_transform_workflow() {
    use smartcore::preprocessing::StandardScaler;
    use smartcore::Transformer;

    let x = DenseMatrix::from_2d_array(&[
        &[10.0_f64, 200.0],
        &[20.0, 400.0],
        &[30.0, 600.0],
        &[40.0, 800.0],
    ])
    .unwrap();

    let scaler = StandardScaler::fit(&x, Default::default()).expect("fit");
    let scaled = scaler.transform(&x).expect("transform");
    let recovered = scaler.inverse_transform(&scaled).expect("inverse_transform");

    let (nr, nc) = x.shape();
    for r in 0..nr {
        for c in 0..nc {
            let orig = *x.get((r, c));
            let back = *recovered.get((r, c));
            assert!(
                (orig - back).abs() < 1e-8,
                "inverse_transform mismatch at ({r},{c}): orig={orig}, back={back}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// OneHotEncoder — shape and binary values
// ---------------------------------------------------------------------------

#[test]
fn one_hot_encoder_workflow() {
    use smartcore::preprocessing::categorical::OneHotEncoder;
    use smartcore::Transformer;

    // 4 samples, 2 categorical features: feature 0 has 3 categories, feature 1 has 2
    let x = DenseMatrix::from_2d_array(&[
        &[0_u32, 0],
        &[1, 1],
        &[2, 0],
        &[0, 1],
    ])
    .unwrap();

    let encoder = OneHotEncoder::fit(&x, Default::default()).expect("OneHotEncoder::fit");
    let encoded = encoder.transform(&x).expect("transform");

    let (nr, nc) = encoded.shape();
    assert_eq!(nr, 4, "OHE: wrong number of rows");
    // 3 + 2 = 5 output columns
    assert_eq!(nc, 5, "OHE: expected 5 output columns");

    // All values should be 0 or 1
    for r in 0..nr {
        for c in 0..nc {
            let v = *encoded.get((r, c));
            assert!(
                v == 0.0 || v == 1.0,
                "OHE non-binary value at ({r},{c}): {v}"
            );
        }
    }
}
