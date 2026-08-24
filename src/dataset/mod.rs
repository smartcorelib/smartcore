#![allow(clippy::ptr_arg, clippy::needless_range_loop)]
//! Datasets
//!
//! In this module you will find small datasets that are used in `smartcore` mostly for demonstration purposes.
pub mod boston;
pub mod breast_cancer;
pub mod diabetes;
pub mod digits;
pub mod generator;
pub mod iris;

#[cfg(not(target_arch = "wasm32"))]
use crate::numbers::{basenum::Number, realnum::RealNumber};
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::io::prelude::*;

/// Dataset
#[derive(Debug)]
pub struct Dataset<X, Y> {
    /// data in one-dimensional array.
    pub data: Vec<X>,
    /// target values or class labels.
    pub target: Vec<Y>,
    /// number of samples (number of rows in matrix form).
    pub num_samples: usize,
    /// number of features (number of columns in matrix form).
    pub num_features: usize,
    /// names of dependent variables.
    pub feature_names: Vec<String>,
    /// names of target variables.
    pub target_names: Vec<String>,
    /// dataset description
    pub description: String,
}

impl<X, Y> Dataset<X, Y> {
    /// Reshape data into a two-dimensional matrix
    #[must_use]
    pub fn as_matrix(&self) -> Vec<Vec<&X>> {
        let mut result: Vec<Vec<&X>> = Vec::with_capacity(self.num_samples);

        for r in 0..self.num_samples {
            let mut row = Vec::with_capacity(self.num_features);
            for c in 0..self.num_features {
                row.push(&self.data[r * self.num_features + c]);
            }
            result.push(row);
        }

        result
    }
}

// Running this in wasm throws: operation not supported on this platform.
#[cfg(not(target_arch = "wasm32"))]
#[expect(dead_code)]
pub(crate) fn serialize_data<X: Number + RealNumber, Y: RealNumber>(
    dataset: &Dataset<X, Y>,
    filename: &str,
) -> Result<(), io::Error> {
    match File::create(filename) {
        Ok(mut file) => {
            // Write header as fixed-width u64 (little-endian) so the .xy files
            // can be read correctly on any target width, including wasm32.
            file.write_all(&(dataset.num_features as u64).to_le_bytes())?;
            file.write_all(&(dataset.num_samples as u64).to_le_bytes())?;
            let x: Vec<u8> = dataset
                .data
                .iter()
                .copied()
                .flat_map(|f| f.to_f32_bits().to_le_bytes().to_vec())
                .collect();
            file.write_all(&x)?;
            let y: Vec<u8> = dataset
                .target
                .iter()
                .copied()
                .flat_map(|f| f.to_f32_bits().to_le_bytes().to_vec())
                .collect();
            file.write_all(&y)?;
        }
        Err(why) => panic!("couldn't create {filename}: {why}"),
    }
    Ok(())
}

/// Deserialise a `.xy` dataset blob embedded via `include_bytes!`.
///
/// # Wire format
/// ```text
/// [u64 LE: num_features][u64 LE: num_samples]
/// [f32 LE × (num_features * num_samples)]   <- X matrix, row-major
/// [f32 LE × num_samples]                    <- y vector
/// ```
///
/// The header uses a **fixed 8-byte (u64) width** regardless of the host
/// pointer size.  Previous versions used `usize`, which is 4 bytes on
/// `wasm32` but 8 bytes on x86-64 — meaning the `.xy` files (generated
/// on x86-64) could not be parsed under WASM and every dataset test
/// returned `data.len() == 0`.
pub(crate) fn deserialize_data(
    bytes: &[u8],
) -> Result<(Vec<f32>, Vec<f32>, usize, usize), io::Error> {
    // Header: two u64 fields, each 8 bytes, platform-independent.
    const FIELD_SIZE: usize = std::mem::size_of::<u64>(); // always 8
    const HEADER_LEN: usize = 2 * FIELD_SIZE; // always 16

    // Reject obviously-truncated buffers before reading any fields.
    if bytes.len() < HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "deserialize_data: buffer too small for header (need {HEADER_LEN} bytes, got {})",
                bytes.len()
            ),
        ));
    }

    let (num_samples, num_features) = {
        let mut buf8 = [0u8; FIELD_SIZE];
        buf8.copy_from_slice(&bytes[0..FIELD_SIZE]);
        let num_features = u64::from_le_bytes(buf8) as usize;
        buf8.copy_from_slice(&bytes[FIELD_SIZE..HEADER_LEN]);
        let num_samples = u64::from_le_bytes(buf8) as usize;
        (num_samples, num_features)
    };

    // Guard against integer overflow in num_samples * num_features.
    let num_x_values = num_samples.checked_mul(num_features).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "deserialize_data: num_samples * num_features overflows usize",
        )
    })?;

    // Validate the total byte length before any allocation.
    // Layout: HEADER_LEN + num_x_values * 4 + num_samples * 4
    let x_bytes = num_x_values.checked_mul(4).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "deserialize_data: x byte range overflows usize",
        )
    })?;
    let y_bytes = num_samples.checked_mul(4).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "deserialize_data: y byte range overflows usize",
        )
    })?;
    let expected_len = HEADER_LEN
        .checked_add(x_bytes)
        .and_then(|n| n.checked_add(y_bytes))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "deserialize_data: total expected length overflows usize",
            )
        })?;
    if bytes.len() < expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "deserialize_data: buffer too short (expected {expected_len} bytes, got {})",
                bytes.len()
            ),
        ));
    }

    let mut x = Vec::with_capacity(num_x_values);
    let mut y = Vec::with_capacity(num_samples);

    let mut buf4 = [0u8; 4];
    let mut c = HEADER_LEN;

    for _ in 0..num_x_values {
        buf4.copy_from_slice(&bytes[c..(c + 4)]);
        let v = f32::from_bits(u32::from_le_bytes(buf4));
        if !v.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "deserialize_data: non-finite value in feature data (bits: {:#010x})",
                    u32::from_le_bytes(buf4)
                ),
            ));
        }
        x.push(v);
        c += 4;
    }

    for _ in 0..num_samples {
        buf4.copy_from_slice(&bytes[c..(c + 4)]);
        let v = f32::from_bits(u32::from_le_bytes(buf4));
        if !v.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "deserialize_data: non-finite value in target data (bits: {:#010x})",
                    u32::from_le_bytes(buf4)
                ),
            ));
        }
        y.push(v);
        c += 4;
    }

    Ok((x, y, num_samples, num_features))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn as_matrix() {
        let dataset = Dataset {
            data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            target: vec![1, 2, 3],
            num_samples: 2,
            num_features: 5,
            feature_names: vec![],
            target_names: vec![],
            description: "".to_string(),
        };

        let m = dataset.as_matrix();

        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 5);
        assert_eq!(*m[1][3], 9);
    }

    // deserialize_data unit tests — run on native AND wasm32.
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn deserialize_data_too_short() {
        let result = deserialize_data(&[0u8; 4]);
        assert!(result.is_err());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn deserialize_data_truncated_body() {
        // Valid header (u64 LE): 1 feature, 1 sample — but no payload bytes.
        // Header is 16 bytes; expected total = 16 + 4 (x) + 4 (y) = 24.
        let mut buf = vec![0u8; 16];
        buf[0..8].copy_from_slice(&1u64.to_le_bytes()); // num_features = 1
        buf[8..16].copy_from_slice(&1u64.to_le_bytes()); // num_samples  = 1
        let result = deserialize_data(&buf);
        assert!(result.is_err());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn deserialize_data_nan_rejected() {
        // Construct a valid 1×1 dataset where the feature value is NaN.
        let nan_bits: u32 = f32::NAN.to_bits();
        let mut buf = vec![0u8; 16 + 4 + 4];
        buf[0..8].copy_from_slice(&1u64.to_le_bytes()); // num_features = 1
        buf[8..16].copy_from_slice(&1u64.to_le_bytes()); // num_samples  = 1
        buf[16..20].copy_from_slice(&nan_bits.to_le_bytes()); // x[0] = NaN
        buf[20..24].copy_from_slice(&1.0f32.to_le_bytes()); // y[0] = 1.0
        let result = deserialize_data(&buf);
        assert!(result.is_err());
    }

    /// Smoke-test that a correctly-formed 1×1 round-trip parses on every
    /// target width, including wasm32.
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn deserialize_data_roundtrip_1x1() {
        let x_val = 3.14f32;
        let y_val = 1.0f32;
        let mut buf = vec![0u8; 16 + 4 + 4];
        buf[0..8].copy_from_slice(&1u64.to_le_bytes()); // num_features = 1
        buf[8..16].copy_from_slice(&1u64.to_le_bytes()); // num_samples  = 1
        buf[16..20].copy_from_slice(&x_val.to_bits().to_le_bytes());
        buf[20..24].copy_from_slice(&y_val.to_bits().to_le_bytes());
        let (x, y, ns, nf) = deserialize_data(&buf).expect("roundtrip must succeed");
        assert_eq!(ns, 1);
        assert_eq!(nf, 1);
        assert_eq!(x.len(), 1);
        assert_eq!(y.len(), 1);
        assert!((x[0] - x_val).abs() < 1e-6);
        assert!((y[0] - y_val).abs() < 1e-6);
    }
}
