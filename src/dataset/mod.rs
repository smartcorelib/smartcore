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
#[allow(dead_code)]
pub(crate) fn serialize_data<X: Number + RealNumber, Y: RealNumber>(
    dataset: &Dataset<X, Y>,
    filename: &str,
) -> Result<(), io::Error> {
    match File::create(filename) {
        Ok(mut file) => {
            file.write_all(&dataset.num_features.to_le_bytes())?;
            file.write_all(&dataset.num_samples.to_le_bytes())?;
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

pub(crate) fn deserialize_data(
    bytes: &[u8],
) -> Result<(Vec<f32>, Vec<f32>, usize, usize), io::Error> {
    const USIZE_SIZE: usize = std::mem::size_of::<usize>();
    // Header occupies two usize fields (num_features + num_samples)
    const HEADER_LEN: usize = 2 * USIZE_SIZE;

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
        let mut buffer = [0u8; USIZE_SIZE];
        buffer.copy_from_slice(&bytes[0..USIZE_SIZE]);
        let num_features = usize::from_le_bytes(buffer);
        buffer.copy_from_slice(&bytes[USIZE_SIZE..HEADER_LEN]);
        let num_samples = usize::from_le_bytes(buffer);
        (num_samples, num_features)
    };

    // Guard against integer overflow in num_samples * num_features.
    let num_x_values = num_samples
        .checked_mul(num_features)
        .ok_or_else(|| {
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

    let mut buffer = [0u8; 4];
    let mut c = HEADER_LEN;

    for _ in 0..num_x_values {
        buffer.copy_from_slice(&bytes[c..(c + 4)]);
        let v = f32::from_bits(u32::from_le_bytes(buffer));
        if !v.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("deserialize_data: non-finite value in feature data (bits: {:#010x})", u32::from_le_bytes(buffer)),
            ));
        }
        x.push(v);
        c += 4;
    }

    for _ in 0..num_samples {
        buffer.copy_from_slice(&bytes[c..(c + 4)]);
        let v = f32::from_bits(u32::from_le_bytes(buffer));
        if !v.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("deserialize_data: non-finite value in target data (bits: {:#010x})", u32::from_le_bytes(buffer)),
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

    #[test]
    fn deserialize_data_too_short() {
        let result = deserialize_data(&[0u8; 4]);
        assert!(result.is_err());
    }

    #[test]
    fn deserialize_data_truncated_body() {
        // Valid header: 1 sample, 1 feature, but no payload bytes
        let mut buf = vec![0u8; 16];
        buf[0..8].copy_from_slice(&1usize.to_le_bytes()); // num_features = 1
        buf[8..16].copy_from_slice(&1usize.to_le_bytes()); // num_samples = 1
        // Expected total: 16 + 4 (x) + 4 (y) = 24 bytes, but we only supply 16
        let result = deserialize_data(&buf);
        assert!(result.is_err());
    }

    #[test]
    fn deserialize_data_nan_rejected() {
        // Construct a valid 1x1 dataset where the feature value is NaN
        let nan_bits: u32 = f32::NAN.to_bits();
        let mut buf = vec![0u8; 16 + 4 + 4];
        buf[0..8].copy_from_slice(&1usize.to_le_bytes()); // num_features = 1
        buf[8..16].copy_from_slice(&1usize.to_le_bytes()); // num_samples = 1
        buf[16..20].copy_from_slice(&nan_bits.to_le_bytes()); // x[0] = NaN
        buf[20..24].copy_from_slice(&1.0f32.to_le_bytes()); // y[0] = 1.0
        let result = deserialize_data(&buf);
        assert!(result.is_err());
    }
}
