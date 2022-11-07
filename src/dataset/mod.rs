//! Datasets
//!
//! In this module you will find small datasets that are used in smartcore mostly for demonstration purposes.
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
        Err(why) => panic!("couldn't create {}: {}", filename, why),
    }
    Ok(())
}

pub(crate) fn deserialize_data(
    bytes: &[u8],
) -> Result<(Vec<f32>, Vec<f32>, usize, usize), io::Error> {
    // read the same file back into a Vec of bytes
    const USIZE_SIZE: usize = std::mem::size_of::<usize>();
    let (num_samples, num_features) = {
        let mut buffer = [0u8; USIZE_SIZE];
        buffer.copy_from_slice(&bytes[0..USIZE_SIZE]);
        let num_features = usize::from_le_bytes(buffer);
        buffer.copy_from_slice(&bytes[8..8 + USIZE_SIZE]);
        let num_samples = usize::from_le_bytes(buffer);
        (num_samples, num_features)
    };

    let mut x = Vec::with_capacity(num_samples * num_features);
    let mut y = Vec::with_capacity(num_samples);

    let mut buffer = [0u8; 4];
    let mut c = 16;
    for _ in 0..(num_samples * num_features) {
        buffer.copy_from_slice(&bytes[c..(c + 4)]);
        x.push(f32::from_bits(u32::from_le_bytes(buffer)));
        c += 4;
    }

    for _ in 0..(num_samples) {
        buffer.copy_from_slice(&bytes[c..(c + 4)]);
        y.push(f32::from_bits(u32::from_le_bytes(buffer)));
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
}
