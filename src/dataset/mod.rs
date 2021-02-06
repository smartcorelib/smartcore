//! Datasets
//!
//! In this module you will find small datasets that are used in SmartCore for demonstration purpose mostly.
pub mod boston;
pub mod breast_cancer;
pub mod diabetes;
pub mod digits;
pub mod generator;
pub mod iris;

use crate::math::num::RealNumber;
use std::fs::File;
use std::io;
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

#[allow(dead_code)]
pub(crate) fn serialize_data<X: RealNumber, Y: RealNumber>(
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
                .flat_map(|f| f.to_f32_bits().to_le_bytes().to_vec().into_iter())
                .collect();
            file.write_all(&x)?;
            let y: Vec<u8> = dataset
                .target
                .iter()
                .copied()
                .flat_map(|f| f.to_f32_bits().to_le_bytes().to_vec().into_iter())
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
    let (num_samples, num_features) = {
        let mut buffer = [0u8; 8];
        buffer.copy_from_slice(&bytes[0..8]);
        let num_features = usize::from_le_bytes(buffer);
        buffer.copy_from_slice(&bytes[8..16]);
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

impl<X: Copy + std::fmt::Debug, Y: Copy + std::fmt::Debug> std::fmt::Display for Dataset<X, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.num_features == self.feature_names.len() {
            struct Target<Y> {
                name: String,
                value: Y,
            }
            struct Feature<X> {
                name: String,
                value: X,
            }
            struct DataPoint<X, Y> {
                labels: Vec<Target<Y>>,
                features: Vec<Feature<X>>,
            }
            impl<X: Copy + std::fmt::Debug, Y: Copy + std::fmt::Debug> std::fmt::Display for DataPoint<X, Y> {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(
                        f,
                        "{} : {}",
                        self.labels
                            .iter()
                            .map(|target| format!("{}:{:?}, ", target.name, target.value))
                            .collect::<String>(),
                        self.features
                            .iter()
                            .map(|feature| format!("{}:{:?}, ", feature.name, feature.value))
                            .collect::<String>()
                    )
                }
            }
            let mut datapoints = Vec::new();
            for sample_index in 0..self.num_samples {
                let mut features = Vec::new();
                for feature_index in 0..self.feature_names.len() {
                    features.push(Feature {
                        name: self.feature_names[feature_index].to_owned(),
                        value: self.data[sample_index * self.num_features + feature_index],
                    });
                }
                let mut targets = Vec::new();
                for target_index in 0..self.target_names.len() {
                    targets.push(Target {
                        name: self.target_names[target_index].to_owned(),
                        value: self.target[sample_index * self.target_names.len() + target_index],
                    });
                }
                datapoints.push(DataPoint {
                    labels: targets,
                    features,
                })
            }
            let mut out = format!("{}\n", self.description);
            for point in datapoints {
                out.push_str(&format!("{}\n", point));
            }
            write!(f, "{}", out)
        } else {
            write!(f, "{:?}", self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn display() {
        let dataset = iris::load_dataset();
        println!("{}", dataset);
    }
}
