//! Datasets
//!
//! In this module you will find small datasets that are used in SmartCore for demonstration purpose mostly.

/// The Boston Housing Dataset
pub mod boston;
/// Iris flower data set
pub mod iris;

pub mod diabetes;

/// Dataset
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
}
