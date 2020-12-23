//! # KFold
//!
//! Defines k-fold cross validator.

use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::model_selection::BaseKFold;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// K-Folds cross-validator
pub struct KFold {
    /// Number of folds. Must be at least 2.
    pub n_splits: usize, // cannot exceed std::usize::MAX
    /// Whether to shuffle the data before splitting into batches
    pub shuffle: bool,
}

impl KFold {
    fn test_indices<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<Vec<usize>> {
        // number of samples (rows) in the matrix
        let n_samples: usize = x.shape().0;

        // initialise indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle {
            indices.shuffle(&mut thread_rng());
        }
        //  return a new array of given shape n_split, filled with each element of n_samples divided by n_splits.
        let mut fold_sizes = vec![n_samples / self.n_splits; self.n_splits];

        // increment by one if odd
        for fold_size in fold_sizes.iter_mut().take(n_samples % self.n_splits) {
            *fold_size += 1;
        }

        // generate the right array of arrays for test indices
        let mut return_values: Vec<Vec<usize>> = Vec::with_capacity(self.n_splits);
        let mut current: usize = 0;
        for fold_size in fold_sizes.drain(..) {
            let stop = current + fold_size;
            return_values.push(indices[current..stop].to_vec());
            current = stop
        }

        return_values
    }

    fn test_masks<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<Vec<bool>> {
        let mut return_values: Vec<Vec<bool>> = Vec::with_capacity(self.n_splits);
        for test_index in self.test_indices(x).drain(..) {
            // init mask
            let mut test_mask = vec![false; x.shape().0];
            // set mask's indices to true according to test indices
            for i in test_index {
                test_mask[i] = true; // can be implemented with map()
            }
            return_values.push(test_mask);
        }
        return_values
    }
}

impl Default for KFold {
    fn default() -> KFold {
        KFold {
            n_splits: 3,
            shuffle: true,
        }
    }
}

impl KFold {
    /// Number of folds. Must be at least 2.
    pub fn with_n_splits(mut self, n_splits: usize) -> Self {
        self.n_splits = n_splits;
        self
    }
    /// Whether to shuffle the data before splitting into batches
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
}

/// An iterator over indices that split data into training and test set.
pub struct KFoldIter {
    indices: Vec<usize>,
    test_indices: Vec<Vec<bool>>,
}

impl Iterator for KFoldIter {
    type Item = (Vec<usize>, Vec<usize>);

    fn next(&mut self) -> Option<(Vec<usize>, Vec<usize>)> {
        self.test_indices.pop().map(|test_index| {
            let train_index = self
                .indices
                .iter()
                .enumerate()
                .filter(|&(idx, _)| !test_index[idx])
                .map(|(idx, _)| idx)
                .collect::<Vec<usize>>(); // filter train indices out according to mask
            let test_index = self
                .indices
                .iter()
                .enumerate()
                .filter(|&(idx, _)| test_index[idx])
                .map(|(idx, _)| idx)
                .collect::<Vec<usize>>(); // filter tests indices out according to mask

            (train_index, test_index)
        })
    }
}

/// Abstract class for all KFold functionalities
impl BaseKFold for KFold {
    type Output = KFoldIter;

    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Self::Output {
        if self.n_splits < 2 {
            panic!("Number of splits is too small: {}", self.n_splits);
        }
        let n_samples: usize = x.shape().0;
        let indices: Vec<usize> = (0..n_samples).collect();
        let mut test_indices = self.test_masks(x);
        test_indices.reverse();

        KFoldIter {
            indices,
            test_indices,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn run_kfold_return_test_indices_simple() {
        let k = KFold {
            n_splits: 3,
            shuffle: false,
        };
        let x: DenseMatrix<f64> = DenseMatrix::rand(33, 100);
        let test_indices = k.test_indices(&x);

        assert_eq!(test_indices[0], (0..11).collect::<Vec<usize>>());
        assert_eq!(test_indices[1], (11..22).collect::<Vec<usize>>());
        assert_eq!(test_indices[2], (22..33).collect::<Vec<usize>>());
    }

    #[test]
    fn run_kfold_return_test_indices_odd() {
        let k = KFold {
            n_splits: 3,
            shuffle: false,
        };
        let x: DenseMatrix<f64> = DenseMatrix::rand(34, 100);
        let test_indices = k.test_indices(&x);

        assert_eq!(test_indices[0], (0..12).collect::<Vec<usize>>());
        assert_eq!(test_indices[1], (12..23).collect::<Vec<usize>>());
        assert_eq!(test_indices[2], (23..34).collect::<Vec<usize>>());
    }

    #[test]
    fn run_kfold_return_test_mask_simple() {
        let k = KFold {
            n_splits: 2,
            shuffle: false,
        };
        let x: DenseMatrix<f64> = DenseMatrix::rand(22, 100);
        let test_masks = k.test_masks(&x);

        for t in &test_masks[0][0..11] {
            // TODO: this can be prob done better
            assert_eq!(*t, true)
        }
        for t in &test_masks[0][11..22] {
            assert_eq!(*t, false)
        }

        for t in &test_masks[1][0..11] {
            assert_eq!(*t, false)
        }
        for t in &test_masks[1][11..22] {
            assert_eq!(*t, true)
        }
    }

    #[test]
    fn run_kfold_return_split_simple() {
        let k = KFold {
            n_splits: 2,
            shuffle: false,
        };
        let x: DenseMatrix<f64> = DenseMatrix::rand(22, 100);
        let train_test_splits: Vec<(Vec<usize>, Vec<usize>)> = k.split(&x).collect();

        assert_eq!(train_test_splits[0].1, (0..11).collect::<Vec<usize>>());
        assert_eq!(train_test_splits[0].0, (11..22).collect::<Vec<usize>>());
        assert_eq!(train_test_splits[1].0, (0..11).collect::<Vec<usize>>());
        assert_eq!(train_test_splits[1].1, (11..22).collect::<Vec<usize>>());
    }

    #[test]
    fn run_kfold_return_split_simple_shuffle() {
        let k = KFold {
            n_splits: 2,
            ..KFold::default()
        };
        let x: DenseMatrix<f64> = DenseMatrix::rand(23, 100);
        let train_test_splits: Vec<(Vec<usize>, Vec<usize>)> = k.split(&x).collect();

        assert_eq!(train_test_splits[0].1.len(), 12_usize);
        assert_eq!(train_test_splits[0].0.len(), 11_usize);
        assert_eq!(train_test_splits[1].0.len(), 12_usize);
        assert_eq!(train_test_splits[1].1.len(), 11_usize);
    }

    #[test]
    fn numpy_parity_test() {
        let k = KFold {
            n_splits: 3,
            shuffle: false,
        };
        let x: DenseMatrix<f64> = DenseMatrix::rand(10, 4);
        let expected: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![4, 5, 6, 7, 8, 9], vec![0, 1, 2, 3]),
            (vec![0, 1, 2, 3, 7, 8, 9], vec![4, 5, 6]),
            (vec![0, 1, 2, 3, 4, 5, 6], vec![7, 8, 9]),
        ];
        for ((train, test), (expected_train, expected_test)) in
            k.split(&x).into_iter().zip(expected)
        {
            assert_eq!(test, expected_test);
            assert_eq!(train, expected_train);
        }
    }

    #[test]
    fn numpy_parity_test_shuffle() {
        let k = KFold {
            n_splits: 3,
            ..KFold::default()
        };
        let x: DenseMatrix<f64> = DenseMatrix::rand(10, 4);
        let expected: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![4, 5, 6, 7, 8, 9], vec![0, 1, 2, 3]),
            (vec![0, 1, 2, 3, 7, 8, 9], vec![4, 5, 6]),
            (vec![0, 1, 2, 3, 4, 5, 6], vec![7, 8, 9]),
        ];
        for ((train, test), (expected_train, expected_test)) in
            k.split(&x).into_iter().zip(expected)
        {
            assert_eq!(test.len(), expected_test.len());
            assert_eq!(train.len(), expected_train.len());
        }
    }
}
