//! # Model Selection methods
//!
//! In statistics and machine learning we usually split our data into multiple subsets: training data and testing data (and sometimes to validate),
//! and fit our model on the train data, in order to make predictions on the test data. We do that to avoid overfitting or underfitting model to our data.
//! Overfitting is bad because the model we trained fits trained data too well and can’t make any inferences on new data.
//! Underfitted is bad because the model is undetrained and does not fit the training data well.
//! Splitting data into multiple subsets helps to find the right combination of hyperparameters, estimate model performance and choose the right model for
//! your data.
//!
//! In SmartCore you can split your data into training and test datasets using `train_test_split` function.
extern crate rand;

use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

/// Splits data into 2 disjoint datasets.
/// * `x` - features, matrix of size _NxM_ where _N_ is number of samples and _M_ is number of attributes.
/// * `y` - target values, should be of size _M_
/// * `test_size`, (0, 1] - the proportion of the dataset to include in the test split.
pub fn train_test_split<T: RealNumber, M: Matrix<T>>(
    x: &M,
    y: &M::RowVector,
    test_size: f32,
) -> (M, M, M::RowVector, M::RowVector) {
    if x.shape().0 != y.len() {
        panic!(
            "x and y should have the same number of samples. |x|: {}, |y|: {}",
            x.shape().0,
            y.len()
        );
    }

    if test_size <= 0. || test_size > 1.0 {
        panic!("test_size should be between 0 and 1");
    }

    let n = y.len();
    let m = x.shape().1;

    let mut rng = rand::thread_rng();
    let mut n_test = 0;
    let mut index = vec![false; n];

    for i in 0..n {
        let p_test: f32 = rng.gen();
        if p_test <= test_size {
            index[i] = true;
            n_test += 1;
        }
    }

    let n_train = n - n_test;

    let mut x_train = M::zeros(n_train, m);
    let mut x_test = M::zeros(n_test, m);
    let mut y_train = M::RowVector::zeros(n_train);
    let mut y_test = M::RowVector::zeros(n_test);

    let mut r_train = 0;
    let mut r_test = 0;

    for r in 0..n {
        if index[r] {
            //sample belongs to test
            for c in 0..m {
                x_test.set(r_test, c, x.get(r, c));
                y_test.set(r_test, y.get(r));
            }
            r_test += 1;
        } else {
            for c in 0..m {
                x_train.set(r_train, c, x.get(r, c));
                y_train.set(r_train, y.get(r));
            }
            r_train += 1;
        }
    }

    (x_train, x_test, y_train, y_test)
}

///
/// KFold Cross-Validation
///
pub trait BaseKFold {
    /// Returns integer indices corresponding to test sets
    fn test_indices<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<Vec<usize>>;

    /// Returns masksk corresponding to test sets
    fn test_masks<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<Vec<bool>>;

    /// Return a tuple containing the the training set indices for that split and
    /// the testing set indices for that split.
    fn split<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<(Vec<usize>, Vec<usize>)>;
}

///
/// An implementation of KFold
///
pub struct KFold {
    n_splits: i32, // cannot exceed std::usize::MAX
    shuffle: bool,
    // TODO: to be implemented later
    // random_state: i32,
}

impl Default for KFold {
    fn default() -> KFold {
        KFold {
            n_splits: 3i32,
            shuffle: true,
        }
    }
}

///
/// Abstract class for all KFold functionalities
///
impl BaseKFold for KFold {
    fn test_indices<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<Vec<usize>> {
        // number of samples (rows) in the matrix
        let n_samples: usize = x.shape().0;

        // initialise indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle == true {
            indices.shuffle(&mut thread_rng());
        }
        //  return a new array of given shape n_split, filled with each element of n_samples divided by n_splits.
        let mut fold_sizes = vec![n_samples / self.n_splits as usize; self.n_splits as usize];

        // increment by one if odd
        for i in 0..(n_samples % self.n_splits as usize) {
            fold_sizes[i] = fold_sizes[i] + 1;
        }

        // generate the right array of arrays for test indices
        let mut return_values: Vec<Vec<usize>> = Vec::with_capacity(self.n_splits as usize);
        let mut current: usize = 0;
        for fold_size in fold_sizes.drain(..) {
            let stop = current + fold_size;
            return_values.push(indices[current..stop].to_vec());
            current = stop
        }

        return_values
    }

    fn test_masks<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<Vec<bool>> {
        let mut return_values: Vec<Vec<bool>> = Vec::with_capacity(self.n_splits as usize);
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

    fn split<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<(Vec<usize>, Vec<usize>)> {
        let n_samples: usize = x.shape().0;
        let indices: Vec<usize> = (0..n_samples).collect();

        let mut return_values: Vec<(Vec<usize>, Vec<usize>)> =
            Vec::with_capacity(self.n_splits as usize); // TODO: init nested vecs with capacities by getting the length of test_index vecs

        for test_index in self.test_masks(x).drain(..) {
            let train_index = indices
                .clone()
                .iter()
                .enumerate()
                .filter(|&(idx, _)| test_index[idx] == false)
                .map(|(idx, _)| idx)
                .collect::<Vec<usize>>(); // filter train indices out according to mask
            let test_index = indices
                .iter()
                .enumerate()
                .filter(|&(idx, _)| test_index[idx] == true)
                .map(|(idx, _)| idx)
                .collect::<Vec<usize>>(); // filter tests indices out according to mask
            return_values.push((train_index, test_index))
        }
        return_values
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::linalg::naive::dense_matrix::*;

    #[test]
    fn run_train_test_split() {
        let n = 100;
        let x: DenseMatrix<f64> = DenseMatrix::rand(100, 3);
        let y = vec![0f64; 100];

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2);

        assert!(
            x_train.shape().0 > (n as f64 * 0.65) as usize
                && x_train.shape().0 < (n as f64 * 0.95) as usize
        );
        assert!(
            x_test.shape().0 > (n as f64 * 0.05) as usize
                && x_test.shape().0 < (n as f64 * 0.35) as usize
        );
        assert_eq!(x_train.shape().0, y_train.len());
        assert_eq!(x_test.shape().0, y_test.len());
    }

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
        let train_test_splits = k.split(&x);

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
        let train_test_splits = k.split(&x);

        assert_eq!(train_test_splits[0].1.len(), 12 as usize);
        assert_eq!(train_test_splits[0].0.len(), 11 as usize);
        assert_eq!(train_test_splits[1].0.len(), 12 as usize);
        assert_eq!(train_test_splits[1].1.len(), 11 as usize);
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
