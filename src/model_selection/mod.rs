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
/// Entities involved in the KFold procedure:
///     * a dataset
///     * a number k of groups (k-folds)
/// 
/// Procedure in `cross_validate()`: 
///   1. Shuffle the dataset randomly.
///   2. Split the dataset into k groups
///   3. For each unique group:
///         1. Take the group as a hold out or test data set
///         2. Take the remaining groups as a training data set
///         3. Fit a model on the training set and evaluate it on the test set
///         4. Retain the evaluation score and discard the model
///   4. Summarize the skill of the model using the sample of model evaluation scores
trait BaseKFold {
    /// Returns integer indices corresponding to test sets
    fn test_indices<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<Vec<usize>>;

    // /// Return matrix corresponding to test sets
    // fn test_matrices<T: RealNumber, M: Matrix<T>>(&self, X: &M) -> Vec<&M>;

    // /// Return a tuple containing the the training set indices for that split and
    // /// the testing set indices for that split.
    // fn split<T: RealNumber, M: Matrix<T>>(&self, X: &M) -> Vec<(Vec<usize>, Vec<usize>)>;
}

/// An implementation of KFold
pub struct KFold {
    n_splits: i32,
    // TODO: to be implemented later
    // shuffle: bool,
    // random_state: i32, 
}

impl BaseKFold for KFold {
    fn test_indices<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<Vec<usize>> {
        // n_samples = _num_samples(X)  # number of sample in an array-like
        let n_samples: usize = x.shape().1;
        println!("n {:?}", &n_samples);

        // indices = np.arange(n_samples)  # an iterator of size len(x)
        let indices: Vec<usize> = (0..n_samples).collect();
        println!("starting indices{:?}", &indices);

        //  Return a new array of given shape n_split, filled with each element of n_samples divided by n_splits.
        // fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        let mut fold_sizes = vec![n_samples / self.n_splits as usize; self.n_splits as usize];
        println!("fold size {:?}", &fold_sizes);

        // fold_sizes[:n_samples % n_splits] += 1
        for i in 0..(n_samples % self.n_splits as usize) {
            fold_sizes[i] = fold_sizes[i] + 1;
        }
        println!("fold size after correction{:?}", &fold_sizes);

        let mut return_values: Vec<Vec<usize>> = Vec::new();
        let mut current: usize = 0;
        for fold_size in fold_sizes.drain(..) {
            let stop = current + fold_size;
            println!("current, stop {:?}, {:?}", &current, &stop);
            return_values.push(
                indices[current..stop].to_vec()
            );
            println!("loop {:?}", &return_values);
            current = stop
        }
        
        return return_values;

    }

    // fn test_matrices<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Vec<&M> {
    //     // Python implementation
    //     for test_index in self.test_indices(x) {
    //         // test_mask = np.zeros(_num_samples(X), dtype=bool)
    //         let test_mask = M::zeros(x.shape().0, x.shape().1);
    //         test_mask[test_index] = True
    //         yield test_mask
    //     }
    //     let tmp = vec![&M::zeros(2, 2)];
    //     return tmp;
    // }

    // fn split<T: RealNumber, M: Matrix<T>>(&self, X: &M) -> Vec<(Vec<usize>, Vec<usize>)> {
    //     // Python implementation
    //     // X, y, groups = indexable(X, y, groups)
    //     // indices = np.arange(_num_samples(X))  // an iterator of len(x)
    //     // for test_index in self._iter_test_masks(X, y, groups):
    //     //     train_index = indices[np.logical_not(test_index)]
    //     //     test_index = indices[test_index]
    //     //     yield train_index, test_index
    //     let tmp = vec![(vec![0, 1], vec![0, 1])];
    //     return tmp;
    // }
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
        let k = KFold { n_splits: 3};

        let x: DenseMatrix<f64> = DenseMatrix::rand(33, 33);

        let test_indices = k.test_indices(&x);

        println!("{:?}", &test_indices);
        assert_eq!(test_indices[0], (0..11).collect::<Vec<usize>>());
        assert_eq!(test_indices[1], (11..22).collect::<Vec<usize>>());
        assert_eq!(test_indices[2], (22..33).collect::<Vec<usize>>());
    }

    #[test]
    fn run_kfold_return_test_indices_odd() {
        let k = KFold { n_splits: 3};

        let x: DenseMatrix<f64> = DenseMatrix::rand(34, 34);

        let test_indices = k.test_indices(&x);

        println!("{:?}", &test_indices);
        assert_eq!(test_indices[0], (0..12).collect::<Vec<usize>>());
        assert_eq!(test_indices[1], (12..23).collect::<Vec<usize>>());
        assert_eq!(test_indices[2], (23..34).collect::<Vec<usize>>());
    }
}
