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
    /// Return a tuple containing the the training set indices for that split and
    /// the testing set indices for that split.
    fn split(&self, X: Matrix) -> Iterator< Item = Tuple(ndarray::ArrayBase, ndarray::ArrayBase)>;

    /// Returns integer indices corresponding to test sets
    fn test_indices(&self, X: Matrix) -> Iterator< Item = i32>;

    /// Return matrix corresponding to test sets
    fn test_matrices(&self, X: Matrix) -> Iterator< Item = Matrix>;

}

pub struct KFold {
    n_splits: i32,
    shuffle: bool,
    random_state: i32, 
}

impl BaseKFold for KFold {
    ...
}
