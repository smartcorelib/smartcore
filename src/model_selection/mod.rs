//! # Model Selection methods
//!
//! In statistics and machine learning we usually split our data into two sets: one for training and the other one for testing.
//! We fit our model to the training data, in order to make predictions on the test data. We do that to avoid overfitting or underfitting model to our data.
//! Overfitting is bad because the model we trained fits trained data too well and can’t make any inferences on new data.
//! Underfitted is bad because the model is undetrained and does not fit the training data well.
//! Splitting data into multiple subsets helps us to find the right combination of hyperparameters, estimate model performance and choose the right model for
//! the data.
//!
//! In SmartCore a random split into training and test sets can be quickly computed with the [train_test_split](./fn.train_test_split.html) helper function.
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::model_selection::train_test_split;
//! use smartcore::linalg::basic::arrays::Array;
//!
//! //Iris data
//! let x = DenseMatrix::from_2d_array(&[
//!           &[5.1, 3.5, 1.4, 0.2],
//!           &[4.9, 3.0, 1.4, 0.2],
//!           &[4.7, 3.2, 1.3, 0.2],
//!           &[4.6, 3.1, 1.5, 0.2],
//!           &[5.0, 3.6, 1.4, 0.2],
//!           &[5.4, 3.9, 1.7, 0.4],
//!           &[4.6, 3.4, 1.4, 0.3],
//!           &[5.0, 3.4, 1.5, 0.2],
//!           &[4.4, 2.9, 1.4, 0.2],
//!           &[4.9, 3.1, 1.5, 0.1],
//!           &[7.0, 3.2, 4.7, 1.4],
//!           &[6.4, 3.2, 4.5, 1.5],
//!           &[6.9, 3.1, 4.9, 1.5],
//!           &[5.5, 2.3, 4.0, 1.3],
//!           &[6.5, 2.8, 4.6, 1.5],
//!           &[5.7, 2.8, 4.5, 1.3],
//!           &[6.3, 3.3, 4.7, 1.6],
//!           &[4.9, 2.4, 3.3, 1.0],
//!           &[6.6, 2.9, 4.6, 1.3],
//!           &[5.2, 2.7, 3.9, 1.4],
//!           ]);
//! let y: Vec<f64> = vec![
//!           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
//! ];
//!
//! let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, None);
//!
//! println!("X train: {:?}, y train: {}, X test: {:?}, y test: {}",
//!             x_train.shape(), y_train.len(), x_test.shape(), y_test.len());
//! ```
//!
//! When we partition the available data into two disjoint sets, we drastically reduce the number of samples that can be used for training.
//!
//! One way to solve this problem is to use k-fold cross-validation. With k-fold validation, the dataset is split into k disjoint sets.
//! A model is trained using k - 1 of the folds, and the resulting model is validated on the remaining portion of the data.
//!
//! The simplest way to run cross-validation is to use the [cross_val_score](./fn.cross_validate.html) helper function on your estimator and the dataset.
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::model_selection::{KFold, cross_validate};
//! use smartcore::metrics::accuracy;
//! use smartcore::linear::logistic_regression::LogisticRegression;
//! use smartcore::api::SupervisedEstimator;
//! use smartcore::linalg::basic::arrays::Array;
//!
//! //Iris data
//! let x = DenseMatrix::from_2d_array(&[
//!           &[5.1, 3.5, 1.4, 0.2],
//!           &[4.9, 3.0, 1.4, 0.2],
//!           &[4.7, 3.2, 1.3, 0.2],
//!           &[4.6, 3.1, 1.5, 0.2],
//!           &[5.0, 3.6, 1.4, 0.2],
//!           &[5.4, 3.9, 1.7, 0.4],
//!           &[4.6, 3.4, 1.4, 0.3],
//!           &[5.0, 3.4, 1.5, 0.2],
//!           &[4.4, 2.9, 1.4, 0.2],
//!           &[4.9, 3.1, 1.5, 0.1],
//!           &[7.0, 3.2, 4.7, 1.4],
//!           &[6.4, 3.2, 4.5, 1.5],
//!           &[6.9, 3.1, 4.9, 1.5],
//!           &[5.5, 2.3, 4.0, 1.3],
//!           &[6.5, 2.8, 4.6, 1.5],
//!           &[5.7, 2.8, 4.5, 1.3],
//!           &[6.3, 3.3, 4.7, 1.6],
//!           &[4.9, 2.4, 3.3, 1.0],
//!           &[6.6, 2.9, 4.6, 1.3],
//!           &[5.2, 2.7, 3.9, 1.4],
//!           ]);
//! let y: Vec<i32> = vec![
//!           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//! ];
//!
//! let cv = KFold::default().with_n_splits(3);
//!
//! let results = cross_validate(
//!     LogisticRegression::new(),   //estimator
//!     &x, &y,                 //data
//!     Default::default(),     //hyperparameters
//!     &cv,                     //cross validation split
//!     &accuracy).unwrap();    //metric
//!
//! println!("Training accuracy: {}, test accuracy: {}",
//!     results.mean_test_score(), results.mean_train_score());
//! ```
//!
//! The function [cross_val_predict](./fn.cross_val_predict.html) has a similar interface to `cross_val_score`,
//! but instead of test error it calculates predictions for all samples in the test set.

use rand::seq::SliceRandom;
use std::fmt::{Debug, Display};

#[allow(unused_imports)]
use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;
use crate::rand_custom::get_rng_impl;

// TODO: fix this module
// pub(crate) mod hyper_tuning;
pub(crate) mod kfold;

// pub use hyper_tuning::{GridSearchCV, GridSearchCVParameters};
pub use kfold::{KFold, KFoldIter};

/// An interface for the K-Folds cross-validator
pub trait BaseKFold {
    /// An iterator over indices that split data into training and test set.
    type Output: Iterator<Item = (Vec<usize>, Vec<usize>)>;
    /// Return a tuple containing the the training set indices for that split and
    /// the testing set indices for that split.
    fn split<T: Number, X: Array2<T>>(&self, x: &X) -> Self::Output;
    /// Returns the number of splits
    fn n_splits(&self) -> usize;
}

/// Splits data into 2 disjoint datasets.
/// * `x` - features, matrix of size _NxM_ where _N_ is number of samples and _M_ is number of attributes.
/// * `y` - target values, should be of size _N_
/// * `test_size`, (0, 1] - the proportion of the dataset to include in the test split.
/// * `shuffle`, - whether or not to shuffle the data before splitting
pub fn train_test_split<
    TX: Debug + Display + Copy + Sized,
    TY: Debug + Display + Copy + Sized,
    X: Array2<TX>,
    Y: Array1<TY>,
>(
    x: &X,
    y: &Y,
    test_size: f32,
    shuffle: bool,
    seed: Option<u64>,
) -> (X, X, Y, Y) {
    if x.shape().0 != y.shape() {
        panic!(
            "x and y should have the same number of samples. |x|: {}, |y|: {}",
            x.shape().0,
            y.shape()
        );
    }
    let mut rng = get_rng_impl(seed);

    if test_size <= 0. || test_size > 1.0 {
        panic!("test_size should be between 0 and 1");
    }

    let n = y.shape();

    let n_test = ((n as f32) * test_size) as usize;

    if n_test < 1 {
        panic!("number of sample is too small {}", n);
    }

    let mut indices: Vec<usize> = (0..n).collect();

    if shuffle {
        indices.shuffle(&mut rng);
    }

    let x_train = x.take(&indices[n_test..n], 0);
    let x_test = x.take(&indices[0..n_test], 0);
    let y_train = y.take(&indices[n_test..n]);
    let y_test = y.take(&indices[0..n_test]);

    (x_train, x_test, y_train, y_test)
}

/// Cross validation results.
#[derive(Clone, Debug)]
pub struct CrossValidationResult {
    /// Vector with test scores on each cv split
    pub test_score: Vec<f64>,
    /// Vector with training scores on each cv split
    pub train_score: Vec<f64>,
}

impl CrossValidationResult {
    /// Average test score
    pub fn mean_test_score(&self) -> f64 {
        let mut sum = 0f64;
        for s in self.test_score.iter() {
            sum += *s;
        }
        sum / self.test_score.len() as f64
    }
    /// Average training score
    pub fn mean_train_score(&self) -> f64 {
        let mut sum = 0f64;
        for s in self.train_score.iter() {
            sum += *s;
        }
        sum / self.train_score.len() as f64
    }
}

/// Evaluate an estimator by cross-validation using given metric.
/// * `fit_estimator` - a `fit` function of an estimator
/// * `x` - features, matrix of size _NxM_ where _N_ is number of samples and _M_ is number of attributes.
/// * `y` - target values, should be of size _N_
/// * `parameters` - parameters of selected estimator. Use `Default::default()` for default parameters.
/// * `cv` - the cross-validation splitting strategy, should be an instance of [`BaseKFold`](./trait.BaseKFold.html)
/// * `score` - a metric to use for evaluation, see [metrics](../metrics/index.html)
pub fn cross_validate<TX, TY, X, Y, H, E, K, S>(
    _estimator: E, // just an empty placeholder to allow passing `fit()`
    x: &X,
    y: &Y,
    parameters: H,
    cv: &K,
    score: &S,
) -> Result<CrossValidationResult, Failed>
where
    TX: Number + RealNumber,
    TY: Number,
    X: Array2<TX>,
    Y: Array1<TY>,
    H: Clone,
    K: BaseKFold,
    E: SupervisedEstimator<X, Y, H>,
    S: Fn(&Y, &Y) -> f64,
{
    let k = cv.n_splits();
    let mut test_score: Vec<f64> = Vec::with_capacity(k);
    let mut train_score: Vec<f64> = Vec::with_capacity(k);

    for (train_idx, test_idx) in cv.split(x) {
        let train_x = x.take(&train_idx, 0);
        let train_y = y.take(&train_idx);
        let test_x = x.take(&test_idx, 0);
        let test_y = y.take(&test_idx);

        // NOTE: we use here only the estimator "class", the  actual struct get dropped
        let computed =
            <E as SupervisedEstimator<X, Y, H>>::fit(&train_x, &train_y, parameters.clone())?;

        train_score.push(score(&train_y, &computed.predict(&train_x)?));
        test_score.push(score(&test_y, &computed.predict(&test_x)?));
    }

    Ok(CrossValidationResult {
        test_score,
        train_score,
    })
}

/// Generate cross-validated estimates for each input data point.
/// The data is split according to the cv parameter. Each sample belongs to exactly one test set, and its prediction is computed with an estimator fitted on the corresponding training set.
/// * `fit_estimator` - a `fit` function of an estimator
/// * `x` - features, matrix of size _NxM_ where _N_ is number of samples and _M_ is number of attributes.
/// * `y` - target values, should be of size _N_
/// * `parameters` - parameters of selected estimator. Use `Default::default()` for default parameters.
/// * `cv` - the cross-validation splitting strategy, should be an instance of [`BaseKFold`](./trait.BaseKFold.html)
pub fn cross_val_predict<TX, TY, X, Y, H, E, K>(
    _estimator: E, // just an empty placeholder to allow passing `fit()`
    x: &X,
    y: &Y,
    parameters: H,
    cv: &K,
) -> Result<Y, Failed>
where
    TX: Number,
    TY: Number,
    X: Array2<TX>,
    Y: Array1<TY>,
    H: Clone,
    K: BaseKFold,
    E: SupervisedEstimator<X, Y, H>,
{
    let mut y_hat = Y::zeros(y.shape());

    for (train_idx, test_idx) in cv.split(x) {
        let train_x = x.take(&train_idx, 0);
        let train_y = y.take(&train_idx);
        let test_x = x.take(&test_idx, 0);

        let computed =
            <E as SupervisedEstimator<X, Y, H>>::fit(&train_x, &train_y, parameters.clone())?;

        let y_test_hat = computed.predict(&test_x)?;
        for (i, &idx) in test_idx.iter().enumerate() {
            y_hat.set(idx, *y_test_hat.get(i));
        }
    }

    Ok(y_hat)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::algorithm::neighbour::KNNAlgorithmName;
    use crate::api::NoParameters;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::linear::logistic_regression::LogisticRegression;
    use crate::metrics::distance::Distances;
    use crate::metrics::{accuracy, mean_absolute_error};
    use crate::model_selection::cross_validate;
    use crate::model_selection::kfold::KFold;
    use crate::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};
    use crate::neighbors::KNNWeightFunction;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn run_train_test_split() {
        let n = 123;
        let x: DenseMatrix<f64> = DenseMatrix::rand(n, 3);
        let y = vec![0f64; n];

        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true, None);

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

    #[derive(Clone)]
    struct BiasedParameters {}
    impl NoParameters for BiasedParameters {}

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_cross_validate_biased() {
        struct BiasedEstimator {}

        impl<X: Array2<f32>, Y: Array1<u32>, P: NoParameters> SupervisedEstimator<X, Y, P>
            for BiasedEstimator
        {
            fn new() -> Self {
                Self {}
            }
            fn fit(_: &X, _: &Y, _: P) -> Result<BiasedEstimator, Failed> {
                Ok(BiasedEstimator {})
            }
        }

        impl<X: Array2<f32>, Y: Array1<u32>> Predictor<X, Y> for BiasedEstimator {
            fn predict(&self, x: &X) -> Result<Y, Failed> {
                let (n, _) = x.shape();
                Ok(Y::zeros(n))
            }
        }

        let x: DenseMatrix<f32> = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ]);
        let y: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let cv = KFold {
            n_splits: 5,
            ..KFold::default()
        };

        let results = cross_validate(
            BiasedEstimator {},
            &x,
            &y,
            BiasedParameters {},
            &cv,
            &accuracy,
        )
        .unwrap();

        assert_eq!(0.4, results.mean_test_score());
        assert_eq!(0.4, results.mean_train_score());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_cross_validate_knn() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159., 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165., 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            &[365.385, 187., 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335., 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);
        let y = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let cv = KFold {
            n_splits: 5,
            ..KFold::default()
        };

        let results = cross_validate(
            KNNRegressor::new(),
            &x,
            &y,
            Default::default(),
            &cv,
            &mean_absolute_error,
        )
        .unwrap();

        assert!(results.mean_test_score() < 15.0);
        assert!(results.mean_train_score() < results.mean_test_score());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_cross_val_predict_knn() {
        let x: DenseMatrix<f64> = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159., 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165., 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            &[365.385, 187., 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335., 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);
        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let cv: KFold = KFold {
            n_splits: 2,
            ..KFold::default()
        };

        let y_hat: Vec<f64> = cross_val_predict(
            KNNRegressor::new(),
            &x,
            &y,
            KNNRegressorParameters::default()
                .with_k(3)
                .with_distance(Distances::euclidian())
                .with_algorithm(KNNAlgorithmName::LinearSearch)
                .with_weight(KNNWeightFunction::Distance),
            &cv,
        )
        .unwrap();

        assert!(mean_absolute_error(&y, &y_hat) < 10.0);
    }

    #[test]
    fn test_cross_validation_accuracy() {
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ]);
        let y: Vec<i32> = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let cv = KFold::default().with_n_splits(3);

        let results = cross_validate(
            LogisticRegression::new(),
            &x,
            &y,
            Default::default(),
            &cv,
            &accuracy,
        )
        .unwrap();
        println!("{:?}", results);
    }
}
