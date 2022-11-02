//! # Metric functions
//!
//! One way to build machine learning models is to use a constructive feedback loop through model evaluation.
//! In a feedback loop you build your model first, then you get feedback from metrics, improve it and repeat until your model achieve desirable performance.
//! Evaluation metrics helps to explain the performance of a model and compare models based on an objective criterion.
//!
//! Choosing the right metric is crucial while evaluating machine learning models. In SmartCore you will find metrics for these classes of ML models:
//!
//! * [Classification metrics](struct.ClassificationMetrics.html)
//! * [Regression metrics](struct.RegressionMetrics.html)
//! * [Clustering metrics](struct.ClusterMetrics.html)
//!
//! Example:
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::linear::logistic_regression::LogisticRegression;
//! use smartcore::metrics::*;
//!
//! let x = DenseMatrix::from_2d_array(&[
//!             &[5.1, 3.5, 1.4, 0.2],
//!             &[4.9, 3.0, 1.4, 0.2],
//!             &[4.7, 3.2, 1.3, 0.2],
//!             &[4.6, 3.1, 1.5, 0.2],
//!             &[5.0, 3.6, 1.4, 0.2],
//!             &[5.4, 3.9, 1.7, 0.4],
//!             &[4.6, 3.4, 1.4, 0.3],
//!             &[5.0, 3.4, 1.5, 0.2],
//!             &[4.4, 2.9, 1.4, 0.2],
//!             &[4.9, 3.1, 1.5, 0.1],
//!             &[7.0, 3.2, 4.7, 1.4],
//!             &[6.4, 3.2, 4.5, 1.5],
//!             &[6.9, 3.1, 4.9, 1.5],
//!             &[5.5, 2.3, 4.0, 1.3],
//!             &[6.5, 2.8, 4.6, 1.5],
//!             &[5.7, 2.8, 4.5, 1.3],
//!             &[6.3, 3.3, 4.7, 1.6],
//!             &[4.9, 2.4, 3.3, 1.0],
//!             &[6.6, 2.9, 4.6, 1.3],
//!             &[5.2, 2.7, 3.9, 1.4],
//!   ]);
//! let y: Vec<i8> = vec![
//!             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//!   ];
//!
//! let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
//!
//! let y_hat = lr.predict(&x).unwrap();
//!
//! let acc = ClassificationMetricsOrd::accuracy().get_score(&y, &y_hat);
//! // or
//! let acc = accuracy(&y, &y_hat);
//! ```

/// Accuracy score.
pub mod accuracy;
// TODO: reimplement AUC
// /// Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
pub mod auc;
/// Compute the homogeneity, completeness and V-Measure scores.
pub mod cluster_hcv;
pub(crate) mod cluster_helpers;
/// Multitude of distance metrics are defined here
pub mod distance;
/// F1 score, also known as balanced F-score or F-measure.
pub mod f1;
/// Mean absolute error regression loss.
pub mod mean_absolute_error;
/// Mean squared error regression loss.
pub mod mean_squared_error;
/// Computes the precision.
pub mod precision;
/// Coefficient of determination (R2).
pub mod r2;
/// Computes the recall.
pub mod recall;

use crate::linalg::basic::arrays::{Array1, ArrayView1};
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

use std::marker::PhantomData;

/// A trait to be implemented by all metrics
pub trait Metrics<T> {
    /// instantiate a new Metrics trait-object
    /// <https://doc.rust-lang.org/error-index.html#E0038>
    fn new() -> Self
    where
        Self: Sized;
    /// used to instantiate metric with a paramenter
    fn new_with(_parameter: f64) -> Self
    where
        Self: Sized;
    /// compute score realated to this metric
    fn get_score(&self, y_true: &dyn ArrayView1<T>, y_pred: &dyn ArrayView1<T>) -> f64;
}

/// Use these metrics to compare classification models.
pub struct ClassificationMetrics<T> {
    phantom: PhantomData<T>,
}

/// Use these metrics to compare classification models for
/// numbers that require `Ord`.
pub struct ClassificationMetricsOrd<T> {
    phantom: PhantomData<T>,
}

/// Metrics for regression models.
pub struct RegressionMetrics<T> {
    phantom: PhantomData<T>,
}

/// Cluster metrics.
pub struct ClusterMetrics<T> {
    phantom: PhantomData<T>,
}

impl<T: Number + RealNumber + FloatNumber> ClassificationMetrics<T> {
    /// Recall, see [recall](recall/index.html).
    pub fn recall() -> recall::Recall<T> {
        recall::Recall::new()
    }

    /// Precision, see [precision](precision/index.html).
    pub fn precision() -> precision::Precision<T> {
        precision::Precision::new()
    }

    /// F1 score, also known as balanced F-score or F-measure, see [F1](f1/index.html).
    pub fn f1(beta: f64) -> f1::F1<T> {
        f1::F1::new_with(beta)
    }

    /// Area Under the Receiver Operating Characteristic Curve (ROC AUC), see [AUC](auc/index.html).
    pub fn roc_auc_score() -> auc::AUC<T> {
        auc::AUC::<T>::new()
    }
}

impl<T: Number + Ord> ClassificationMetricsOrd<T> {
    /// Accuracy score, see [accuracy](accuracy/index.html).
    pub fn accuracy() -> accuracy::Accuracy<T> {
        accuracy::Accuracy::new()
    }
}

impl<T: Number + FloatNumber> RegressionMetrics<T> {
    /// Mean squared error, see [mean squared error](mean_squared_error/index.html).
    pub fn mean_squared_error() -> mean_squared_error::MeanSquareError<T> {
        mean_squared_error::MeanSquareError::new()
    }

    /// Mean absolute error, see [mean absolute error](mean_absolute_error/index.html).
    pub fn mean_absolute_error() -> mean_absolute_error::MeanAbsoluteError<T> {
        mean_absolute_error::MeanAbsoluteError::new()
    }

    /// Coefficient of determination (R2), see [R2](r2/index.html).
    pub fn r2() -> r2::R2<T> {
        r2::R2::<T>::new()
    }
}

impl<T: Number + Ord> ClusterMetrics<T> {
    /// Homogeneity and completeness and V-Measure scores at once.
    pub fn hcv_score() -> cluster_hcv::HCVScore<T> {
        cluster_hcv::HCVScore::<T>::new()
    }
}

/// Function that calculated accuracy score, see [accuracy](accuracy/index.html).
/// * `y_true` - cround truth (correct) labels
/// * `y_pred` - predicted labels, as returned by a classifier.
pub fn accuracy<T: Number + Ord, V: ArrayView1<T>>(y_true: &V, y_pred: &V) -> f64 {
    let obj = ClassificationMetricsOrd::<T>::accuracy();
    obj.get_score(y_true, y_pred)
}

/// Calculated recall score, see [recall](recall/index.html)
/// * `y_true` - cround truth (correct) labels.
/// * `y_pred` - predicted labels, as returned by a classifier.
pub fn recall<T: Number + RealNumber + FloatNumber, V: ArrayView1<T>>(
    y_true: &V,
    y_pred: &V,
) -> f64 {
    let obj = ClassificationMetrics::<T>::recall();
    obj.get_score(y_true, y_pred)
}

/// Calculated precision score, see [precision](precision/index.html).
/// * `y_true` - cround truth (correct) labels.
/// * `y_pred` - predicted labels, as returned by a classifier.
pub fn precision<T: Number + RealNumber + FloatNumber, V: ArrayView1<T>>(
    y_true: &V,
    y_pred: &V,
) -> f64 {
    let obj = ClassificationMetrics::<T>::precision();
    obj.get_score(y_true, y_pred)
}

/// Computes F1 score, see [F1](f1/index.html).
/// * `y_true` - cround truth (correct) labels.
/// * `y_pred` - predicted labels, as returned by a classifier.
pub fn f1<T: Number + RealNumber + FloatNumber, V: ArrayView1<T>>(
    y_true: &V,
    y_pred: &V,
    beta: f64,
) -> f64 {
    let obj = ClassificationMetrics::<T>::f1(beta);
    obj.get_score(y_true, y_pred)
}

/// AUC score, see [AUC](auc/index.html).
/// * `y_true` - cround truth (correct) labels.
/// * `y_pred_probabilities` - probability estimates, as returned by a classifier.
pub fn roc_auc_score<T: Number + RealNumber + FloatNumber + PartialOrd, V: ArrayView1<T> + Array1<T> + Array1<T>>(
    y_true: &V,
    y_pred_probabilities: &V,
) -> f64 {
    let obj = ClassificationMetrics::<T>::roc_auc_score();
    obj.get_score(y_true, y_pred_probabilities)
}

/// Computes mean squared error, see [mean squared error](mean_squared_error/index.html).
/// * `y_true` - Ground truth (correct) target values.
/// * `y_pred` - Estimated target values.
pub fn mean_squared_error<T: Number + FloatNumber, V: ArrayView1<T>>(
    y_true: &V,
    y_pred: &V,
) -> f64 {
    RegressionMetrics::<T>::mean_squared_error().get_score(y_true, y_pred)
}

/// Computes mean absolute error, see [mean absolute error](mean_absolute_error/index.html).
/// * `y_true` - Ground truth (correct) target values.
/// * `y_pred` - Estimated target values.
pub fn mean_absolute_error<T: Number + FloatNumber, V: ArrayView1<T>>(
    y_true: &V,
    y_pred: &V,
) -> f64 {
    RegressionMetrics::<T>::mean_absolute_error().get_score(y_true, y_pred)
}

/// Computes R2 score, see [R2](r2/index.html).
/// * `y_true` - Ground truth (correct) target values.
/// * `y_pred` - Estimated target values.
pub fn r2<T: Number + FloatNumber, V: ArrayView1<T>>(y_true: &V, y_pred: &V) -> f64 {
    RegressionMetrics::<T>::r2().get_score(y_true, y_pred)
}

/// Homogeneity metric of a cluster labeling given a ground truth (range is between 0.0 and 1.0).
/// A cluster result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.
/// * `labels_true` - ground truth class labels to be used as a reference.
/// * `labels_pred` - cluster labels to evaluate.
pub fn homogeneity_score<
    T: Number + FloatNumber + RealNumber + Ord,
    V: ArrayView1<T> + Array1<T>,
>(
    y_true: &V,
    y_pred: &V,
) -> f64 {
    let mut obj = ClusterMetrics::<T>::hcv_score();
    obj.compute(y_true, y_pred);
    obj.homogeneity().unwrap()
}

///
/// Completeness metric of a cluster labeling given a ground truth (range is between 0.0 and 1.0).
/// * `labels_true` - ground truth class labels to be used as a reference.
/// * `labels_pred` - cluster labels to evaluate.
pub fn completeness_score<
    T: Number + FloatNumber + RealNumber + Ord,
    V: ArrayView1<T> + Array1<T>,
>(
    y_true: &V,
    y_pred: &V,
) -> f64 {
    let mut obj = ClusterMetrics::<T>::hcv_score();
    obj.compute(y_true, y_pred);
    obj.completeness().unwrap()
}

/// The harmonic mean between homogeneity and completeness.
/// * `labels_true` - ground truth class labels to be used as a reference.
/// * `labels_pred` - cluster labels to evaluate.
pub fn v_measure_score<T: Number + FloatNumber + RealNumber + Ord, V: ArrayView1<T> + Array1<T>>(
    y_true: &V,
    y_pred: &V,
) -> f64 {
    let mut obj = ClusterMetrics::<T>::hcv_score();
    obj.compute(y_true, y_pred);
    obj.v_measure().unwrap()
}
