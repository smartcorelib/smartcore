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
//!
//! Example:
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::linear::logistic_regression::LogisticRegression;
//! use smartcore::metrics::*;
//!
//! let x = DenseMatrix::from_array(&[
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
//! let y: Vec<f64> = vec![
//!             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
//!   ];
//!
//! let lr = LogisticRegression::fit(&x, &y);
//!
//! let y_hat = lr.predict(&x);
//!
//! let acc = ClassificationMetrics::accuracy().get_score(&y, &y_hat);
//! // or
//! let acc = accuracy(&y, &y_hat);
//! ```

/// Accuracy score.
pub mod accuracy;
/// Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
pub mod auc;
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

use crate::linalg::BaseVector;
use crate::math::num::RealNumber;

/// Use these metrics to compare classification models.
pub struct ClassificationMetrics {}

/// Metrics for regression models.
pub struct RegressionMetrics {}

impl ClassificationMetrics {
    /// Accuracy score, see [accuracy](accuracy/index.html).
    pub fn accuracy() -> accuracy::Accuracy {
        accuracy::Accuracy {}
    }

    /// Recall, see [recall](recall/index.html).
    pub fn recall() -> recall::Recall {
        recall::Recall {}
    }

    /// Precision, see [precision](precision/index.html).
    pub fn precision() -> precision::Precision {
        precision::Precision {}
    }

    /// F1 score, also known as balanced F-score or F-measure, see [F1](f1/index.html).
    pub fn f1<T: RealNumber>(beta: T) -> f1::F1<T> {
        f1::F1 { beta: beta }
    }

    /// Area Under the Receiver Operating Characteristic Curve (ROC AUC), see [AUC](auc/index.html).
    pub fn roc_auc_score() -> auc::AUC {
        auc::AUC {}
    }
}

impl RegressionMetrics {
    /// Mean squared error, see [mean squared error](mean_squared_error/index.html).
    pub fn mean_squared_error() -> mean_squared_error::MeanSquareError {
        mean_squared_error::MeanSquareError {}
    }

    /// Mean absolute error, see [mean absolute error](mean_absolute_error/index.html).
    pub fn mean_absolute_error() -> mean_absolute_error::MeanAbsoluteError {
        mean_absolute_error::MeanAbsoluteError {}
    }

    /// Coefficient of determination (R2), see [R2](r2/index.html).
    pub fn r2() -> r2::R2 {
        r2::R2 {}
    }
}

/// Function that calculated accuracy score, see [accuracy](accuracy/index.html).
/// * `y_true` - cround truth (correct) labels
/// * `y_pred` - predicted labels, as returned by a classifier.
pub fn accuracy<T: RealNumber, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    ClassificationMetrics::accuracy().get_score(y_true, y_pred)
}

/// Calculated recall score, see [recall](recall/index.html)
/// * `y_true` - cround truth (correct) labels.
/// * `y_pred` - predicted labels, as returned by a classifier.
pub fn recall<T: RealNumber, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    ClassificationMetrics::recall().get_score(y_true, y_pred)
}

/// Calculated precision score, see [precision](precision/index.html).
/// * `y_true` - cround truth (correct) labels.
/// * `y_pred` - predicted labels, as returned by a classifier.
pub fn precision<T: RealNumber, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    ClassificationMetrics::precision().get_score(y_true, y_pred)
}

/// Computes F1 score, see [F1](f1/index.html).
/// * `y_true` - cround truth (correct) labels.
/// * `y_pred` - predicted labels, as returned by a classifier.
pub fn f1<T: RealNumber, V: BaseVector<T>>(y_true: &V, y_pred: &V, beta: T) -> T {
    ClassificationMetrics::f1(beta).get_score(y_true, y_pred)
}

/// AUC score, see [AUC](auc/index.html).
/// * `y_true` - cround truth (correct) labels.
/// * `y_pred_probabilities` - probability estimates, as returned by a classifier.
pub fn roc_auc_score<T: RealNumber, V: BaseVector<T>>(y_true: &V, y_pred_probabilities: &V) -> T {
    ClassificationMetrics::roc_auc_score().get_score(y_true, y_pred_probabilities)
}

/// Computes mean squared error, see [mean squared error](mean_squared_error/index.html).
/// * `y_true` - Ground truth (correct) target values.
/// * `y_pred` - Estimated target values.
pub fn mean_squared_error<T: RealNumber, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    RegressionMetrics::mean_squared_error().get_score(y_true, y_pred)
}

/// Computes mean absolute error, see [mean absolute error](mean_absolute_error/index.html).
/// * `y_true` - Ground truth (correct) target values.
/// * `y_pred` - Estimated target values.
pub fn mean_absolute_error<T: RealNumber, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    RegressionMetrics::mean_absolute_error().get_score(y_true, y_pred)
}

/// Computes R2 score, see [R2](r2/index.html).
/// * `y_true` - Ground truth (correct) target values.
/// * `y_pred` - Estimated target values.
pub fn r2<T: RealNumber, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    RegressionMetrics::r2().get_score(y_true, y_pred)
}
