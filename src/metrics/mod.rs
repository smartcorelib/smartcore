pub mod accuracy;
pub mod auc;
pub mod f1;
pub mod mean_absolute_error;
pub mod mean_squared_error;
pub mod precision;
pub mod r2;
pub mod recall;

use crate::linalg::BaseVector;
use crate::math::num::FloatExt;

pub struct ClassificationMetrics {}

pub struct RegressionMetrics {}

impl ClassificationMetrics {
    pub fn accuracy() -> accuracy::Accuracy {
        accuracy::Accuracy {}
    }

    pub fn recall() -> recall::Recall {
        recall::Recall {}
    }

    pub fn precision() -> precision::Precision {
        precision::Precision {}
    }

    pub fn f1() -> f1::F1 {
        f1::F1 {}
    }

    pub fn roc_auc_score() -> auc::AUC {
        auc::AUC {}
    }
}

impl RegressionMetrics {
    pub fn mean_squared_error() -> mean_squared_error::MeanSquareError {
        mean_squared_error::MeanSquareError {}
    }

    pub fn mean_absolute_error() -> mean_absolute_error::MeanAbsoluteError {
        mean_absolute_error::MeanAbsoluteError {}
    }

    pub fn r2() -> r2::R2 {
        r2::R2 {}
    }
}

pub fn accuracy<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    ClassificationMetrics::accuracy().get_score(y_true, y_pred)
}

pub fn recall<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    ClassificationMetrics::recall().get_score(y_true, y_pred)
}

pub fn precision<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    ClassificationMetrics::precision().get_score(y_true, y_pred)
}

pub fn f1<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    ClassificationMetrics::f1().get_score(y_true, y_pred)
}

pub fn roc_auc_score<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_pred_probabilities: &V) -> T {
    ClassificationMetrics::roc_auc_score().get_score(y_true, y_pred_probabilities)
}

pub fn mean_squared_error<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    RegressionMetrics::mean_squared_error().get_score(y_true, y_pred)
}

pub fn mean_absolute_error<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    RegressionMetrics::mean_absolute_error().get_score(y_true, y_pred)
}

pub fn r2<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_pred: &V) -> T {
    RegressionMetrics::r2().get_score(y_true, y_pred)
}
