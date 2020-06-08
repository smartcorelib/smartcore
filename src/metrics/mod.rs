pub mod accuracy;
pub mod auc;
pub mod f1;
pub mod precision;
pub mod recall;

use crate::linalg::BaseVector;
use crate::math::num::FloatExt;

pub struct ClassificationMetrics {}

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
}

pub fn accuracy<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_prod: &V) -> T {
    ClassificationMetrics::accuracy().get_score(y_true, y_prod)
}

pub fn recall<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_prod: &V) -> T {
    ClassificationMetrics::recall().get_score(y_true, y_prod)
}

pub fn precision<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_prod: &V) -> T {
    ClassificationMetrics::precision().get_score(y_true, y_prod)
}
