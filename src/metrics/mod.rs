pub mod accuracy;
pub mod recall;
pub mod precision;

use crate::math::num::FloatExt;
use crate::linalg::BaseVector;

pub struct ClassificationMetrics{}

impl ClassificationMetrics {
    pub fn accuracy() -> accuracy::Accuracy{
        accuracy::Accuracy {}
    }
    
    pub fn recall() -> recall::Recall{
        recall::Recall {}
    }

    pub fn precision() -> precision::Precision{
        precision::Precision {}
    }
}

pub fn accuracy<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_prod: &V) -> T{
    ClassificationMetrics::accuracy().get_score(y_true, y_prod)
}

pub fn recall<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_prod: &V) -> T{
    ClassificationMetrics::recall().get_score(y_true, y_prod)
}

pub fn precision<T: FloatExt, V: BaseVector<T>>(y_true: &V, y_prod: &V) -> T{
    ClassificationMetrics::precision().get_score(y_true, y_prod)
}