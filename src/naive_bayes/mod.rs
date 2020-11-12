use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use std::marker::PhantomData;

/// Distribution used in the Naive Bayes classifier.
pub(crate) trait NBDistribution<T: RealNumber, M: Matrix<T>> {
    /// Prior of class at the given index.
    fn prior(&self, class_index: usize) -> T;

    /// Conditional probability of sample j given class in the specified index.
    fn conditional_probability(&self, class_index: usize, j: &M::RowVector) -> T;

    /// Possible classes of the distribution.
    fn classes(&self) -> &Vec<T>;
}

/// Base struct for the Naive Bayes classifier.
pub(crate) struct BaseNaiveBayes<T: RealNumber, M: Matrix<T>, D: NBDistribution<T, M>> {
    distribution: D,
    _phantom_t: PhantomData<T>,
    _phantom_m: PhantomData<M>,
}

impl<T: RealNumber, M: Matrix<T>, D: NBDistribution<T, M>> BaseNaiveBayes<T, M, D> {
    /// Fits NB classifier to a given NBdistribution.
    /// * `distribution` - NBDistribution of the training data
    pub fn fit(distribution: D) -> Result<Self, Failed> {
        Ok(Self {
            distribution,
            _phantom_t: PhantomData,
            _phantom_m: PhantomData,
        })
    }

    /// Estimates the class labels for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    /// Returns a vector of size N with class estimates.
    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        let y_classes = self.distribution.classes();
        let (rows, _) = x.shape();
        let predictions = (0..rows)
            .map(|row_index| {
                let row = x.get_row(row_index);
                let (prediction, _probability) = y_classes
                    .iter()
                    .enumerate()
                    .map(|(class_index, class)| {
                        (
                            class,
                            self.distribution.conditional_probability(class_index, &row)
                                * self.distribution.prior(class_index),
                        )
                    })
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .unwrap();
                *prediction
            })
            .collect::<Vec<T>>();
        let y_hat = M::RowVector::from_slice(&predictions);
        Ok(y_hat)
    }
}
mod categorical;
pub use categorical::{CategoricalNB, CategoricalNBParameters};
