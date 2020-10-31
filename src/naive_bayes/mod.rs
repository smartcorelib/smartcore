use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use std::marker::PhantomData;

pub trait NBDistribution<T: RealNumber, M: Matrix<T>> {
    // Fit distribution to some continuous or discrete data
    fn fit(x: &M, y: &M::RowVector) -> Self;

    // Prior of class k
    fn prior(&self, k: T) -> T;

    // Conditional probability of feature j give class k
    fn conditional_probability(&self, k: T, j: &M::RowVector) -> T;

    // Possible classes of the distribution
    fn classes(&self) -> Vec<T>;
}

pub struct BaseNaiveBayes<T: RealNumber, M: Matrix<T>, D: NBDistribution<T, M>> {
    distribution: D,
    _phantom_t: PhantomData<T>,
    _phantom_m: PhantomData<M>,
}

impl<T: RealNumber, M: Matrix<T>, D: NBDistribution<T, M>> BaseNaiveBayes<T, M, D> {
    pub fn fit(distribution: D) -> Result<Self, Failed> {
        Ok(Self {
            distribution,
            _phantom_t: PhantomData,
            _phantom_m: PhantomData,
        })
    }

    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        let y_classes = self.distribution.classes();
        let (rows, _) = x.shape();
        let predictions = (0..rows)
            .map(|row_index| {
                let row = x.get_row(row_index);
                let (prediction, _probability) = y_classes
                    .iter()
                    .map(|class| {
                        (
                            class,
                            self.distribution.conditional_probability(*class, &row)
                                * self.distribution.prior(*class),
                        )
                    })
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .unwrap();
                prediction.clone()
            })
            .collect::<Vec<T>>();
        let mut y_hat = M::RowVector::zeros(rows);
        for i in 0..rows {
            y_hat.set(i, predictions[i]);
        }
        Ok(y_hat)
    }
}
