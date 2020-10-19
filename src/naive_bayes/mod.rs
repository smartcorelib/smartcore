use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
pub trait NBDistribution {
    // Fit distribution to some continuous or discrete data
    fn fit<T: RealNumber, M: Matrix<T>>(x: &M, y: &M::RowVector) -> Self;

    // Prior of class k
    fn prior<T: RealNumber>(&self, k: T) -> T;

    // Conditional probability of feature j give class k
    fn conditional_probability<T: RealNumber, M: Matrix<T>>(&self, k: T, j: &M::RowVector) -> T;

    // Possible classes of the distribution
    fn classes<T: RealNumber>(&self) -> Vec<T>;
}

pub struct BaseNaiveBayes<D: NBDistribution> {
    distribution: D,
}

impl<D: NBDistribution> BaseNaiveBayes<D> {
    pub fn fit(distribution: D) -> Result<Self, Failed> {
        Ok(Self { distribution })
    }

    pub fn predict<T: RealNumber, M: Matrix<T>>(&self, x: &M) -> Result<M::RowVector, Failed> {
        let y_classes = self.distribution.classes::<T>();
        let (rows, _) = x.shape();
        let predictions = (0..rows)
            .map(|row_index| {
                let row = x.get_row(row_index);
                let (prediction, _probability) = y_classes
                    .iter()
                    .map(|class| {
                        (
                            class,
                            self.distribution
                                .conditional_probability::<T, M>(*class, &row)
                                * self.distribution.prior::<T>(*class),
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
