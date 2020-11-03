use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use std::marker::PhantomData;

/// Distribution used in the Naive Bayes classifier.
pub trait NBDistribution<T: RealNumber, M: Matrix<T>> {
    /// Fits the distribution to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data
    /// * `y` - vector with target values (classes) of length N
    fn fit(x: &M, y: &M::RowVector) -> Self;

    /// Prior of class k
    fn prior(&self, k: T) -> T;

    /// Conditional probability of feature j given class k
    fn conditional_probability(&self, k: T, j: &M::RowVector) -> T;

    /// Possible classes of the distribution
    fn classes(&self) -> &Vec<T>;
}

/// Base struct for the Naive Bayes classifier.
pub struct BaseNaiveBayes<T: RealNumber, M: Matrix<T>, D: NBDistribution<T, M>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    struct FakeDistribution<T: RealNumber, M: Matrix<T>> {
        y_labels: Vec<T>,
        label_count: Vec<usize>,
        data_size: usize,
        x: M,
        y: M::RowVector,
    }

    impl<T: RealNumber, M: Matrix<T>> NBDistribution<T, M> for FakeDistribution<T, M> {
        fn fit(x: &M, y: &M::RowVector) -> Self {
            let x = x.clone();
            let y = y.clone();
            let mut y_sorted = y.to_vec();
            let data_size = y.len();
            y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut y_labels = vec![];
            let mut label_count = vec![];

            y_labels.push(y_sorted[0]);
            let mut current_count = 1;

            for idx in 1..data_size {
                if y_sorted[idx] == y_sorted[idx - 1] {
                    current_count += 1;
                } else {
                    label_count.push(current_count);
                    y_labels.push(y_sorted[idx]);
                    current_count = 1
                }
            }

            label_count.push(current_count);
            Self {
                data_size,
                y_labels,
                label_count,
                x,
                y,
            }
        }

        fn prior(&self, k: T) -> T {
            match self.y_labels.iter().position(|&t| t == k) {
                Some(idx) => {
                    T::from(self.label_count[idx]).unwrap() / T::from(self.data_size).unwrap()
                }
                None => T::zero(),
            }
        }

        fn conditional_probability(&self, k: T, j: &M::RowVector) -> T {
            let d = j.len();
            let mut count = 0;
            let mut probs: Vec<T> = vec![T::zero(); d];
            for idx in 0..self.data_size {
                if self.y.get(idx) != k {
                    continue;
                }
                count += 1;
                let row = self.x.get_row(idx);
                for feature in 0..d {
                    if j.get(feature) == row.get(feature) {
                        probs[feature] += T::one();
                    }
                }
            }

            if count == 0 {
                T::zero()
            } else {
                let mut prob = T::one();
                for feat in probs.into_iter() {
                    prob *= feat / T::from(count).unwrap()
                }
                prob
            }
        }

        fn classes(&self) -> &Vec<T> {
            &self.y_labels
        }
    }
    #[test]
    fn run_base_naive_bayes() {
        let x = DenseMatrix::from_2d_array(&[
            &[0., 2., 1., 0.],
            &[0., 2., 1., 1.],
            &[1., 2., 1., 0.],
            &[2., 1., 1., 0.],
            &[2., 0., 0., 0.],
            &[2., 0., 0., 1.],
            &[1., 0., 0., 1.],
            &[0., 1., 1., 0.],
            &[0., 0., 0., 0.],
            &[2., 1., 0., 0.],
            &[0., 1., 0., 1.],
            &[1., 1., 1., 1.],
            &[1., 2., 0., 0.],
            &[2., 1., 1., 1.],
        ]);
        let y = vec![0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0.];

        let distribution = FakeDistribution::<f64, DenseMatrix<f64>>::fit(&x, &y);
        let nbc = BaseNaiveBayes::fit(distribution).unwrap();
        let x_test = DenseMatrix::from_2d_array(&[&[0., 2., 1., 0.], &[2., 2., 0., 0.]]);
        let y_hat = nbc.predict(&x_test).unwrap();
        assert_eq!(y_hat, vec![0., 1.]);
    }
}
