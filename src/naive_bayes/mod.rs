use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::math::vector::RealNumberVector;
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

struct CategoricalNB<T: RealNumber> {
    class_labels: Vec<T>,
    class_probabilities: Vec<T>,
    coef: Vec<Vec<Vec<T>>>,
    feature_kinds: Vec<Vec<T>>,
    alpha: T,
}

impl<T: RealNumber, M: Matrix<T>> NBDistribution<T, M> for CategoricalNB<T> {
    fn fit(x: &M, y: &M::RowVector) -> Self {
        let alpha = T::one();
        let (n_samples, n_features) = x.shape();
        let y = y.to_vec();
        let (class_labels, y_indices) = <Vec<T> as RealNumberVector<T>>::unique(&y);
        let mut classes_count: Vec<T> = vec![T::zero(); class_labels.len()];
        for index in y_indices.into_iter() {
            classes_count[index] += T::one();
        }

        let x = x.transpose();
        let mut feature_kinds: Vec<Vec<T>> = Vec::with_capacity(n_features);
        for feature in 0..n_features {
            let feature_types = x.get_row(feature).unique();
            feature_kinds.push(feature_types);
        }
        let mut coef: Vec<Vec<Vec<T>>> = Vec::with_capacity(class_labels.len());
        for (label, label_count) in class_labels.iter().zip(classes_count.iter()) {
            let mut coef_i: Vec<Vec<T>> = Vec::with_capacity(n_features);
            for feature in 0..n_features {
                let row = x
                    .get_row_as_vec(feature)
                    .iter()
                    .enumerate()
                    .filter(|(i, _j)| y[*i] == *label)
                    .map(|(_, j)| j.clone())
                    .collect::<Vec<T>>();
                let mut feat_count: Vec<usize> = Vec::with_capacity(feature_kinds[feature].len());
                for k in feature_kinds[feature].iter() {
                    let feat_k_count = row.iter().filter(|&v| v == k).collect::<Vec<_>>().len();
                    feat_count.push(feat_k_count);
                }

                let coef_i_j = feat_count
                    .iter()
                    .map(|c| {
                        (T::from(*c).unwrap() + alpha)
                            / (T::from(*label_count).unwrap()
                                + T::from(feature_kinds[feature].len()).unwrap() * alpha)
                    })
                    .collect::<Vec<T>>();
                coef_i.push(coef_i_j);
            }
            coef.push(coef_i);
        }
        let class_probabilities = classes_count
            .into_iter()
            .map(|count| count / T::from(n_samples).unwrap())
            .collect::<Vec<T>>();

        Self {
            alpha,
            coef,
            class_labels,
            class_probabilities,
            feature_kinds,
        }
    }

    fn prior(&self, k: T) -> T {
        match self.class_labels.iter().position(|&t| t == k) {
            Some(idx) => self.class_probabilities[idx],
            None => T::zero(),
        }
    }

    fn conditional_probability(&self, k: T, j: &M::RowVector) -> T {
        match self.class_labels.iter().position(|&t| t == k) {
            Some(idx) => {
                let mut prob = T::one();
                for feature in 0..j.len() {
                    let value = j.get(feature);
                    match self.feature_kinds[feature].iter().position(|&t| t == value) {
                        Some(_i) => prob *= self.coef[idx][feature][_i],
                        None => return T::zero(),
                    }
                }
                prob
            }
            None => T::zero(),
        }
    }

    fn classes(&self) -> &Vec<T> {
        &self.class_labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

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

        let distribution = CategoricalNB::<f64>::fit(&x, &y);
        let nbc = BaseNaiveBayes::fit(distribution).unwrap();
        let x_test = DenseMatrix::from_2d_array(&[&[0., 2., 1., 0.], &[2., 2., 0., 0.]]);
        let y_hat = nbc.predict(&x_test).unwrap();
        assert_eq!(y_hat, vec![0., 1.]);
    }
}
