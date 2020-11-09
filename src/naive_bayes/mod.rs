use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use std::marker::PhantomData;

/// Distribution used in the Naive Bayes classifier.
pub(crate) trait NBDistribution<T: RealNumber, M: Matrix<T>> {
    /// Prior of class k
    fn prior(&self, k: T) -> T;

    /// Conditional probability of sample j given class k
    fn conditional_probability(&self, k: T, j: &M::RowVector) -> T;

    /// Possible classes of the distribution
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
                    .map(|class| {
                        (
                            class,
                            self.distribution.conditional_probability(*class, &row)
                                * self.distribution.prior(*class),
                        )
                    })
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .unwrap();
                *prediction
            })
            .collect::<Vec<T>>();
        let mut y_hat = M::RowVector::zeros(rows);
        for (i, prediction) in predictions.iter().enumerate().take(rows) {
            y_hat.set(i, *prediction);
        }
        Ok(y_hat)
    }
}

/// Naive Bayes classifier for categorical features
pub(crate) struct CategoricalNBDistribution<T: RealNumber> {
    class_labels: Vec<T>,
    class_probabilities: Vec<T>,
    coef: Vec<Vec<Vec<T>>>,
    feature_categories: Vec<Vec<T>>,
}

impl<T: RealNumber, M: Matrix<T>> NBDistribution<T, M> for CategoricalNBDistribution<T> {
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
                    match self.feature_categories[feature]
                        .iter()
                        .position(|&t| t == value)
                    {
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

impl<T: RealNumber> CategoricalNBDistribution<T> {
    /// Fits the distribution to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data.
    /// * `y` - vector with target values (classes) of length N.
    /// * `alpha` - Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub fn fit<M: Matrix<T>>(x: &M, y: &M::RowVector, alpha: Option<T>) -> Result<Self, Failed> {
        let alpha = alpha.unwrap_or(T::one());
        if alpha < T::zero() {
            return Err(Failed::fit(&format!(
                "alpha should be >= 0, alpha=[{}]",
                alpha
            )));
        }

        let (n_samples, n_features) = x.shape();
        let y_samples = y.len();
        if y_samples != n_samples {
            return Err(Failed::fit(&format!(
                "Size of x should equal size of y; |x|=[{}], |y|=[{}]",
                n_samples, y_samples
            )));
        }

        if n_samples == 0 {
            return Err(Failed::fit(&format!(
                "Size of x and y should greater than 0; |x|=[{}]",
                n_samples
            )));
        }

        let mut y_sorted = y.to_vec();
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut class_labels = Vec::with_capacity(y.len());
        class_labels.push(y_sorted[0]);
        let mut classes_count = Vec::with_capacity(y.len());
        let mut current_count = T::one();
        for idx in 1..y_samples {
            if y_sorted[idx] == y_sorted[idx - 1] {
                current_count += T::one();
            } else {
                classes_count.push(current_count);
                class_labels.push(y_sorted[idx]);
                current_count = T::one()
            }
            classes_count.push(current_count);
        }

        let mut feature_categories: Vec<Vec<T>> = Vec::with_capacity(n_features);

        for feature in 0..n_features {
            let feature_types = x.get_col_as_vec(feature).unique();
            feature_categories.push(feature_types);
        }
        let mut coef: Vec<Vec<Vec<T>>> = Vec::with_capacity(class_labels.len());
        for (label, label_count) in class_labels.iter().zip(classes_count.iter()) {
            let mut coef_i: Vec<Vec<T>> = Vec::with_capacity(n_features);
            for (feature_index, feature_options) in
                feature_categories.iter().enumerate().take(n_features)
            {
                let col = x
                    .get_col_as_vec(feature_index)
                    .iter()
                    .enumerate()
                    .filter(|(i, _j)| y.get(*i) == *label)
                    .map(|(_, j)| *j)
                    .collect::<Vec<T>>();
                let mut feat_count: Vec<usize> = Vec::with_capacity(feature_options.len());
                for k in feature_options.iter() {
                    let feat_k_count = col.iter().filter(|&v| v == k).count();
                    feat_count.push(feat_k_count);
                }

                let coef_i_j = feat_count
                    .iter()
                    .map(|c| {
                        (T::from(*c).unwrap() + alpha)
                            / (T::from(*label_count).unwrap()
                                + T::from(feature_options.len()).unwrap() * alpha)
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

        Ok(Self {
            class_labels,
            class_probabilities,
            coef,
            feature_categories,
        })
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

        let distribution = CategoricalNBDistribution::<f64>::fit(&x, &y, None).unwrap();
        let nbc = BaseNaiveBayes::fit(distribution).unwrap();
        let x_test = DenseMatrix::from_2d_array(&[&[0., 2., 1., 0.], &[2., 2., 0., 0.]]);
        let y_hat = nbc.predict(&x_test).unwrap();
        assert_eq!(y_hat, vec![0., 1.]);
    }
}
