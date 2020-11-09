use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::naive_bayes::{BaseNaiveBayes, NBDistribution};
use serde::{Deserialize, Serialize};

/// Naive Bayes classifier for categorical features
struct CategoricalNBDistribution<T: RealNumber> {
    class_labels: Vec<T>,
    class_probabilities: Vec<T>,
    coef: Vec<Vec<Vec<T>>>,
    feature_categories: Vec<Vec<T>>,
}

impl<T: RealNumber, M: Matrix<T>> NBDistribution<T, M> for CategoricalNBDistribution<T> {
    fn prior(&self, class_index: usize) -> T {
        if class_index >= self.class_labels.len() {
            T::zero()
        } else {
            self.class_probabilities[class_index]
        }
    }

    fn conditional_probability(&self, class_index: usize, j: &M::RowVector) -> T {
        if class_index < self.class_labels.len() {
            let mut prob = T::one();
            for feature in 0..j.len() {
                let value = j.get(feature);
                match self.feature_categories[feature]
                    .iter()
                    .position(|&t| t == value)
                {
                    Some(_i) => prob *= self.coef[class_index][feature][_i],
                    None => return T::zero(),
                }
            }
            prob
        } else {
            T::zero()
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
    pub fn fit<M: Matrix<T>>(x: &M, y: &M::RowVector, alpha: T) -> Result<Self, Failed> {
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

/// `CategoricalNB` parameters. Use `Default::default()` for default values.
#[derive(Serialize, Deserialize, Debug)]
pub struct CategoricalNBParameters<T: RealNumber> {
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: T,
}

impl<T: RealNumber> CategoricalNBParameters<T> {
    /// Create CategoricalNBParameters with specific paramaters.
    pub fn new(alpha: T) -> Result<Self, Failed> {
        if alpha > T::zero() {
            Ok(Self { alpha })
        } else {
            Err(Failed::fit(&format!(
                "alpha should be >= 0, alpha=[{}]",
                alpha
            )))
        }
    }
}
impl<T: RealNumber> Default for CategoricalNBParameters<T> {
    fn default() -> Self {
        Self { alpha: T::one() }
    }
}

/// CategoricalNB implements the categorical naive Bayes algorithm for categorically distributed data.
pub struct CategoricalNB<T: RealNumber, M: Matrix<T>> {
    inner: BaseNaiveBayes<T, M, CategoricalNBDistribution<T>>,
}

impl<T: RealNumber, M: Matrix<T>> CategoricalNB<T, M> {
    /// Fits CategoricalNB with given data
    /// * `x` - training data of size NxM where N is the number of samples and M is the number of
    /// features.
    /// * `y` - vector with target values (classes) of length N.
    /// * `parameters` - additional parameters like alpha for smoothing
    pub fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: CategoricalNBParameters<T>,
    ) -> Result<Self, Failed> {
        let alpha = parameters.alpha;
        let distribution = CategoricalNBDistribution::fit(x, y, alpha)?;
        let inner = BaseNaiveBayes::fit(distribution)?;
        Ok(Self { inner })
    }

    /// Estimates the class labels for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    /// Returns a vector of size N with class estimates.
    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.inner.predict(x)
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

        let cnb = CategoricalNB::fit(&x, &y, Default::default()).unwrap();
        let x_test = DenseMatrix::from_2d_array(&[&[0., 2., 1., 0.], &[2., 2., 0., 0.]]);
        let y_hat = cnb.predict(&x_test).unwrap();
        assert_eq!(y_hat, vec![0., 1.]);
    }
}
