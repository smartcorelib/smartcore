//! # Categorical Naive Bayes
//!
//! Categorical Naive Bayes is a variant of [Naive Bayes](../index.html) for the categorically distributed data.
//! It assumes that each feature has its own categorical distribution.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::naive_bayes::categorical::CategoricalNB;
//!
//! let x = DenseMatrix::from_2d_array(&[
//!              &[3., 4., 0., 1.],
//!              &[3., 0., 0., 1.],
//!              &[4., 4., 1., 2.],
//!              &[4., 2., 4., 3.],
//!              &[4., 2., 4., 2.],
//!              &[4., 1., 1., 0.],
//!              &[1., 1., 1., 1.],
//!              &[0., 4., 1., 0.],
//!              &[0., 3., 2., 1.],
//!              &[0., 3., 1., 1.],
//!              &[3., 4., 0., 1.],
//!              &[3., 4., 2., 4.],
//!              &[0., 3., 1., 2.],
//!              &[0., 4., 1., 2.],
//!          ]);
//! let y = vec![0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0.];
//!
//! let nb = CategoricalNB::fit(&x, &y, Default::default()).unwrap();
//! let y_hat = nb.predict(&x).unwrap();
//! ```
use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::naive_bayes::{BaseNaiveBayes, NBDistribution};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Naive Bayes classifier for categorical features
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
struct CategoricalNBDistribution<T: RealNumber> {
    class_count: Vec<usize>,
    class_labels: Vec<T>,
    class_priors: Vec<T>,
    coefficients: Vec<Vec<Vec<T>>>,
}

impl<T: RealNumber> PartialEq for CategoricalNBDistribution<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.class_labels == other.class_labels && self.class_priors == other.class_priors {
            if self.coefficients.len() != other.coefficients.len() {
                return false;
            }
            for (a, b) in self.coefficients.iter().zip(other.coefficients.iter()) {
                if a.len() != b.len() {
                    return false;
                }
                for (a_i, b_i) in a.iter().zip(b.iter()) {
                    if a_i.len() != b_i.len() {
                        return false;
                    }
                    for (a_i_j, b_i_j) in a_i.iter().zip(b_i.iter()) {
                        if (*a_i_j - *b_i_j).abs() > T::epsilon() {
                            return false;
                        }
                    }
                }
            }
            true
        } else {
            false
        }
    }
}

impl<T: RealNumber, M: Matrix<T>> NBDistribution<T, M> for CategoricalNBDistribution<T> {
    fn prior(&self, class_index: usize) -> T {
        if class_index >= self.class_labels.len() {
            T::zero()
        } else {
            self.class_priors[class_index]
        }
    }

    fn log_likelihood(&self, class_index: usize, j: &M::RowVector) -> T {
        if class_index < self.class_labels.len() {
            let mut likelihood = T::zero();
            for feature in 0..j.len() {
                let value = j.get(feature).floor().to_usize().unwrap();
                if self.coefficients[class_index][feature].len() > value {
                    likelihood += self.coefficients[class_index][feature][value];
                } else {
                    return T::zero();
                }
            }
            likelihood
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
        let y: Vec<usize> = y
            .to_vec()
            .iter()
            .map(|y_i| y_i.floor().to_usize().unwrap())
            .collect();

        let y_max = y
            .iter()
            .max()
            .ok_or_else(|| Failed::fit(&"Failed to get the labels of y.".to_string()))?;

        let class_labels: Vec<T> = (0..*y_max + 1)
            .map(|label| T::from(label).unwrap())
            .collect();
        let mut class_count = vec![0_usize; class_labels.len()];
        for elem in y.iter() {
            class_count[*elem] += 1;
        }

        let mut feature_categories: Vec<Vec<T>> = Vec::with_capacity(n_features);
        for feature in 0..n_features {
            let feature_max = x
                .get_col_as_vec(feature)
                .iter()
                .map(|f_i| f_i.floor().to_usize().unwrap())
                .max()
                .ok_or_else(|| {
                    Failed::fit(&format!(
                        "Failed to get the categories for feature = {}",
                        feature
                    ))
                })?;
            let feature_types = (0..feature_max + 1)
                .map(|feat| T::from(feat).unwrap())
                .collect();
            feature_categories.push(feature_types);
        }

        let mut coefficients: Vec<Vec<Vec<T>>> = Vec::with_capacity(class_labels.len());
        for (label, &label_count) in class_labels.iter().zip(class_count.iter()) {
            let mut coef_i: Vec<Vec<T>> = Vec::with_capacity(n_features);
            for (feature_index, feature_options) in
                feature_categories.iter().enumerate().take(n_features)
            {
                let col = x
                    .get_col_as_vec(feature_index)
                    .iter()
                    .enumerate()
                    .filter(|(i, _j)| T::from(y[*i]).unwrap() == *label)
                    .map(|(_, j)| *j)
                    .collect::<Vec<T>>();
                let mut feat_count: Vec<T> = vec![T::zero(); feature_options.len()];
                for row in col.iter() {
                    let index = row.floor().to_usize().unwrap();
                    feat_count[index] += T::one();
                }
                let coef_i_j = feat_count
                    .iter()
                    .map(|c| {
                        ((*c + alpha)
                            / (T::from(label_count).unwrap()
                                + T::from(feature_options.len()).unwrap() * alpha))
                            .ln()
                    })
                    .collect::<Vec<T>>();
                coef_i.push(coef_i_j);
            }
            coefficients.push(coef_i);
        }

        let class_priors = class_count
            .iter()
            .map(|&count| T::from(count).unwrap() / T::from(n_samples).unwrap())
            .collect::<Vec<T>>();

        Ok(Self {
            class_count,
            class_labels,
            class_priors,
            coefficients,
        })
    }
}

/// `CategoricalNB` parameters. Use `Default::default()` for default values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct CategoricalNBParameters<T: RealNumber> {
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: T,
}

impl<T: RealNumber> CategoricalNBParameters<T> {
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<T: RealNumber> Default for CategoricalNBParameters<T> {
    fn default() -> Self {
        Self { alpha: T::one() }
    }
}

/// CategoricalNB implements the categorical naive Bayes algorithm for categorically distributed data.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub struct CategoricalNB<T: RealNumber, M: Matrix<T>> {
    inner: BaseNaiveBayes<T, M, CategoricalNBDistribution<T>>,
}

impl<T: RealNumber, M: Matrix<T>> SupervisedEstimator<M, M::RowVector, CategoricalNBParameters<T>>
    for CategoricalNB<T, M>
{
    fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: CategoricalNBParameters<T>,
    ) -> Result<Self, Failed> {
        CategoricalNB::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>> Predictor<M, M::RowVector> for CategoricalNB<T, M> {
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
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

    /// Class labels known to the classifier.
    /// Returns a vector of size n_classes.
    pub fn classes(&self) -> &Vec<T> {
        &self.inner.distribution.class_labels
    }

    /// Number of training samples observed in each class.
    /// Returns a vector of size n_classes.
    pub fn class_count(&self) -> &Vec<usize> {
        &self.inner.distribution.class_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn run_categorical_naive_bayes() {
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

        assert_eq!(cnb.classes(), &[0., 1.]);
        assert_eq!(cnb.class_count(), &[5, 9]);

        let x_test = DenseMatrix::from_2d_array(&[&[0., 2., 1., 0.], &[2., 2., 0., 0.]]);
        let y_hat = cnb.predict(&x_test).unwrap();
        assert_eq!(y_hat, vec![0., 1.]);
    }

    #[test]
    fn run_categorical_naive_bayes2() {
        let x = DenseMatrix::from_2d_array(&[
            &[3., 4., 0., 1.],
            &[3., 0., 0., 1.],
            &[4., 4., 1., 2.],
            &[4., 2., 4., 3.],
            &[4., 2., 4., 2.],
            &[4., 1., 1., 0.],
            &[1., 1., 1., 1.],
            &[0., 4., 1., 0.],
            &[0., 3., 2., 1.],
            &[0., 3., 1., 1.],
            &[3., 4., 0., 1.],
            &[3., 4., 2., 4.],
            &[0., 3., 1., 2.],
            &[0., 4., 1., 2.],
        ]);
        let y = vec![0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0.];

        let cnb = CategoricalNB::fit(&x, &y, Default::default()).unwrap();
        let y_hat = cnb.predict(&x).unwrap();
        assert_eq!(
            y_hat,
            vec![0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1., 1.]
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[3., 4., 0., 1.],
            &[3., 0., 0., 1.],
            &[4., 4., 1., 2.],
            &[4., 2., 4., 3.],
            &[4., 2., 4., 2.],
            &[4., 1., 1., 0.],
            &[1., 1., 1., 1.],
            &[0., 4., 1., 0.],
            &[0., 3., 2., 1.],
            &[0., 3., 1., 1.],
            &[3., 4., 0., 1.],
            &[3., 4., 2., 4.],
            &[0., 3., 1., 2.],
            &[0., 4., 1., 2.],
        ]);

        let y = vec![0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0.];
        let cnb = CategoricalNB::fit(&x, &y, Default::default()).unwrap();

        let deserialized_cnb: CategoricalNB<f64, DenseMatrix<f64>> =
            serde_json::from_str(&serde_json::to_string(&cnb).unwrap()).unwrap();

        assert_eq!(cnb, deserialized_cnb);
    }
}
