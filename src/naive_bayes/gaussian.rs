//! # Gaussian Naive Bayes
//!
//! Gaussian Naive Bayes is a variant of [Naive Bayes](../index.html) for the data that follows Gaussian distribution and
//! it supports continuous valued features conforming to a normal distribution.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::naive_bayes::gaussian::GaussianNB;
//!
//! let x = DenseMatrix::from_2d_array(&[
//!              &[-1., -1.],
//!              &[-2., -1.],
//!              &[-3., -2.],
//!              &[ 1.,  1.],
//!              &[ 2.,  1.],
//!              &[ 3.,  2.],
//!          ]);
//! let y = vec![1., 1., 1., 2., 2., 2.];
//!
//! let nb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
//! let y_hat = nb.predict(&x).unwrap();
//! ```
use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::row_iter;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::math::vector::RealNumberVector;
use crate::naive_bayes::{BaseNaiveBayes, NBDistribution};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Naive Bayes classifier using Gaussian distribution
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
struct GaussianNBDistribution<T: RealNumber> {
    /// class labels known to the classifier
    class_labels: Vec<T>,
    /// number of training samples observed in each class
    class_count: Vec<usize>,
    /// probability of each class.
    class_priors: Vec<T>,
    /// variance of each feature per class
    var: Vec<Vec<T>>,
    /// mean of each feature per class
    theta: Vec<Vec<T>>,
}

impl<T: RealNumber, M: Matrix<T>> NBDistribution<T, M> for GaussianNBDistribution<T> {
    fn prior(&self, class_index: usize) -> T {
        if class_index >= self.class_labels.len() {
            T::zero()
        } else {
            self.class_priors[class_index]
        }
    }

    fn log_likelihood(&self, class_index: usize, j: &M::RowVector) -> T {
        let mut likelihood = T::zero();
        for feature in 0..j.len() {
            let value = j.get(feature);
            let mean = self.theta[class_index][feature];
            let variance = self.var[class_index][feature];
            likelihood += self.calculate_log_probability(value, mean, variance);
        }
        likelihood
    }

    fn classes(&self) -> &Vec<T> {
        &self.class_labels
    }
}

/// `GaussianNB` parameters. Use `Default::default()` for default values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct GaussianNBParameters<T: RealNumber> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Option<Vec<T>>,
}

impl<T: RealNumber> GaussianNBParameters<T> {
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub fn with_priors(mut self, priors: Vec<T>) -> Self {
        self.priors = Some(priors);
        self
    }
}

impl<T: RealNumber> Default for GaussianNBParameters<T> {
    fn default() -> Self {
        Self { priors: None }
    }
}

/// GaussianNB grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct GaussianNBSearchParameters<T: RealNumber> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Vec<Option<Vec<T>>>,
}

/// GaussianNB grid search iterator
pub struct GaussianNBSearchParametersIterator<T: RealNumber> {
    gaussian_nb_search_parameters: GaussianNBSearchParameters<T>,
    current_priors: usize,
}

impl<T: RealNumber> IntoIterator for GaussianNBSearchParameters<T> {
    type Item = GaussianNBParameters<T>;
    type IntoIter = GaussianNBSearchParametersIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        GaussianNBSearchParametersIterator {
            gaussian_nb_search_parameters: self,
            current_priors: 0,
        }
    }
}

impl<T: RealNumber> Iterator for GaussianNBSearchParametersIterator<T> {
    type Item = GaussianNBParameters<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_priors == self.gaussian_nb_search_parameters.priors.len() {
            return None;
        }

        let next = GaussianNBParameters {
            priors: self.gaussian_nb_search_parameters.priors[self.current_priors].clone(),
        };

        self.current_priors += 1;

        Some(next)
    }
}

impl<T: RealNumber> Default for GaussianNBSearchParameters<T> {
    fn default() -> Self {
        let default_params = GaussianNBParameters::default();

        GaussianNBSearchParameters {
            priors: vec![default_params.priors],
        }
    }
}

impl<T: RealNumber> GaussianNBDistribution<T> {
    /// Fits the distribution to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data.
    /// * `y` - vector with target values (classes) of length N.
    /// * `priors` - Optional vector with prior probabilities of the classes. If not defined,
    /// priors are adjusted according to the data.
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        priors: Option<Vec<T>>,
    ) -> Result<Self, Failed> {
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
        let y = y.to_vec();
        let (class_labels, indices) = <Vec<T> as RealNumberVector<T>>::unique_with_indices(&y);

        let mut class_count = vec![0_usize; class_labels.len()];

        let mut subdataset: Vec<Vec<Vec<T>>> = vec![vec![]; class_labels.len()];

        for (row, class_index) in row_iter(x).zip(indices.iter()) {
            class_count[*class_index] += 1;
            subdataset[*class_index].push(row);
        }

        let class_priors = if let Some(class_priors) = priors {
            if class_priors.len() != class_labels.len() {
                return Err(Failed::fit(
                    "Size of priors provided does not match the number of classes of the data.",
                ));
            }
            class_priors
        } else {
            class_count
                .iter()
                .map(|&c| T::from(c).unwrap() / T::from(n_samples).unwrap())
                .collect()
        };

        let subdataset: Vec<M> = subdataset
            .into_iter()
            .map(|v| {
                let mut m = M::zeros(v.len(), n_features);
                for (row_i, v_i) in v.iter().enumerate() {
                    for (col_j, v_i_j) in v_i.iter().enumerate().take(n_features) {
                        m.set(row_i, col_j, *v_i_j);
                    }
                }
                m
            })
            .collect();

        let (var, theta): (Vec<Vec<T>>, Vec<Vec<T>>) = subdataset
            .iter()
            .map(|data| (data.var(0), data.mean(0)))
            .unzip();

        Ok(Self {
            class_labels,
            class_count,
            class_priors,
            var,
            theta,
        })
    }

    /// Calculate probability of x equals to a value of a Gaussian distribution given its mean and its
    /// variance.
    fn calculate_log_probability(&self, value: T, mean: T, variance: T) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap();
        -((value - mean).powf(T::two()) / (T::two() * variance))
            - (T::two() * pi).ln() / T::two()
            - (variance).ln() / T::two()
    }
}

/// GaussianNB implements the naive Bayes algorithm for data that follows the Gaussian
/// distribution.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub struct GaussianNB<T: RealNumber, M: Matrix<T>> {
    inner: BaseNaiveBayes<T, M, GaussianNBDistribution<T>>,
}

impl<T: RealNumber, M: Matrix<T>> SupervisedEstimator<M, M::RowVector, GaussianNBParameters<T>>
    for GaussianNB<T, M>
{
    fn fit(x: &M, y: &M::RowVector, parameters: GaussianNBParameters<T>) -> Result<Self, Failed> {
        GaussianNB::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>> Predictor<M, M::RowVector> for GaussianNB<T, M> {
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber, M: Matrix<T>> GaussianNB<T, M> {
    /// Fits GaussianNB with given data
    /// * `x` - training data of size NxM where N is the number of samples and M is the number of
    /// features.
    /// * `y` - vector with target values (classes) of length N.
    /// * `parameters` - additional parameters like class priors.
    pub fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: GaussianNBParameters<T>,
    ) -> Result<Self, Failed> {
        let distribution = GaussianNBDistribution::fit(x, y, parameters.priors)?;
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

    /// Probability of each class
    /// Returns a vector of size n_classes.
    pub fn class_priors(&self) -> &Vec<T> {
        &self.inner.distribution.class_priors
    }

    /// Mean of each feature per class
    /// Returns a 2d vector of shape (n_classes, n_features).
    pub fn theta(&self) -> &Vec<Vec<T>> {
        &self.inner.distribution.theta
    }

    /// Variance of each feature per class
    /// Returns a 2d vector of shape (n_classes, n_features).
    pub fn var(&self) -> &Vec<Vec<T>> {
        &self.inner.distribution.var
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn search_parameters() {
        let parameters = GaussianNBSearchParameters {
            priors: vec![Some(vec![1.]), Some(vec![2.])],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.priors, Some(vec![1.]));
        let next = iter.next().unwrap();
        assert_eq!(next.priors, Some(vec![2.]));
        assert!(iter.next().is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn run_gaussian_naive_bayes() {
        let x = DenseMatrix::from_2d_array(&[
            &[-1., -1.],
            &[-2., -1.],
            &[-3., -2.],
            &[1., 1.],
            &[2., 1.],
            &[3., 2.],
        ]);
        let y = vec![1., 1., 1., 2., 2., 2.];

        let gnb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
        let y_hat = gnb.predict(&x).unwrap();
        assert_eq!(y_hat, y);

        assert_eq!(gnb.classes(), &[1., 2.]);

        assert_eq!(gnb.class_count(), &[3, 3]);

        assert_eq!(
            gnb.var(),
            &[
                &[0.666666666666667, 0.22222222222222232],
                &[0.666666666666667, 0.22222222222222232]
            ]
        );

        assert_eq!(gnb.class_priors(), &[0.5, 0.5]);

        assert_eq!(
            gnb.theta(),
            &[&[-2., -1.3333333333333333], &[2., 1.3333333333333333]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn run_gaussian_naive_bayes_with_priors() {
        let x = DenseMatrix::from_2d_array(&[
            &[-1., -1.],
            &[-2., -1.],
            &[-3., -2.],
            &[1., 1.],
            &[2., 1.],
            &[3., 2.],
        ]);
        let y = vec![1., 1., 1., 2., 2., 2.];

        let priors = vec![0.3, 0.7];
        let parameters = GaussianNBParameters::default().with_priors(priors.clone());
        let gnb = GaussianNB::fit(&x, &y, parameters).unwrap();

        assert_eq!(gnb.class_priors(), &priors);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[-1., -1.],
            &[-2., -1.],
            &[-3., -2.],
            &[1., 1.],
            &[2., 1.],
            &[3., 2.],
        ]);
        let y = vec![1., 1., 1., 2., 2., 2.];

        let gnb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
        let deserialized_gnb: GaussianNB<f64, DenseMatrix<f64>> =
            serde_json::from_str(&serde_json::to_string(&gnb).unwrap()).unwrap();

        assert_eq!(gnb, deserialized_gnb);
    }
}
