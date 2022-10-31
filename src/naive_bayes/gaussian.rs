//! # Gaussian Naive Bayes
//!
//! Gaussian Naive Bayes is a variant of [Naive Bayes](../index.html) for the data that follows Gaussian distribution and
//! it supports continuous valued features conforming to a normal distribution.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
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
//! let y: Vec<u32> = vec![1, 1, 1, 2, 2, 2];
//!
//! let nb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
//! let y_hat = nb.predict(&x).unwrap();
//! ```
use num_traits::Unsigned;

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2, ArrayView1};
use crate::naive_bayes::{BaseNaiveBayes, NBDistribution};
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Naive Bayes classifier using Gaussian distribution
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq, Clone)]
struct GaussianNBDistribution<T: Number> {
    /// class labels known to the classifier
    class_labels: Vec<T>,
    /// number of training samples observed in each class
    class_count: Vec<usize>,
    /// probability of each class.
    class_priors: Vec<f64>,
    /// variance of each feature per class
    var: Vec<Vec<f64>>,
    /// mean of each feature per class
    theta: Vec<Vec<f64>>,
}

impl<X: Number + RealNumber, Y: Number + Ord + Unsigned> NBDistribution<X, Y>
    for GaussianNBDistribution<Y>
{
    fn prior(&self, class_index: usize) -> f64 {
        if class_index >= self.class_labels.len() {
            0f64
        } else {
            self.class_priors[class_index]
        }
    }

    fn log_likelihood<'a>(&self, class_index: usize, j: &'a Box<dyn ArrayView1<X> + 'a>) -> f64 {
        let mut likelihood = 0f64;
        for feature in 0..j.shape() {
            let value = X::to_f64(j.get(feature)).unwrap();
            let mean = self.theta[class_index][feature];
            let variance = self.var[class_index][feature];
            likelihood += self.calculate_log_probability(value, mean, variance);
        }
        likelihood
    }

    fn classes(&self) -> &Vec<Y> {
        &self.class_labels
    }
}

/// `GaussianNB` parameters. Use `Default::default()` for default values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Default, Clone)]
pub struct GaussianNBParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Option<Vec<f64>>,
}

impl GaussianNBParameters {
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub fn with_priors(mut self, priors: Vec<f64>) -> Self {
        self.priors = Some(priors);
        self
    }
}

impl GaussianNBParameters {
    fn default() -> Self {
        Self {
            priors: Option::None,
        }
    }
}

/// GaussianNB grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct GaussianNBSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Vec<Option<Vec<f64>>>,
}

/// GaussianNB grid search iterator
pub struct GaussianNBSearchParametersIterator {
    gaussian_nb_search_parameters: GaussianNBSearchParameters,
    current_priors: usize,
}

impl IntoIterator for GaussianNBSearchParameters {
    type Item = GaussianNBParameters;
    type IntoIter = GaussianNBSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        GaussianNBSearchParametersIterator {
            gaussian_nb_search_parameters: self,
            current_priors: 0,
        }
    }
}

impl Iterator for GaussianNBSearchParametersIterator {
    type Item = GaussianNBParameters;

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

impl Default for GaussianNBSearchParameters {
    fn default() -> Self {
        let default_params = GaussianNBParameters::default();

        GaussianNBSearchParameters {
            priors: vec![default_params.priors],
        }
    }
}

impl<TY: Number + Ord + Unsigned> GaussianNBDistribution<TY> {
    /// Fits the distribution to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data.
    /// * `y` - vector with target values (classes) of length N.
    /// * `priors` - Optional vector with prior probabilities of the classes. If not defined,
    /// priors are adjusted according to the data.
    pub fn fit<TX: Number + RealNumber, X: Array2<TX>, Y: Array1<TY>>(
        x: &X,
        y: &Y,
        priors: Option<Vec<f64>>,
    ) -> Result<Self, Failed> {
        let (n_samples, _) = x.shape();
        let y_samples = y.shape();
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
        let (class_labels, indices) = y.unique_with_indices();

        let mut class_count = vec![0_usize; class_labels.len()];

        let mut subdataset: Vec<Vec<Box<dyn ArrayView1<TX>>>> =
            (0..class_labels.len()).map(|_| vec![]).collect();

        for (row, class_index) in x.row_iter().zip(indices.iter()) {
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
                .map(|&c| c as f64 / n_samples as f64)
                .collect()
        };

        let subdataset: Vec<X> = subdataset
            .iter()
            .map(|v| {
                X::concatenate_1d(
                    &v.iter()
                        .map(|v| v.as_ref())
                        .collect::<Vec<&dyn ArrayView1<TX>>>(),
                    0,
                )
            })
            .collect();

        let (var, theta): (Vec<Vec<f64>>, Vec<Vec<f64>>) = subdataset
            .iter()
            .map(|data| (data.variance(0), data.mean_by(0)))
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
    fn calculate_log_probability(&self, value: f64, mean: f64, variance: f64) -> f64 {
        let pi = std::f64::consts::PI;
        -((value - mean).powf(2.0) / (2.0 * variance))
            - (2.0 * pi).ln() / 2.0
            - (variance).ln() / 2.0
    }
}

/// GaussianNB implements the naive Bayes algorithm for data that follows the Gaussian
/// distribution.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub struct GaussianNB<
    TX: Number + RealNumber + RealNumber,
    TY: Number + Ord + Unsigned,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    inner: Option<BaseNaiveBayes<TX, TY, X, Y, GaussianNBDistribution<TY>>>,
}

impl<
        TX: Number + RealNumber + RealNumber,
        TY: Number + Ord + Unsigned,
        X: Array2<TX>,
        Y: Array1<TY>,
    > SupervisedEstimator<X, Y, GaussianNBParameters> for GaussianNB<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            inner: Option::None,
        }
    }

    fn fit(x: &X, y: &Y, parameters: GaussianNBParameters) -> Result<Self, Failed> {
        GaussianNB::fit(x, y, parameters)
    }
}

impl<
        TX: Number + RealNumber + RealNumber,
        TY: Number + Ord + Unsigned,
        X: Array2<TX>,
        Y: Array1<TY>,
    > Predictor<X, Y> for GaussianNB<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number + RealNumber, TY: Number + Ord + Unsigned, X: Array2<TX>, Y: Array1<TY>>
    GaussianNB<TX, TY, X, Y>
{
    /// Fits GaussianNB with given data
    /// * `x` - training data of size NxM where N is the number of samples and M is the number of
    /// features.
    /// * `y` - vector with target values (classes) of length N.
    /// * `parameters` - additional parameters like class priors.
    pub fn fit(x: &X, y: &Y, parameters: GaussianNBParameters) -> Result<Self, Failed> {
        let distribution = GaussianNBDistribution::fit(x, y, parameters.priors)?;
        let inner = BaseNaiveBayes::fit(distribution)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Estimates the class labels for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    /// Returns a vector of size N with class estimates.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.inner.as_ref().unwrap().predict(x)
    }

    /// Class labels known to the classifier.
    /// Returns a vector of size n_classes.
    pub fn classes(&self) -> &Vec<TY> {
        &self.inner.as_ref().unwrap().distribution.class_labels
    }

    /// Number of training samples observed in each class.
    /// Returns a vector of size n_classes.
    pub fn class_count(&self) -> &Vec<usize> {
        &self.inner.as_ref().unwrap().distribution.class_count
    }

    /// Probability of each class
    /// Returns a vector of size n_classes.
    pub fn class_priors(&self) -> &Vec<f64> {
        &self.inner.as_ref().unwrap().distribution.class_priors
    }

    /// Mean of each feature per class
    /// Returns a 2d vector of shape (n_classes, n_features).
    pub fn theta(&self) -> &Vec<Vec<f64>> {
        &self.inner.as_ref().unwrap().distribution.theta
    }

    /// Variance of each feature per class
    /// Returns a 2d vector of shape (n_classes, n_features).
    pub fn var(&self) -> &Vec<Vec<f64>> {
        &self.inner.as_ref().unwrap().distribution.var
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

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
        let y: Vec<u32> = vec![1, 1, 1, 2, 2, 2];

        let gnb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
        let y_hat = gnb.predict(&x).unwrap();
        assert_eq!(y_hat, y);

        assert_eq!(gnb.classes(), &[1, 2]);

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
        let y: Vec<u32> = vec![1, 1, 1, 2, 2, 2];

        let priors = vec![0.3, 0.7];
        let parameters = GaussianNBParameters::default().with_priors(priors.clone());
        let gnb = GaussianNB::fit(&x, &y, parameters).unwrap();

        assert_eq!(gnb.class_priors(), &priors);
    }

    // TODO: implement serialization
    // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    // #[test]
    // #[cfg(feature = "serde")]
    // fn serde() {
    //     let x = DenseMatrix::<f64>::from_2d_array(&[
    //         &[-1., -1.],
    //         &[-2., -1.],
    //         &[-3., -2.],
    //         &[1., 1.],
    //         &[2., 1.],
    //         &[3., 2.],
    //     ]);
    //     let y: Vec<u32> = vec![1, 1, 1, 2, 2, 2];

    //     let gnb = GaussianNB::fit(&x, &y, Default::default()).unwrap();
    //     let deserialized_gnb: GaussianNB<f64, u32, DenseMatrix<f64>, Vec<u32>> =
    //         serde_json::from_str(&serde_json::to_string(&gnb).unwrap()).unwrap();

    //     assert_eq!(gnb, deserialized_gnb);
    // }
}
