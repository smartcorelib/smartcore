//! # Bernoulli Naive Bayes
//!
//! Bernoulli Naive Bayes classifier is a variant of [Naive Bayes](../index.html) for the data that is distributed according to multivariate Bernoulli distribution.
//! It is used for discrete data with binary features. One example of a binary feature is a word that occurs in the text or not.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::naive_bayes::bernoulli::BernoulliNB;
//!
//! // Training data points are:
//! // Chinese Beijing Chinese (class: China)
//! // Chinese Chinese Shanghai (class: China)
//! // Chinese Macao (class: China)
//! // Tokyo Japan Chinese (class: Japan)
//! let x = DenseMatrix::<f64>::from_2d_array(&[
//!           &[1., 1., 0., 0., 0., 0.],
//!           &[0., 1., 0., 0., 1., 0.],
//!           &[0., 1., 0., 1., 0., 0.],
//!           &[0., 1., 1., 0., 0., 1.],
//! ]);
//! let y = vec![0., 0., 0., 1.];
//!
//! let nb = BernoulliNB::fit(&x, &y, Default::default()).unwrap();
//!
//! // Testing data point is:
//! // Chinese Chinese Chinese Tokyo Japan
//! let x_test = DenseMatrix::<f64>::from_2d_array(&[&[0., 1., 1., 0., 0., 1.]]);
//! let y_hat = nb.predict(&x_test).unwrap();
//! ```
//!
//! ## References:
//!
//! * ["Introduction to Information Retrieval", Manning C. D., Raghavan P., Schutze H., 2009, Chapter 13 ](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
use crate::base::Predictor;
use crate::error::Failed;
use crate::linalg::row_iter;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::math::vector::RealNumberVector;
use crate::naive_bayes::{BaseNaiveBayes, NBDistribution};

use serde::{Deserialize, Serialize};

/// Naive Bayes classifier for Bearnoulli features
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct BernoulliNBDistribution<T: RealNumber> {
    /// class labels known to the classifier
    class_labels: Vec<T>,
    class_priors: Vec<T>,
    feature_prob: Vec<Vec<T>>,
}

impl<T: RealNumber, M: Matrix<T>> NBDistribution<T, M> for BernoulliNBDistribution<T> {
    fn prior(&self, class_index: usize) -> T {
        self.class_priors[class_index]
    }

    fn log_likelihood(&self, class_index: usize, j: &M::RowVector) -> T {
        let mut likelihood = T::zero();
        for feature in 0..j.len() {
            let value = j.get(feature);
            if value == T::one() {
                likelihood += self.feature_prob[class_index][feature].ln();
            } else {
                likelihood += (T::one() - self.feature_prob[class_index][feature]).ln();
            }
        }
        likelihood
    }

    fn classes(&self) -> &Vec<T> {
        &self.class_labels
    }
}

/// `BernoulliNB` parameters. Use `Default::default()` for default values.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BernoulliNBParameters<T: RealNumber> {
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: T,
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Option<Vec<T>>,
    /// Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.
    pub binarize: Option<T>,
}

impl<T: RealNumber> BernoulliNBParameters<T> {
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub fn with_priors(mut self, priors: Vec<T>) -> Self {
        self.priors = Some(priors);
        self
    }
    /// Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.
    pub fn with_binarize(mut self, binarize: T) -> Self {
        self.binarize = Some(binarize);
        self
    }
}

impl<T: RealNumber> Default for BernoulliNBParameters<T> {
    fn default() -> Self {
        Self {
            alpha: T::one(),
            priors: None,
            binarize: Some(T::zero()),
        }
    }
}

impl<T: RealNumber> BernoulliNBDistribution<T> {
    /// Fits the distribution to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data.
    /// * `y` - vector with target values (classes) of length N.
    /// * `priors` - Optional vector with prior probabilities of the classes. If not defined,
    /// priors are adjusted according to the data.
    /// * `alpha` - Additive (Laplace/Lidstone) smoothing parameter.
    /// * `binarize` - Threshold for binarizing.
    pub fn fit<M: Matrix<T>>(
        x: &M,
        y: &M::RowVector,
        alpha: T,
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
        if alpha < T::zero() {
            return Err(Failed::fit(&format!(
                "Alpha should be greater than 0; |alpha|=[{}]",
                alpha
            )));
        }

        let y = y.to_vec();

        let (class_labels, indices) = <Vec<T> as RealNumberVector<T>>::unique_with_indices(&y);
        let mut class_count = vec![T::zero(); class_labels.len()];

        for class_index in indices.iter() {
            class_count[*class_index] += T::one();
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
                .map(|&c| c / T::from(n_samples).unwrap())
                .collect()
        };

        let mut feature_in_class_counter = vec![vec![T::zero(); n_features]; class_labels.len()];

        for (row, class_index) in row_iter(x).zip(indices) {
            for (idx, row_i) in row.iter().enumerate().take(n_features) {
                feature_in_class_counter[class_index][idx] += *row_i;
            }
        }

        let feature_prob = feature_in_class_counter
            .iter()
            .enumerate()
            .map(|(class_index, feature_count)| {
                feature_count
                    .iter()
                    .map(|&count| (count + alpha) / (class_count[class_index] + alpha * T::two()))
                    .collect()
            })
            .collect();

        Ok(Self {
            class_labels,
            class_priors,
            feature_prob,
        })
    }
}

/// BernoulliNB implements the categorical naive Bayes algorithm for categorically distributed data.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct BernoulliNB<T: RealNumber, M: Matrix<T>> {
    inner: BaseNaiveBayes<T, M, BernoulliNBDistribution<T>>,
    binarize: Option<T>,
}

impl<T: RealNumber, M: Matrix<T>> Predictor<M, M::RowVector> for BernoulliNB<T, M> {
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber, M: Matrix<T>> BernoulliNB<T, M> {
    /// Fits BernoulliNB with given data
    /// * `x` - training data of size NxM where N is the number of samples and M is the number of
    /// features.
    /// * `y` - vector with target values (classes) of length N.
    /// * `parameters` - additional parameters like class priors, alpha for smoothing and
    /// binarizing threshold.
    pub fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: BernoulliNBParameters<T>,
    ) -> Result<Self, Failed> {
        let distribution = if let Some(threshold) = parameters.binarize {
            BernoulliNBDistribution::fit(
                &(x.binarize(threshold)),
                y,
                parameters.alpha,
                parameters.priors,
            )?
        } else {
            BernoulliNBDistribution::fit(x, y, parameters.alpha, parameters.priors)?
        };

        let inner = BaseNaiveBayes::fit(distribution)?;
        Ok(Self {
            inner,
            binarize: parameters.binarize,
        })
    }

    /// Estimates the class labels for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    /// Returns a vector of size N with class estimates.
    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        if let Some(threshold) = self.binarize {
            self.inner.predict(&(x.binarize(threshold)))
        } else {
            self.inner.predict(x)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn run_bernoulli_naive_bayes() {
        // Tests that BernoulliNB when alpha=1.0 gives the same values as
        // those given for the toy example in Manning, Raghavan, and
        // Schuetze's "Introduction to Information Retrieval" book:
        // https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

        // Training data points are:
        // Chinese Beijing Chinese (class: China)
        // Chinese Chinese Shanghai (class: China)
        // Chinese Macao (class: China)
        // Tokyo Japan Chinese (class: Japan)
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1., 1., 0., 0., 0., 0.],
            &[0., 1., 0., 0., 1., 0.],
            &[0., 1., 0., 1., 0., 0.],
            &[0., 1., 1., 0., 0., 1.],
        ]);
        let y = vec![0., 0., 0., 1.];
        let bnb = BernoulliNB::fit(&x, &y, Default::default()).unwrap();

        assert_eq!(bnb.inner.distribution.class_priors, &[0.75, 0.25]);
        assert_eq!(
            bnb.inner.distribution.feature_prob,
            &[
                &[0.4, 0.8, 0.2, 0.4, 0.4, 0.2],
                &[1. / 3.0, 2. / 3.0, 2. / 3.0, 1. / 3.0, 1. / 3.0, 2. / 3.0]
            ]
        );

        // Testing data point is:
        //  Chinese Chinese Chinese Tokyo Japan
        let x_test = DenseMatrix::<f64>::from_2d_array(&[&[0., 1., 1., 0., 0., 1.]]);
        let y_hat = bnb.predict(&x_test).unwrap();

        assert_eq!(y_hat, &[1.]);
    }

    #[test]
    fn bernoulli_nb_scikit_parity() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[2., 4., 0., 0., 2., 1., 2., 4., 2., 0.],
            &[3., 4., 0., 2., 1., 0., 1., 4., 0., 3.],
            &[1., 4., 2., 4., 1., 0., 1., 2., 3., 2.],
            &[0., 3., 3., 4., 1., 0., 3., 1., 1., 1.],
            &[0., 2., 1., 4., 3., 4., 1., 2., 3., 1.],
            &[3., 2., 4., 1., 3., 0., 2., 4., 0., 2.],
            &[3., 1., 3., 0., 2., 0., 4., 4., 3., 4.],
            &[2., 2., 2., 0., 1., 1., 2., 1., 0., 1.],
            &[3., 3., 2., 2., 0., 2., 3., 2., 2., 3.],
            &[4., 3., 4., 4., 4., 2., 2., 0., 1., 4.],
            &[3., 4., 2., 2., 1., 4., 4., 4., 1., 3.],
            &[3., 0., 1., 4., 4., 0., 0., 3., 2., 4.],
            &[2., 0., 3., 3., 1., 2., 0., 2., 4., 1.],
            &[2., 4., 0., 4., 2., 4., 1., 3., 1., 4.],
            &[0., 2., 2., 3., 4., 0., 4., 4., 4., 4.],
        ]);
        let y = vec![2., 2., 0., 0., 0., 2., 1., 1., 0., 1., 0., 0., 2., 0., 2.];
        let bnb = BernoulliNB::fit(&x, &y, Default::default()).unwrap();

        let y_hat = bnb.predict(&x).unwrap();

        assert!(bnb
            .inner
            .distribution
            .class_priors
            .approximate_eq(&vec!(0.46, 0.2, 0.33), 1e-2));
        assert!(bnb.inner.distribution.feature_prob[1].approximate_eq(
            &vec!(0.8, 0.8, 0.8, 0.4, 0.8, 0.6, 0.8, 0.6, 0.6, 0.8),
            1e-1
        ));
        assert!(y_hat.approximate_eq(
            &vec!(2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            1e-5
        ));
    }

    #[test]
    fn serde() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1., 1., 0., 0., 0., 0.],
            &[0., 1., 0., 0., 1., 0.],
            &[0., 1., 0., 1., 0., 0.],
            &[0., 1., 1., 0., 0., 1.],
        ]);
        let y = vec![0., 0., 0., 1.];

        let bnb = BernoulliNB::fit(&x, &y, Default::default()).unwrap();
        let deserialized_bnb: BernoulliNB<f64, DenseMatrix<f64>> =
            serde_json::from_str(&serde_json::to_string(&bnb).unwrap()).unwrap();

        assert_eq!(bnb, deserialized_bnb);
    }
}
