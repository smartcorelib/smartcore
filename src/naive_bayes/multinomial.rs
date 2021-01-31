//! # Multinomial Naive Bayes
//!
//! Multinomial Naive Bayes classifier is a variant of [Naive Bayes](../index.html) for the multinomially distributed data.
//! It is often used for discrete data with predictors representing the number of times an event was observed in a particular instance,
//! for example frequency of the words present in the document.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::naive_bayes::multinomial::MultinomialNB;
//!
//! // Training data points are:
//! // Chinese Beijing Chinese (class: China)
//! // Chinese Chinese Shanghai (class: China)
//! // Chinese Macao (class: China)
//! // Tokyo Japan Chinese (class: Japan)
//! let x = DenseMatrix::<f64>::from_2d_array(&[
//!                   &[1., 2., 0., 0., 0., 0.],
//!                   &[0., 2., 0., 0., 1., 0.],
//!                   &[0., 1., 0., 1., 0., 0.],
//!                   &[0., 1., 1., 0., 0., 1.],
//!         ]);
//! let y = vec![0., 0., 0., 1.];
//! let nb = MultinomialNB::fit(&x, &y, Default::default()).unwrap();
//!
//! // Testing data point is:
//! //  Chinese Chinese Chinese Tokyo Japan
//! let x_test = DenseMatrix::<f64>::from_2d_array(&[&[0., 3., 1., 0., 0., 1.]]);
//! let y_hat = nb.predict(&x_test).unwrap();
//! ```
//!
//! ## References:
//!
//! * ["Introduction to Information Retrieval", Manning C. D., Raghavan P., Schutze H., 2009, Chapter 13 ](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
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

/// Naive Bayes classifier for Multinomial features
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
struct MultinomialNBDistribution<T: RealNumber> {
    /// class labels known to the classifier
    class_labels: Vec<T>,
    class_count: Vec<usize>,
    class_priors: Vec<T>,
    /// Empirical log probability of features given a class
    feature_log_prob: Vec<Vec<T>>,
}

impl<T: RealNumber, M: Matrix<T>> NBDistribution<T, M> for MultinomialNBDistribution<T> {
    fn prior(&self, class_index: usize) -> T {
        self.class_priors[class_index]
    }

    fn log_likelihood(&self, class_index: usize, j: &M::RowVector) -> T {
        let mut likelihood = T::zero();
        for feature in 0..j.len() {
            let value = j.get(feature);
            likelihood += value * self.feature_log_prob[class_index][feature];
        }
        likelihood
    }

    fn classes(&self) -> &Vec<T> {
        &self.class_labels
    }
}

/// `MultinomialNB` parameters. Use `Default::default()` for default values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MultinomialNBParameters<T: RealNumber> {
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: T,
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Option<Vec<T>>,
}

impl<T: RealNumber> MultinomialNBParameters<T> {
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
}

impl<T: RealNumber> Default for MultinomialNBParameters<T> {
    fn default() -> Self {
        Self {
            alpha: T::one(),
            priors: None,
        }
    }
}

impl<T: RealNumber> MultinomialNBDistribution<T> {
    /// Fits the distribution to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data.
    /// * `y` - vector with target values (classes) of length N.
    /// * `priors` - Optional vector with prior probabilities of the classes. If not defined,
    /// priors are adjusted according to the data.
    /// * `alpha` - Additive (Laplace/Lidstone) smoothing parameter.
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
        let mut class_count = vec![0_usize; class_labels.len()];

        for class_index in indices.iter() {
            class_count[*class_index] += 1;
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

        let mut feature_in_class_counter = vec![vec![T::zero(); n_features]; class_labels.len()];

        for (row, class_index) in row_iter(x).zip(indices) {
            for (idx, row_i) in row.iter().enumerate().take(n_features) {
                feature_in_class_counter[class_index][idx] += *row_i;
            }
        }

        let feature_log_prob = feature_in_class_counter
            .iter()
            .map(|feature_count| {
                let n_c = feature_count.sum();
                feature_count
                    .iter()
                    .map(|&count| {
                        ((count + alpha) / (n_c + alpha * T::from(n_features).unwrap())).ln()
                    })
                    .collect()
            })
            .collect();

        Ok(Self {
            class_count,
            class_labels,
            class_priors,
            feature_log_prob,
        })
    }
}

/// MultinomialNB implements the categorical naive Bayes algorithm for categorically distributed data.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub struct MultinomialNB<T: RealNumber, M: Matrix<T>> {
    inner: BaseNaiveBayes<T, M, MultinomialNBDistribution<T>>,
}

impl<T: RealNumber, M: Matrix<T>> SupervisedEstimator<M, M::RowVector, MultinomialNBParameters<T>>
    for MultinomialNB<T, M>
{
    fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: MultinomialNBParameters<T>,
    ) -> Result<Self, Failed> {
        MultinomialNB::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>> Predictor<M, M::RowVector> for MultinomialNB<T, M> {
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber, M: Matrix<T>> MultinomialNB<T, M> {
    /// Fits MultinomialNB with given data
    /// * `x` - training data of size NxM where N is the number of samples and M is the number of
    /// features.
    /// * `y` - vector with target values (classes) of length N.
    /// * `parameters` - additional parameters like class priors, alpha for smoothing and
    /// binarizing threshold.
    pub fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: MultinomialNBParameters<T>,
    ) -> Result<Self, Failed> {
        let distribution =
            MultinomialNBDistribution::fit(x, y, parameters.alpha, parameters.priors)?;
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

    /// Empirical log probability of features given a class, P(x_i|y).
    /// Returns a 2d vector of shape (n_classes, n_features)
    pub fn feature_log_prob(&self) -> &Vec<Vec<T>> {
        &self.inner.distribution.feature_log_prob
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn run_multinomial_naive_bayes() {
        // Tests that MultinomialNB when alpha=1.0 gives the same values as
        // those given for the toy example in Manning, Raghavan, and
        // Schuetze's "Introduction to Information Retrieval" book:
        // https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

        // Training data points are:
        // Chinese Beijing Chinese (class: China)
        // Chinese Chinese Shanghai (class: China)
        // Chinese Macao (class: China)
        // Tokyo Japan Chinese (class: Japan)
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1., 2., 0., 0., 0., 0.],
            &[0., 2., 0., 0., 1., 0.],
            &[0., 1., 0., 1., 0., 0.],
            &[0., 1., 1., 0., 0., 1.],
        ]);
        let y = vec![0., 0., 0., 1.];
        let mnb = MultinomialNB::fit(&x, &y, Default::default()).unwrap();

        assert_eq!(mnb.classes(), &[0., 1.]);
        assert_eq!(mnb.class_count(), &[3, 1]);

        assert_eq!(mnb.inner.distribution.class_priors, &[0.75, 0.25]);
        assert_eq!(
            mnb.feature_log_prob(),
            &[
                &[
                    (1_f64 / 7_f64).ln(),
                    (3_f64 / 7_f64).ln(),
                    (1_f64 / 14_f64).ln(),
                    (1_f64 / 7_f64).ln(),
                    (1_f64 / 7_f64).ln(),
                    (1_f64 / 14_f64).ln()
                ],
                &[
                    (1_f64 / 9_f64).ln(),
                    (2_f64 / 9_f64).ln(),
                    (2_f64 / 9_f64).ln(),
                    (1_f64 / 9_f64).ln(),
                    (1_f64 / 9_f64).ln(),
                    (2_f64 / 9_f64).ln()
                ]
            ]
        );

        // Testing data point is:
        //  Chinese Chinese Chinese Tokyo Japan
        let x_test = DenseMatrix::<f64>::from_2d_array(&[&[0., 3., 1., 0., 0., 1.]]);
        let y_hat = mnb.predict(&x_test).unwrap();

        assert_eq!(y_hat, &[0.]);
    }

    #[test]
    fn multinomial_nb_scikit_parity() {
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
        let nb = MultinomialNB::fit(&x, &y, Default::default()).unwrap();

        let y_hat = nb.predict(&x).unwrap();

        assert!(nb
            .inner
            .distribution
            .class_priors
            .approximate_eq(&vec!(0.46, 0.2, 0.33), 1e-2));
        assert!(nb.feature_log_prob()[1].approximate_eq(
            &vec![
                -2.00148,
                -2.35815494,
                -2.00148,
                -2.69462718,
                -2.22462355,
                -2.91777073,
                -2.10684052,
                -2.51230562,
                -2.69462718,
                -2.00148
            ],
            1e-5
        ));
        assert!(y_hat.approximate_eq(
            &vec!(2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0),
            1e-5
        ));
    }
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1., 1., 0., 0., 0., 0.],
            &[0., 1., 0., 0., 1., 0.],
            &[0., 1., 0., 1., 0., 0.],
            &[0., 1., 1., 0., 0., 1.],
        ]);
        let y = vec![0., 0., 0., 1.];

        let mnb = MultinomialNB::fit(&x, &y, Default::default()).unwrap();
        let deserialized_mnb: MultinomialNB<f64, DenseMatrix<f64>> =
            serde_json::from_str(&serde_json::to_string(&mnb).unwrap()).unwrap();

        assert_eq!(mnb, deserialized_mnb);
    }
}
