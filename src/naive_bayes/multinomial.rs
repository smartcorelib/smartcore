//! # Multinomial Naive Bayes
//!
//! Multinomial Naive Bayes classifier is a variant of [Naive Bayes](../index.html) for the multinomially distributed data.
//! It is often used for discrete data with predictors representing the number of times an event was observed in a particular instance,
//! for example frequency of the words present in the document.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::naive_bayes::multinomial::MultinomialNB;
//!
//! // Training data points are:
//! // Chinese Beijing Chinese (class: China)
//! // Chinese Chinese Shanghai (class: China)
//! // Chinese Macao (class: China)
//! // Tokyo Japan Chinese (class: Japan)
//! let x = DenseMatrix::<u32>::from_2d_array(&[
//!                   &[1, 2, 0, 0, 0, 0],
//!                   &[0, 2, 0, 0, 1, 0],
//!                   &[0, 1, 0, 1, 0, 0],
//!                   &[0, 1, 1, 0, 0, 1],
//!         ]);
//! let y: Vec<u32> = vec![0, 0, 0, 1];
//! let nb = MultinomialNB::fit(&x, &y, Default::default()).unwrap();
//!
//! // Testing data point is:
//! //  Chinese Chinese Chinese Tokyo Japan
//! let x_test = DenseMatrix::from_2d_array(&[&[0, 3, 1, 0, 0, 1]]);
//! let y_hat = nb.predict(&x_test).unwrap();
//! ```
//!
//! ## References:
//!
//! * ["Introduction to Information Retrieval", Manning C. D., Raghavan P., Schutze H., 2009, Chapter 13 ](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
use std::fmt;

use num_traits::Unsigned;

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2, ArrayView1};
use crate::naive_bayes::{BaseNaiveBayes, NBDistribution};
use crate::numbers::basenum::Number;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Naive Bayes classifier for Multinomial features
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq, Clone)]
struct MultinomialNBDistribution<T: Number> {
    /// class labels known to the classifier
    class_labels: Vec<T>,
    /// number of training samples observed in each class
    class_count: Vec<usize>,
    /// probability of each class
    class_priors: Vec<f64>,
    /// Empirical log probability of features given a class
    feature_log_prob: Vec<Vec<f64>>,
    /// Number of samples encountered for each (class, feature)
    feature_count: Vec<Vec<usize>>,
    /// Number of features of each sample
    n_features: usize,
}

impl<T: Number + Ord + Unsigned> fmt::Display for MultinomialNBDistribution<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "MultinomialNBDistribution: n_features: {:?}",
            self.n_features
        )?;
        writeln!(f, "class_labels: {:?}", self.class_labels)?;
        Ok(())
    }
}

impl<X: Number + Unsigned, Y: Number + Ord + Unsigned> NBDistribution<X, Y>
    for MultinomialNBDistribution<Y>
{
    fn prior(&self, class_index: usize) -> f64 {
        self.class_priors[class_index]
    }

    fn log_likelihood<'a>(&self, class_index: usize, j: &'a Box<dyn ArrayView1<X> + 'a>) -> f64 {
        let mut likelihood = 0f64;
        for feature in 0..j.shape() {
            let value = X::to_f64(j.get(feature)).unwrap();
            likelihood += value * self.feature_log_prob[class_index][feature];
        }
        likelihood
    }

    fn classes(&self) -> &Vec<Y> {
        &self.class_labels
    }
}

/// `MultinomialNB` parameters. Use `Default::default()` for default values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MultinomialNBParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: f64,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Option<Vec<f64>>,
}

impl MultinomialNBParameters {
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub fn with_priors(mut self, priors: Vec<f64>) -> Self {
        self.priors = Some(priors);
        self
    }
}

impl Default for MultinomialNBParameters {
    fn default() -> Self {
        Self {
            alpha: 1f64,
            priors: Option::None,
        }
    }
}

/// MultinomialNB grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MultinomialNBSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Vec<Option<Vec<f64>>>,
}

/// MultinomialNB grid search iterator
pub struct MultinomialNBSearchParametersIterator {
    multinomial_nb_search_parameters: MultinomialNBSearchParameters,
    current_alpha: usize,
    current_priors: usize,
}

impl IntoIterator for MultinomialNBSearchParameters {
    type Item = MultinomialNBParameters;
    type IntoIter = MultinomialNBSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        MultinomialNBSearchParametersIterator {
            multinomial_nb_search_parameters: self,
            current_alpha: 0,
            current_priors: 0,
        }
    }
}

impl Iterator for MultinomialNBSearchParametersIterator {
    type Item = MultinomialNBParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_alpha == self.multinomial_nb_search_parameters.alpha.len()
            && self.current_priors == self.multinomial_nb_search_parameters.priors.len()
        {
            return None;
        }

        let next = MultinomialNBParameters {
            alpha: self.multinomial_nb_search_parameters.alpha[self.current_alpha],
            priors: self.multinomial_nb_search_parameters.priors[self.current_priors].clone(),
        };

        if self.current_alpha + 1 < self.multinomial_nb_search_parameters.alpha.len() {
            self.current_alpha += 1;
        } else if self.current_priors + 1 < self.multinomial_nb_search_parameters.priors.len() {
            self.current_alpha = 0;
            self.current_priors += 1;
        } else {
            self.current_alpha += 1;
            self.current_priors += 1;
        }

        Some(next)
    }
}

impl Default for MultinomialNBSearchParameters {
    fn default() -> Self {
        let default_params = MultinomialNBParameters::default();

        MultinomialNBSearchParameters {
            alpha: vec![default_params.alpha],
            priors: vec![default_params.priors],
        }
    }
}

impl<TY: Number + Ord + Unsigned> MultinomialNBDistribution<TY> {
    /// Fits the distribution to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data.
    /// * `y` - vector with target values (classes) of length N.
    /// * `priors` - Optional vector with prior probabilities of the classes. If not defined,
    /// priors are adjusted according to the data.
    /// * `alpha` - Additive (Laplace/Lidstone) smoothing parameter.
    pub fn fit<TX: Number + Unsigned, X: Array2<TX>, Y: Array1<TY>>(
        x: &X,
        y: &Y,
        alpha: f64,
        priors: Option<Vec<f64>>,
    ) -> Result<Self, Failed> {
        let (n_samples, n_features) = x.shape();
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
        if alpha < 0f64 {
            return Err(Failed::fit(&format!(
                "Alpha should be greater than 0; |alpha|=[{}]",
                alpha
            )));
        }

        let (class_labels, indices) = y.unique_with_indices();
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
                .map(|&c| c as f64 / n_samples as f64)
                .collect()
        };

        let mut feature_in_class_counter = vec![vec![0_usize; n_features]; class_labels.len()];

        for (row, class_index) in x.row_iter().zip(indices) {
            for (idx, row_i) in row.iterator(0).enumerate().take(n_features) {
                feature_in_class_counter[class_index][idx] +=
                    row_i.to_usize().ok_or_else(|| {
                        Failed::fit(&format!(
                            "Elements of the matrix should be convertible to usize |found|=[{}]",
                            row_i
                        ))
                    })?;
            }
        }

        let feature_log_prob = feature_in_class_counter
            .iter()
            .map(|feature_count| {
                let n_c: usize = feature_count.iter().sum();
                feature_count
                    .iter()
                    .map(|&count| {
                        ((count as f64 + alpha) / (n_c as f64 + alpha * n_features as f64)).ln()
                    })
                    .collect()
            })
            .collect();

        Ok(Self {
            class_count,
            class_labels,
            class_priors,
            feature_log_prob,
            feature_count: feature_in_class_counter,
            n_features,
        })
    }
}

/// MultinomialNB implements the naive Bayes algorithm for multinomially distributed data.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub struct MultinomialNB<
    TX: Number + Unsigned,
    TY: Number + Ord + Unsigned,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    inner: Option<BaseNaiveBayes<TX, TY, X, Y, MultinomialNBDistribution<TY>>>,
}

impl<TX: Number + Unsigned, TY: Number + Ord + Unsigned, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, MultinomialNBParameters> for MultinomialNB<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            inner: Option::None,
        }
    }

    fn fit(x: &X, y: &Y, parameters: MultinomialNBParameters) -> Result<Self, Failed> {
        MultinomialNB::fit(x, y, parameters)
    }
}

impl<TX: Number + Unsigned, TY: Number + Ord + Unsigned, X: Array2<TX>, Y: Array1<TY>>
    Predictor<X, Y> for MultinomialNB<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number + Unsigned, TY: Number + Ord + Unsigned, X: Array2<TX>, Y: Array1<TY>>
    MultinomialNB<TX, TY, X, Y>
{
    /// Fits MultinomialNB with given data
    /// * `x` - training data of size NxM where N is the number of samples and M is the number of
    /// features.
    /// * `y` - vector with target values (classes) of length N.
    /// * `parameters` - additional parameters like class priors, alpha for smoothing and
    /// binarizing threshold.
    pub fn fit(x: &X, y: &Y, parameters: MultinomialNBParameters) -> Result<Self, Failed> {
        let distribution =
            MultinomialNBDistribution::fit(x, y, parameters.alpha, parameters.priors)?;
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

    /// Empirical log probability of features given a class, P(x_i|y).
    /// Returns a 2d vector of shape (n_classes, n_features)
    pub fn feature_log_prob(&self) -> &Vec<Vec<f64>> {
        &self.inner.as_ref().unwrap().distribution.feature_log_prob
    }

    /// Number of features of each sample
    pub fn n_features(&self) -> usize {
        self.inner.as_ref().unwrap().distribution.n_features
    }

    /// Number of samples encountered for each (class, feature)
    /// Returns a 2d vector of shape (n_classes, n_features)
    pub fn feature_count(&self) -> &Vec<Vec<usize>> {
        &self.inner.as_ref().unwrap().distribution.feature_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn search_parameters() {
        let parameters = MultinomialNBSearchParameters {
            alpha: vec![1., 2.],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.alpha, 1.);
        let next = iter.next().unwrap();
        assert_eq!(next.alpha, 2.);
        assert!(iter.next().is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let x = DenseMatrix::from_2d_array(&[
            &[1, 2, 0, 0, 0, 0],
            &[0, 2, 0, 0, 1, 0],
            &[0, 1, 0, 1, 0, 0],
            &[0, 1, 1, 0, 0, 1],
        ]);
        let y: Vec<u32> = vec![0, 0, 0, 1];
        let mnb = MultinomialNB::fit(&x, &y, Default::default()).unwrap();

        assert_eq!(mnb.classes(), &[0, 1]);
        assert_eq!(mnb.class_count(), &[3, 1]);

        let distribution = mnb.inner.clone().unwrap().distribution;

        assert_eq!(&distribution.class_priors, &[0.75, 0.25]);
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
        let x_test = DenseMatrix::<u32>::from_2d_array(&[&[0, 3, 1, 0, 0, 1]]);
        let y_hat = mnb.predict(&x_test).unwrap();

        assert_eq!(y_hat, &[0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn multinomial_nb_scikit_parity() {
        let x = DenseMatrix::<u32>::from_2d_array(&[
            &[2, 4, 0, 0, 2, 1, 2, 4, 2, 0],
            &[3, 4, 0, 2, 1, 0, 1, 4, 0, 3],
            &[1, 4, 2, 4, 1, 0, 1, 2, 3, 2],
            &[0, 3, 3, 4, 1, 0, 3, 1, 1, 1],
            &[0, 2, 1, 4, 3, 4, 1, 2, 3, 1],
            &[3, 2, 4, 1, 3, 0, 2, 4, 0, 2],
            &[3, 1, 3, 0, 2, 0, 4, 4, 3, 4],
            &[2, 2, 2, 0, 1, 1, 2, 1, 0, 1],
            &[3, 3, 2, 2, 0, 2, 3, 2, 2, 3],
            &[4, 3, 4, 4, 4, 2, 2, 0, 1, 4],
            &[3, 4, 2, 2, 1, 4, 4, 4, 1, 3],
            &[3, 0, 1, 4, 4, 0, 0, 3, 2, 4],
            &[2, 0, 3, 3, 1, 2, 0, 2, 4, 1],
            &[2, 4, 0, 4, 2, 4, 1, 3, 1, 4],
            &[0, 2, 2, 3, 4, 0, 4, 4, 4, 4],
        ]);
        let y: Vec<u32> = vec![2, 2, 0, 0, 0, 2, 1, 1, 0, 1, 0, 0, 2, 0, 2];
        let nb = MultinomialNB::fit(&x, &y, Default::default()).unwrap();

        assert_eq!(nb.n_features(), 10);
        assert_eq!(
            nb.feature_count(),
            &[
                &[12, 20, 11, 24, 12, 14, 13, 17, 13, 18],
                &[9, 6, 9, 4, 7, 3, 8, 5, 4, 9],
                &[10, 12, 9, 9, 11, 3, 9, 18, 10, 10]
            ]
        );

        let y_hat = nb.predict(&x).unwrap();

        let distribution = nb.inner.clone().unwrap().distribution;

        assert_eq!(
            &distribution.class_priors,
            &vec!(0.4666666666666667, 0.2, 0.3333333333333333)
        );

        // Due to float differences in WASM32,
        // we disable this test for that arch
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(
            &nb.feature_log_prob()[1],
            &vec![
                -2.001480000210124,
                -2.3581549441488563,
                -2.001480000210124,
                -2.6946271807700692,
                -2.2246235515243336,
                -2.917770732084279,
                -2.10684051586795,
                -2.512305623976115,
                -2.6946271807700692,
                -2.001480000210124
            ]
        );
        assert_eq!(y_hat, vec!(2, 2, 0, 0, 0, 2, 2, 1, 0, 1, 0, 2, 0, 0, 2));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::from_2d_array(&[
            &[1, 1, 0, 0, 0, 0],
            &[0, 1, 0, 0, 1, 0],
            &[0, 1, 0, 1, 0, 0],
            &[0, 1, 1, 0, 0, 1],
        ]);
        let y = vec![0, 0, 0, 1];

        let mnb = MultinomialNB::fit(&x, &y, Default::default()).unwrap();
        let deserialized_mnb: MultinomialNB<u32, u32, DenseMatrix<u32>, Vec<u32>> =
            serde_json::from_str(&serde_json::to_string(&mnb).unwrap()).unwrap();

        assert_eq!(mnb, deserialized_mnb);
    }
}
