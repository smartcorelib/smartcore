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

/// Naive Bayes classifier for Bearnoulli features
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
struct BernoulliNBDistribution<T: RealNumber> {
    /// class labels known to the classifier
    class_labels: Vec<T>,
    /// number of training samples observed in each class
    class_count: Vec<usize>,
    /// probability of each class
    class_priors: Vec<T>,
    /// Number of samples encountered for each (class, feature)
    feature_count: Vec<Vec<usize>>,
    /// probability of features per class
    feature_log_prob: Vec<Vec<T>>,
    /// Number of features of each sample
    n_features: usize,
}

impl<T: RealNumber> PartialEq for BernoulliNBDistribution<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.class_labels == other.class_labels
            && self.class_count == other.class_count
            && self.class_priors == other.class_priors
            && self.feature_count == other.feature_count
            && self.n_features == other.n_features
        {
            for (a, b) in self
                .feature_log_prob
                .iter()
                .zip(other.feature_log_prob.iter())
            {
                if !a.approximate_eq(b, T::epsilon()) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
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
                likelihood += self.feature_log_prob[class_index][feature];
            } else {
                likelihood += (T::one() - self.feature_log_prob[class_index][feature].exp()).ln();
            }
        }
        likelihood
    }

    fn classes(&self) -> &Vec<T> {
        &self.class_labels
    }
}

/// `BernoulliNB` parameters. Use `Default::default()` for default values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct BernoulliNBParameters<T: RealNumber> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: T,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Option<Vec<T>>,
    #[cfg_attr(feature = "serde", serde(default))]
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

/// BernoulliNB grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct BernoulliNBSearchParameters<T: RealNumber> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: Vec<T>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Prior probabilities of the classes. If specified the priors are not adjusted according to the data
    pub priors: Vec<Option<Vec<T>>>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.
    pub binarize: Vec<Option<T>>,
}

/// BernoulliNB grid search iterator
pub struct BernoulliNBSearchParametersIterator<T: RealNumber> {
    bernoulli_nb_search_parameters: BernoulliNBSearchParameters<T>,
    current_alpha: usize,
    current_priors: usize,
    current_binarize: usize,
}

impl<T: RealNumber> IntoIterator for BernoulliNBSearchParameters<T> {
    type Item = BernoulliNBParameters<T>;
    type IntoIter = BernoulliNBSearchParametersIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        BernoulliNBSearchParametersIterator {
            bernoulli_nb_search_parameters: self,
            current_alpha: 0,
            current_priors: 0,
            current_binarize: 0,
        }
    }
}

impl<T: RealNumber> Iterator for BernoulliNBSearchParametersIterator<T> {
    type Item = BernoulliNBParameters<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_alpha == self.bernoulli_nb_search_parameters.alpha.len()
            && self.current_priors == self.bernoulli_nb_search_parameters.priors.len()
            && self.current_binarize == self.bernoulli_nb_search_parameters.binarize.len()
        {
            return None;
        }

        let next = BernoulliNBParameters {
            alpha: self.bernoulli_nb_search_parameters.alpha[self.current_alpha],
            priors: self.bernoulli_nb_search_parameters.priors[self.current_priors].clone(),
            binarize: self.bernoulli_nb_search_parameters.binarize[self.current_binarize],
        };

        if self.current_alpha + 1 < self.bernoulli_nb_search_parameters.alpha.len() {
            self.current_alpha += 1;
        } else if self.current_priors + 1 < self.bernoulli_nb_search_parameters.priors.len() {
            self.current_alpha = 0;
            self.current_priors += 1;
        } else if self.current_binarize + 1 < self.bernoulli_nb_search_parameters.binarize.len() {
            self.current_alpha = 0;
            self.current_priors = 0;
            self.current_binarize += 1;
        } else {
            self.current_alpha += 1;
            self.current_priors += 1;
            self.current_binarize += 1;
        }

        Some(next)
    }
}

impl<T: RealNumber> Default for BernoulliNBSearchParameters<T> {
    fn default() -> Self {
        let default_params = BernoulliNBParameters::default();

        BernoulliNBSearchParameters {
            alpha: vec![default_params.alpha],
            priors: vec![default_params.priors],
            binarize: vec![default_params.binarize],
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

        let mut feature_in_class_counter = vec![vec![0_usize; n_features]; class_labels.len()];

        for (row, class_index) in row_iter(x).zip(indices) {
            for (idx, row_i) in row.iter().enumerate().take(n_features) {
                feature_in_class_counter[class_index][idx] +=
                    row_i.to_usize().ok_or_else(|| {
                        Failed::fit(&format!(
                            "Elements of the matrix should be 1.0 or 0.0 |found|=[{}]",
                            row_i
                        ))
                    })?;
            }
        }

        let feature_log_prob = feature_in_class_counter
            .iter()
            .enumerate()
            .map(|(class_index, feature_count)| {
                feature_count
                    .iter()
                    .map(|&count| {
                        ((T::from(count).unwrap() + alpha)
                            / (T::from(class_count[class_index]).unwrap() + alpha * T::two()))
                        .ln()
                    })
                    .collect()
            })
            .collect();

        Ok(Self {
            class_labels,
            class_priors,
            class_count,
            feature_count: feature_in_class_counter,
            feature_log_prob,
            n_features,
        })
    }
}

/// BernoulliNB implements the naive Bayes algorithm for data that follows the Bernoulli
/// distribution.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub struct BernoulliNB<T: RealNumber, M: Matrix<T>> {
    inner: BaseNaiveBayes<T, M, BernoulliNBDistribution<T>>,
    binarize: Option<T>,
}

impl<T: RealNumber, M: Matrix<T>> SupervisedEstimator<M, M::RowVector, BernoulliNBParameters<T>>
    for BernoulliNB<T, M>
{
    fn fit(x: &M, y: &M::RowVector, parameters: BernoulliNBParameters<T>) -> Result<Self, Failed> {
        BernoulliNB::fit(x, y, parameters)
    }
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

    /// Number of features of each sample
    pub fn n_features(&self) -> usize {
        self.inner.distribution.n_features
    }

    /// Number of samples encountered for each (class, feature)
    /// Returns a 2d vector of shape (n_classes, n_features)
    pub fn feature_count(&self) -> &Vec<Vec<usize>> {
        &self.inner.distribution.feature_count
    }

    /// Empirical log probability of features given a class
    pub fn feature_log_prob(&self) -> &Vec<Vec<T>> {
        &self.inner.distribution.feature_log_prob
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn search_parameters() {
        let parameters = BernoulliNBSearchParameters {
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
            bnb.feature_log_prob(),
            &[
                &[
                    -0.916290731874155,
                    -0.2231435513142097,
                    -1.6094379124341003,
                    -0.916290731874155,
                    -0.916290731874155,
                    -1.6094379124341003
                ],
                &[
                    -1.0986122886681098,
                    -0.40546510810816444,
                    -0.40546510810816444,
                    -1.0986122886681098,
                    -1.0986122886681098,
                    -0.40546510810816444
                ]
            ]
        );

        // Testing data point is:
        //  Chinese Chinese Chinese Tokyo Japan
        let x_test = DenseMatrix::<f64>::from_2d_array(&[&[0., 1., 1., 0., 0., 1.]]);
        let y_hat = bnb.predict(&x_test).unwrap();

        assert_eq!(y_hat, &[1.]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

        assert_eq!(bnb.classes(), &[0., 1., 2.]);
        assert_eq!(bnb.class_count(), &[7, 3, 5]);
        assert_eq!(bnb.n_features(), 10);
        assert_eq!(
            bnb.feature_count(),
            &[
                &[5, 6, 6, 7, 6, 4, 6, 7, 7, 7],
                &[3, 3, 3, 1, 3, 2, 3, 2, 2, 3],
                &[4, 4, 3, 4, 5, 2, 4, 5, 3, 4]
            ]
        );

        assert!(bnb
            .inner
            .distribution
            .class_priors
            .approximate_eq(&vec!(0.46, 0.2, 0.33), 1e-2));
        assert!(bnb.feature_log_prob()[1].approximate_eq(
            &vec![
                -0.22314355,
                -0.22314355,
                -0.22314355,
                -0.91629073,
                -0.22314355,
                -0.51082562,
                -0.22314355,
                -0.51082562,
                -0.51082562,
                -0.22314355
            ],
            1e-1
        ));
        assert!(y_hat.approximate_eq(
            &vec!(2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            1e-5
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

        let bnb = BernoulliNB::fit(&x, &y, Default::default()).unwrap();
        let deserialized_bnb: BernoulliNB<f64, DenseMatrix<f64>> =
            serde_json::from_str(&serde_json::to_string(&bnb).unwrap()).unwrap();

        assert_eq!(bnb, deserialized_bnb);
    }
}
