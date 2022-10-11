//! # Categorical Naive Bayes
//!
//! Categorical Naive Bayes is a variant of [Naive Bayes](../index.html) for the categorically distributed data.
//! It assumes that each feature has its own categorical distribution.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::naive_bayes::categorical::CategoricalNB;
//!
//! let x = DenseMatrix::from_2d_array(&[
//!              &[3, 4, 0, 1],
//!              &[3, 0, 0, 1],
//!              &[4, 4, 1, 2],
//!              &[4, 2, 4, 3],
//!              &[4, 2, 4, 2],
//!              &[4, 1, 1, 0],
//!              &[1, 1, 1, 1],
//!              &[0, 4, 1, 0],
//!              &[0, 3, 2, 1],
//!              &[0, 3, 1, 1],
//!              &[3, 4, 0, 1],
//!              &[3, 4, 2, 4],
//!              &[0, 3, 1, 2],
//!              &[0, 4, 1, 2],
//!          ]);
//! let y: Vec<u32> = vec![0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0];
//!
//! let nb = CategoricalNB::fit(&x, &y, Default::default()).unwrap();
//! let y_hat = nb.predict(&x).unwrap();
//! ```
use num_traits::Unsigned;

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::base::{Array1, Array2, ArrayView1};
use crate::naive_bayes::{BaseNaiveBayes, NBDistribution};
use crate::numbers::basenum::Number;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Naive Bayes classifier for categorical features
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
struct CategoricalNBDistribution<T: Number + Unsigned> {
    /// number of training samples observed in each class
    class_count: Vec<usize>,
    /// class labels known to the classifier
    class_labels: Vec<T>,
    /// probability of each class
    class_priors: Vec<f64>,
    coefficients: Vec<Vec<Vec<f64>>>,
    /// Number of features of each sample
    n_features: usize,
    /// Number of categories for each feature
    n_categories: Vec<usize>,
    /// Holds arrays of shape (n_classes, n_categories of respective feature)
    /// for each feature. Each array provides the number of samples
    /// encountered for each class and category of the specific feature.
    category_count: Vec<Vec<Vec<usize>>>,
}

impl<T: Number + Unsigned> PartialEq for CategoricalNBDistribution<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.class_labels == other.class_labels
            && self.class_priors == other.class_priors
            && self.n_features == other.n_features
            && self.n_categories == other.n_categories
            && self.class_count == other.class_count
        {
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
                        if (*a_i_j - *b_i_j).abs() > std::f64::EPSILON {
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

impl<T: Number + Unsigned> NBDistribution<T, T> for CategoricalNBDistribution<T> {
    fn prior(&self, class_index: usize) -> f64 {
        if class_index >= self.class_labels.len() {
            0f64
        } else {
            self.class_priors[class_index]
        }
    }

    fn log_likelihood<'a>(&'a self, class_index: usize, j: &'a Box<dyn ArrayView1<T> + 'a>) -> f64 {
        if class_index < self.class_labels.len() {
            let mut likelihood = 0f64;
            for feature in 0..j.shape() {
                let value = j.get(feature).to_usize().unwrap();
                if self.coefficients[feature][class_index].len() > value {
                    likelihood += self.coefficients[feature][class_index][value];
                } else {
                    return 0f64;
                }
            }
            likelihood
        } else {
            0f64
        }
    }

    fn classes(&self) -> &Vec<T> {
        &self.class_labels
    }
}

impl<T: Number + Unsigned> CategoricalNBDistribution<T> {
    /// Fits the distribution to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data.
    /// * `y` - vector with target values (classes) of length N.
    /// * `alpha` - Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub fn fit<X: Array2<T>, Y: Array1<T>>(x: &X, y: &Y, alpha: f64) -> Result<Self, Failed> {
        if alpha < 0f64 {
            return Err(Failed::fit(&format!(
                "alpha should be >= 0, alpha=[{}]",
                alpha
            )));
        }

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
        let y: Vec<usize> = y.iterator(0).map(|y_i| y_i.to_usize().unwrap()).collect();

        let y_max = y
            .iter()
            .max()
            .ok_or_else(|| Failed::fit("Failed to get the labels of y."))?;

        let class_labels: Vec<T> = (0..*y_max + 1)
            .map(|label| T::from_usize(label).unwrap())
            .collect();
        let mut class_count = vec![0_usize; class_labels.len()];
        for elem in y.iter() {
            class_count[*elem] += 1;
        }

        let mut n_categories: Vec<usize> = Vec::with_capacity(n_features);
        for feature in 0..n_features {
            let feature_max = x
                .get_col(feature)
                .iterator(0)
                .map(|f_i| f_i.to_usize().unwrap())
                .max()
                .ok_or_else(|| {
                    Failed::fit(&format!(
                        "Failed to get the categories for feature = {}",
                        feature
                    ))
                })?;
            n_categories.push(feature_max + 1);
        }

        let mut coefficients: Vec<Vec<Vec<f64>>> = Vec::with_capacity(class_labels.len());
        let mut category_count: Vec<Vec<Vec<usize>>> = Vec::with_capacity(class_labels.len());
        for (feature_index, &n_categories_i) in n_categories.iter().enumerate().take(n_features) {
            let mut coef_i: Vec<Vec<f64>> = Vec::with_capacity(n_features);
            let mut category_count_i: Vec<Vec<usize>> = Vec::with_capacity(n_features);
            for (label, &label_count) in class_labels.iter().zip(class_count.iter()) {
                let col = x
                    .get_col(feature_index)
                    .iterator(0)
                    .enumerate()
                    .filter(|(i, _j)| T::from_usize(y[*i]).unwrap() == *label)
                    .map(|(_, j)| *j)
                    .collect::<Vec<T>>();
                let mut feat_count: Vec<usize> = vec![0_usize; n_categories_i];
                for row in col.iter() {
                    let index = row.to_usize().unwrap();
                    feat_count[index] += 1;
                }

                let coef_i_j = feat_count
                    .iter()
                    .map(|&c| {
                        ((c as f64 + alpha) / (label_count as f64 + n_categories_i as f64 * alpha))
                            .ln()
                    })
                    .collect::<Vec<f64>>();
                category_count_i.push(feat_count);
                coef_i.push(coef_i_j);
            }
            category_count.push(category_count_i);
            coefficients.push(coef_i);
        }

        let class_priors = class_count
            .iter()
            .map(|&count| count as f64 / n_samples as f64)
            .collect::<Vec<f64>>();

        Ok(Self {
            class_count,
            class_labels,
            class_priors,
            coefficients,
            n_features,
            n_categories,
            category_count,
        })
    }
}

/// `CategoricalNB` parameters. Use `Default::default()` for default values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct CategoricalNBParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: f64,
}

impl CategoricalNBParameters {
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

impl Default for CategoricalNBParameters {
    fn default() -> Self {
        Self { alpha: 1f64 }
    }
}

/// CategoricalNB grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct CategoricalNBSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    pub alpha: Vec<f64>,
}

/// CategoricalNB grid search iterator
pub struct CategoricalNBSearchParametersIterator {
    categorical_nb_search_parameters: CategoricalNBSearchParameters,
    current_alpha: usize,
}

impl IntoIterator for CategoricalNBSearchParameters {
    type Item = CategoricalNBParameters;
    type IntoIter = CategoricalNBSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        CategoricalNBSearchParametersIterator {
            categorical_nb_search_parameters: self,
            current_alpha: 0,
        }
    }
}

impl Iterator for CategoricalNBSearchParametersIterator {
    type Item = CategoricalNBParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_alpha == self.categorical_nb_search_parameters.alpha.len() {
            return None;
        }

        let next = CategoricalNBParameters {
            alpha: self.categorical_nb_search_parameters.alpha[self.current_alpha],
        };

        self.current_alpha += 1;

        Some(next)
    }
}

impl Default for CategoricalNBSearchParameters {
    fn default() -> Self {
        let default_params = CategoricalNBParameters::default();

        CategoricalNBSearchParameters {
            alpha: vec![default_params.alpha],
        }
    }
}

/// CategoricalNB implements the categorical naive Bayes algorithm for categorically distributed data.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub struct CategoricalNB<T: Number + Unsigned, X: Array2<T>, Y: Array1<T>> {
    inner: BaseNaiveBayes<T, T, X, Y, CategoricalNBDistribution<T>>,
}

impl<T: Number + Unsigned, X: Array2<T>, Y: Array1<T>>
    SupervisedEstimator<X, Y, CategoricalNBParameters> for CategoricalNB<T, X, Y>
{
    fn fit(x: &X, y: &Y, parameters: CategoricalNBParameters) -> Result<Self, Failed> {
        CategoricalNB::fit(x, y, parameters)
    }
}

impl<T: Number + Unsigned, X: Array2<T>, Y: Array1<T>> Predictor<X, Y> for CategoricalNB<T, X, Y> {
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<T: Number + Unsigned, X: Array2<T>, Y: Array1<T>> CategoricalNB<T, X, Y> {
    /// Fits CategoricalNB with given data
    /// * `x` - training data of size NxM where N is the number of samples and M is the number of
    /// features.
    /// * `y` - vector with target values (classes) of length N.
    /// * `parameters` - additional parameters like alpha for smoothing
    pub fn fit(x: &X, y: &Y, parameters: CategoricalNBParameters) -> Result<Self, Failed> {
        let alpha = parameters.alpha;
        let distribution = CategoricalNBDistribution::fit(x, y, alpha)?;
        let inner = BaseNaiveBayes::fit(distribution)?;
        Ok(Self { inner })
    }

    /// Estimates the class labels for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    /// Returns a vector of size N with class estimates.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
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

    /// Number of features of each sample
    pub fn n_features(&self) -> usize {
        self.inner.distribution.n_features
    }

    /// Number of features of each sample
    pub fn n_categories(&self) -> &Vec<usize> {
        &self.inner.distribution.n_categories
    }

    /// Holds arrays of shape (n_classes, n_categories of respective feature)
    /// for each feature. Each array provides the number of samples
    /// encountered for each class and category of the specific feature.
    pub fn category_count(&self) -> &Vec<Vec<Vec<usize>>> {
        &self.inner.distribution.category_count
    }
    /// Holds arrays of shape (n_classes, n_categories of respective feature)
    /// for each feature. Each array provides the empirical log probability
    /// of categories given the respective feature and class, ``P(x_i|y)``.
    pub fn feature_log_prob(&self) -> &Vec<Vec<Vec<f64>>> {
        &self.inner.distribution.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn search_parameters() {
        let parameters = CategoricalNBSearchParameters {
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
    fn run_categorical_naive_bayes() {
        let x = DenseMatrix::<u32>::from_2d_array(&[
            &[0, 2, 1, 0],
            &[0, 2, 1, 1],
            &[1, 2, 1, 0],
            &[2, 1, 1, 0],
            &[2, 0, 0, 0],
            &[2, 0, 0, 1],
            &[1, 0, 0, 1],
            &[0, 1, 1, 0],
            &[0, 0, 0, 0],
            &[2, 1, 0, 0],
            &[0, 1, 0, 1],
            &[1, 1, 1, 1],
            &[1, 2, 0, 0],
            &[2, 1, 1, 1],
        ]);
        let y: Vec<u32> = vec![0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0];

        let cnb = CategoricalNB::fit(&x, &y, Default::default()).unwrap();

        // checking parity with scikit
        assert_eq!(cnb.classes(), &[0, 1]);
        assert_eq!(cnb.class_count(), &[5, 9]);
        assert_eq!(cnb.n_features(), 4);
        assert_eq!(cnb.n_categories(), &[3, 3, 2, 2]);
        assert_eq!(
            cnb.category_count(),
            &vec![
                vec![vec![3, 0, 2], vec![2, 4, 3]],
                vec![vec![1, 2, 2], vec![3, 4, 2]],
                vec![vec![1, 4], vec![6, 3]],
                vec![vec![2, 3], vec![6, 3]]
            ]
        );

        assert_eq!(
            cnb.feature_log_prob(),
            &vec![
                vec![
                    vec![
                        -0.6931471805599453,
                        -2.0794415416798357,
                        -0.9808292530117262
                    ],
                    vec![
                        -1.3862943611198906,
                        -0.8754687373538999,
                        -1.0986122886681098
                    ]
                ],
                vec![
                    vec![
                        -1.3862943611198906,
                        -0.9808292530117262,
                        -0.9808292530117262
                    ],
                    vec![
                        -1.0986122886681098,
                        -0.8754687373538999,
                        -1.3862943611198906
                    ]
                ],
                vec![
                    vec![-1.252762968495368, -0.3364722366212129],
                    vec![-0.45198512374305727, -1.0116009116784799]
                ],
                vec![
                    vec![-0.8472978603872037, -0.5596157879354228],
                    vec![-0.45198512374305727, -1.0116009116784799]
                ]
            ]
        );

        let x_test = DenseMatrix::from_2d_array(&[&[0, 2, 1, 0], &[2, 2, 0, 0]]);
        let y_hat = cnb.predict(&x_test).unwrap();
        assert_eq!(y_hat, vec![0, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn run_categorical_naive_bayes2() {
        let x = DenseMatrix::<u32>::from_2d_array(&[
            &[3, 4, 0, 1],
            &[3, 0, 0, 1],
            &[4, 4, 1, 2],
            &[4, 2, 4, 3],
            &[4, 2, 4, 2],
            &[4, 1, 1, 0],
            &[1, 1, 1, 1],
            &[0, 4, 1, 0],
            &[0, 3, 2, 1],
            &[0, 3, 1, 1],
            &[3, 4, 0, 1],
            &[3, 4, 2, 4],
            &[0, 3, 1, 2],
            &[0, 4, 1, 2],
        ]);
        let y: Vec<u32> = vec![0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0];

        let cnb = CategoricalNB::fit(&x, &y, Default::default()).unwrap();
        let y_hat = cnb.predict(&x).unwrap();
        assert_eq!(y_hat, vec![0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::from_2d_array(&[
            &[3, 4, 0, 1],
            &[3, 0, 0, 1],
            &[4, 4, 1, 2],
            &[4, 2, 4, 3],
            &[4, 2, 4, 2],
            &[4, 1, 1, 0],
            &[1, 1, 1, 1],
            &[0, 4, 1, 0],
            &[0, 3, 2, 1],
            &[0, 3, 1, 1],
            &[3, 4, 0, 1],
            &[3, 4, 2, 4],
            &[0, 3, 1, 2],
            &[0, 4, 1, 2],
        ]);

        let y: Vec<u32> = vec![0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0];
        let cnb = CategoricalNB::fit(&x, &y, Default::default()).unwrap();

        let deserialized_cnb: CategoricalNB<u32, DenseMatrix<u32>, Vec<u32>> =
            serde_json::from_str(&serde_json::to_string(&cnb).unwrap()).unwrap();

        assert_eq!(cnb, deserialized_cnb);
    }
}
