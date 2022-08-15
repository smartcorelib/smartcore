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
    /// number of training samples observed in each class
    class_count: Vec<usize>,
    /// class labels known to the classifier
    class_labels: Vec<T>,
    /// probability of each class
    class_priors: Vec<T>,
    coefficients: Vec<Vec<Vec<T>>>,
    /// Number of features of each sample
    n_features: usize,
    /// Number of categories for each feature
    n_categories: Vec<usize>,
    /// Holds arrays of shape (n_classes, n_categories of respective feature)
    /// for each feature. Each array provides the number of samples
    /// encountered for each class and category of the specific feature.
    category_count: Vec<Vec<Vec<usize>>>,
}

impl<T: RealNumber> PartialEq for CategoricalNBDistribution<T> {
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
                if self.coefficients[feature][class_index].len() > value {
                    likelihood += self.coefficients[feature][class_index][value];
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
            .ok_or_else(|| Failed::fit("Failed to get the labels of y."))?;

        let class_labels: Vec<T> = (0..*y_max + 1)
            .map(|label| T::from(label).unwrap())
            .collect();
        let mut class_count = vec![0_usize; class_labels.len()];
        for elem in y.iter() {
            class_count[*elem] += 1;
        }

        let mut n_categories: Vec<usize> = Vec::with_capacity(n_features);
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
            n_categories.push(feature_max + 1);
        }

        let mut coefficients: Vec<Vec<Vec<T>>> = Vec::with_capacity(class_labels.len());
        let mut category_count: Vec<Vec<Vec<usize>>> = Vec::with_capacity(class_labels.len());
        for (feature_index, &n_categories_i) in n_categories.iter().enumerate().take(n_features) {
            let mut coef_i: Vec<Vec<T>> = Vec::with_capacity(n_features);
            let mut category_count_i: Vec<Vec<usize>> = Vec::with_capacity(n_features);
            for (label, &label_count) in class_labels.iter().zip(class_count.iter()) {
                let col = x
                    .get_col_as_vec(feature_index)
                    .iter()
                    .enumerate()
                    .filter(|(i, _j)| T::from(y[*i]).unwrap() == *label)
                    .map(|(_, j)| *j)
                    .collect::<Vec<T>>();
                let mut feat_count: Vec<usize> = vec![0_usize; n_categories_i];
                for row in col.iter() {
                    let index = row.floor().to_usize().unwrap();
                    feat_count[index] += 1;
                }

                let coef_i_j = feat_count
                    .iter()
                    .map(|c| {
                        ((T::from(*c).unwrap() + alpha)
                            / (T::from(label_count).unwrap()
                                + T::from(n_categories_i).unwrap() * alpha))
                            .ln()
                    })
                    .collect::<Vec<T>>();
                category_count_i.push(feat_count);
                coef_i.push(coef_i_j);
            }
            category_count.push(category_count_i);
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
            n_features,
            n_categories,
            category_count,
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
    pub fn feature_log_prob(&self) -> &Vec<Vec<Vec<T>>> {
        &self.inner.distribution.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

        // checking parity with scikit
        assert_eq!(cnb.classes(), &[0., 1.]);
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

        let x_test = DenseMatrix::from_2d_array(&[&[0., 2., 1., 0.], &[2., 2., 0., 0.]]);
        let y_hat = cnb.predict(&x_test).unwrap();
        assert_eq!(y_hat, vec![0., 1.]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
