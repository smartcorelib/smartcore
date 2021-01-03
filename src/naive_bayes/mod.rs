//! # Naive Bayes
//!
//! Naive Bayes (NB) is a simple but powerful machine learning algorithm.
//! Naive Bayes classifier is based on Bayes’ Theorem with an ssumption of conditional independence
//! between every pair of features given the value of the class variable.
//!
//! Bayes’ theorem can be written as
//!
//! \\[ P(y | X) = \frac{P(y)P(X| y)}{P(X)} \\]
//!
//! where
//!
//! * \\(X = (x_1,...x_n)\\) represents the predictors.
//! * \\(P(y | X)\\) is the probability of class _y_ given the data X
//! * \\(P(X| y)\\) is the probability of data X given the class _y_.
//! * \\(P(y)\\) is the probability of class y. This is called the prior probability of y.
//! * \\(P(y | X)\\) is the probability of the data (regardless of the class value).
//!
//! The naive conditional independence assumption let us rewrite this equation as
//!
//! \\[ P(y | x_1,...x_n) = \frac{P(y)\prod_{i=1}^nP(x_i|y)}{P(x_1,...x_n)} \\]
//!
//!
//! The denominator can be removed since \\(P(x_1,...x_n)\\) is constrant for all the entries in the dataset.
//!
//! \\[ P(y | x_1,...x_n) \propto P(y)\prod_{i=1}^nP(x_i|y) \\]
//!
//! To find class y from predictors X we use this equation
//!
//! \\[ y = \underset{y}{argmax} P(y)\prod_{i=1}^nP(x_i|y) \\]
//!
//! ## References:
//!
//! * ["Machine Learning: A Probabilistic Perspective", Kevin P. Murphy, 2012, Chapter 3 ](https://mitpress.mit.edu/books/machine-learning-1)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Distribution used in the Naive Bayes classifier.
pub(crate) trait NBDistribution<T: RealNumber, M: Matrix<T>> {
    /// Prior of class at the given index.
    fn prior(&self, class_index: usize) -> T;

    /// Logarithm of conditional probability of sample j given class in the specified index.
    fn log_likelihood(&self, class_index: usize, j: &M::RowVector) -> T;

    /// Possible classes of the distribution.
    fn classes(&self) -> &Vec<T>;
}

/// Base struct for the Naive Bayes classifier.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub(crate) struct BaseNaiveBayes<T: RealNumber, M: Matrix<T>, D: NBDistribution<T, M>> {
    distribution: D,
    _phantom_t: PhantomData<T>,
    _phantom_m: PhantomData<M>,
}

impl<T: RealNumber, M: Matrix<T>, D: NBDistribution<T, M>> BaseNaiveBayes<T, M, D> {
    /// Fits NB classifier to a given NBdistribution.
    /// * `distribution` - NBDistribution of the training data
    pub fn fit(distribution: D) -> Result<Self, Failed> {
        Ok(Self {
            distribution,
            _phantom_t: PhantomData,
            _phantom_m: PhantomData,
        })
    }

    /// Estimates the class labels for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    /// Returns a vector of size N with class estimates.
    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        let y_classes = self.distribution.classes();
        let (rows, _) = x.shape();
        let predictions = (0..rows)
            .map(|row_index| {
                let row = x.get_row(row_index);
                let (prediction, _probability) = y_classes
                    .iter()
                    .enumerate()
                    .map(|(class_index, class)| {
                        (
                            class,
                            self.distribution.log_likelihood(class_index, &row)
                                + self.distribution.prior(class_index).ln(),
                        )
                    })
                    .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
                    .unwrap();
                *prediction
            })
            .collect::<Vec<T>>();
        let y_hat = M::RowVector::from_array(&predictions);
        Ok(y_hat)
    }
}
pub mod bernoulli;
pub mod categorical;
pub mod gaussian;
pub mod multinomial;
