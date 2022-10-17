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
use crate::linalg::basic::arrays::{Array1, Array2, ArrayView1};
use crate::numbers::basenum::Number;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Distribution used in the Naive Bayes classifier.
pub(crate) trait NBDistribution<X: Number, Y: Number> {
    /// Prior of class at the given index.
    fn prior(&self, class_index: usize) -> f64;

    /// Logarithm of conditional probability of sample j given class in the specified index.
    fn log_likelihood<'a>(&'a self, class_index: usize, j: &'a Box<dyn ArrayView1<X> + 'a>) -> f64;

    /// Possible classes of the distribution.
    fn classes(&self) -> &Vec<Y>;
}

/// Base struct for the Naive Bayes classifier.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub(crate) struct BaseNaiveBayes<
    TX: Number,
    TY: Number,
    X: Array2<TX>,
    Y: Array1<TY>,
    D: NBDistribution<TX, TY>,
> {
    distribution: D,
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number, TY: Number, X: Array2<TX>, Y: Array1<TY>, D: NBDistribution<TX, TY>>
    BaseNaiveBayes<TX, TY, X, Y, D>
{
    /// Fits NB classifier to a given NBdistribution.
    /// * `distribution` - NBDistribution of the training data
    pub fn fit(distribution: D) -> Result<Self, Failed> {
        Ok(Self {
            distribution,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        })
    }

    /// Estimates the class labels for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    /// Returns a vector of size N with class estimates.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
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
            .collect::<Vec<TY>>();
        let y_hat = Y::from_vec_slice(&predictions);
        Ok(y_hat)
    }
}
pub mod bernoulli;
pub mod categorical;
pub mod gaussian;
pub mod multinomial;
