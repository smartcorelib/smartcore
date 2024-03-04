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
use std::{cmp::Ordering, marker::PhantomData};

/// Distribution used in the Naive Bayes classifier.
pub(crate) trait NBDistribution<X: Number, Y: Number>: Clone {
    /// Prior of class at the given index.
    fn prior(&self, class_index: usize) -> f64;

    /// Logarithm of conditional probability of sample j given class in the specified index.
    #[allow(clippy::borrowed_box)]
    fn log_likelihood<'a>(&'a self, class_index: usize, j: &'a Box<dyn ArrayView1<X> + 'a>) -> f64;

    /// Possible classes of the distribution.
    fn classes(&self) -> &Vec<Y>;
}

/// Base struct for the Naive Bayes classifier.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq, Clone)]
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
        let predictions = x
            .row_iter()
            .map(|row| {
                y_classes
                    .iter()
                    .enumerate()
                    .map(|(class_index, class)| {
                        (
                            class,
                            self.distribution.log_likelihood(class_index, &row)
                                + self.distribution.prior(class_index).ln(),
                        )
                    })
                    // For some reason, the max_by method cannot use NaNs for finding the maximum value, it panics.
                    // NaN must be considered as minimum values,
                    // therefore it's like NaNs would not be considered for choosing the maximum value.
                    // So we need to handle this case for avoiding panicking by using `Option::unwrap`.
                    .max_by(|(_, p1), (_, p2)| match p1.partial_cmp(p2) {
                        Some(ordering) => ordering,
                        None => {
                            if p1.is_nan() {
                                Ordering::Less
                            } else if p2.is_nan() {
                                Ordering::Greater
                            } else {
                                Ordering::Equal
                            }
                        }
                    })
                    .map(|(prediction, _probability)| *prediction)
                    .ok_or_else(|| Failed::predict("Failed to predict, there is no result"))
            })
            .collect::<Result<Vec<TY>, Failed>>()?;
        let y_hat = Y::from_vec_slice(&predictions);
        Ok(y_hat)
    }
}
pub mod bernoulli;
pub mod categorical;
pub mod gaussian;
pub mod multinomial;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;
    use num_traits::float::Float;

    type Model<'d> = BaseNaiveBayes<i32, i32, DenseMatrix<i32>, Vec<i32>, TestDistribution<'d>>;

    #[derive(Debug, PartialEq, Clone)]
    struct TestDistribution<'d>(&'d Vec<i32>);

    impl<'d> NBDistribution<i32, i32> for TestDistribution<'d> {
        fn prior(&self, _class_index: usize) -> f64 {
            1.
        }

        fn log_likelihood<'a>(
            &'a self,
            class_index: usize,
            _j: &'a Box<dyn ArrayView1<i32> + 'a>,
        ) -> f64 {
            match self.0.get(class_index) {
                &v @ 2 | &v @ 10 | &v @ 20 => v as f64,
                _ => f64::nan(),
            }
        }

        fn classes(&self) -> &Vec<i32> {
            &self.0
        }
    }

    #[test]
    fn test_predict() {
        let matrix = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]);

        let val = vec![];
        match Model::fit(TestDistribution(&val)).unwrap().predict(&matrix) {
            Ok(_) => panic!("Should return error in case of empty classes"),
            Err(err) => assert_eq!(
                err.to_string(),
                "Predict failed: Failed to predict, there is no result"
            ),
        }

        let val = vec![1, 2, 3];
        match Model::fit(TestDistribution(&val)).unwrap().predict(&matrix) {
            Ok(r) => assert_eq!(r, vec![2, 2, 2]),
            Err(_) => panic!("Should success in normal case with NaNs"),
        }

        let val = vec![20, 2, 10];
        match Model::fit(TestDistribution(&val)).unwrap().predict(&matrix) {
            Ok(r) => assert_eq!(r, vec![20, 20, 20]),
            Err(_) => panic!("Should success in normal case without NaNs"),
        }
    }
}
