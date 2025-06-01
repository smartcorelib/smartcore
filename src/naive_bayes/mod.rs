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
    ///
    /// Returns a vector of size N with class estimates.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let y_classes = self.distribution.classes();

        if y_classes.is_empty() {
            return Err(Failed::predict("Failed to predict, no classes available"));
        }

        let (rows, _) = x.shape();
        let mut predictions = Vec::with_capacity(rows);
        let mut all_probs_nan = true;

        for row_index in 0..rows {
            let row = x.get_row(row_index);
            let mut max_log_prob = f64::NEG_INFINITY;
            let mut max_class = None;

            for (class_index, class) in y_classes.iter().enumerate() {
                let log_likelihood = self.distribution.log_likelihood(class_index, &row);
                let log_prob = log_likelihood + self.distribution.prior(class_index).ln();

                if !log_prob.is_nan() && log_prob > max_log_prob {
                    max_log_prob = log_prob;
                    max_class = Some(*class);
                    all_probs_nan = false;
                }
            }

            predictions.push(max_class.unwrap_or(y_classes[0]));
        }

        if all_probs_nan {
            Err(Failed::predict(
                "Failed to predict, all probabilities were NaN",
            ))
        } else {
            Ok(Y::from_vec_slice(&predictions))
        }
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

    impl NBDistribution<i32, i32> for TestDistribution<'_> {
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
            self.0
        }
    }

    #[test]
    fn test_predict() {
        let matrix = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).unwrap();

        let val = vec![];
        match Model::fit(TestDistribution(&val)).unwrap().predict(&matrix) {
            Ok(_) => panic!("Should return error in case of empty classes"),
            Err(err) => assert_eq!(
                err.to_string(),
                "Predict failed: Failed to predict, no classes available"
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

    // A simple test distribution using float
    #[derive(Debug, PartialEq, Clone)]
    struct TestDistributionAgain {
        classes: Vec<u32>,
        probs: Vec<f64>,
    }

    impl NBDistribution<f64, u32> for TestDistributionAgain {
        fn classes(&self) -> &Vec<u32> {
            &self.classes
        }
        fn prior(&self, class_index: usize) -> f64 {
            self.probs[class_index]
        }
        fn log_likelihood<'a>(
            &'a self,
            class_index: usize,
            _j: &'a Box<dyn ArrayView1<f64> + 'a>,
        ) -> f64 {
            self.probs[class_index].ln()
        }
    }

    type TestNB = BaseNaiveBayes<f64, u32, DenseMatrix<f64>, Vec<u32>, TestDistributionAgain>;

    #[test]
    fn test_predict_empty_classes() {
        let dist = TestDistributionAgain {
            classes: vec![],
            probs: vec![],
        };
        let nb = TestNB::fit(dist).unwrap();
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        assert!(nb.predict(&x).is_err());
    }

    #[test]
    fn test_predict_single_class() {
        let dist = TestDistributionAgain {
            classes: vec![1],
            probs: vec![1.0],
        };
        let nb = TestNB::fit(dist).unwrap();
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let result = nb.predict(&x).unwrap();
        assert_eq!(result, vec![1, 1]);
    }

    #[test]
    fn test_predict_multiple_classes() {
        let dist = TestDistributionAgain {
            classes: vec![1, 2, 3],
            probs: vec![0.2, 0.5, 0.3],
        };
        let nb = TestNB::fit(dist).unwrap();
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]).unwrap();
        let result = nb.predict(&x).unwrap();
        assert_eq!(result, vec![2, 2, 2]);
    }

    #[test]
    fn test_predict_with_nans() {
        let dist = TestDistributionAgain {
            classes: vec![1, 2],
            probs: vec![f64::NAN, 0.5],
        };
        let nb = TestNB::fit(dist).unwrap();
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let result = nb.predict(&x).unwrap();
        assert_eq!(result, vec![2, 2]);
    }

    #[test]
    fn test_predict_all_nans() {
        let dist = TestDistributionAgain {
            classes: vec![1, 2],
            probs: vec![f64::NAN, f64::NAN],
        };
        let nb = TestNB::fit(dist).unwrap();
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        assert!(nb.predict(&x).is_err());
    }

    #[test]
    fn test_predict_extreme_probabilities() {
        let dist = TestDistributionAgain {
            classes: vec![1, 2],
            probs: vec![1e-300, 1e-301],
        };
        let nb = TestNB::fit(dist).unwrap();
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let result = nb.predict(&x).unwrap();
        assert_eq!(result, vec![1, 1]);
    }

    #[test]
    fn test_predict_with_infinity() {
        let dist = TestDistributionAgain {
            classes: vec![1, 2, 3],
            probs: vec![f64::INFINITY, 1.0, 2.0],
        };
        let nb = TestNB::fit(dist).unwrap();
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let result = nb.predict(&x).unwrap();
        assert_eq!(result, vec![1, 1]);
    }

    #[test]
    fn test_predict_with_negative_infinity() {
        let dist = TestDistributionAgain {
            classes: vec![1, 2, 3],
            probs: vec![f64::NEG_INFINITY, 1.0, 2.0],
        };
        let nb = TestNB::fit(dist).unwrap();
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let result = nb.predict(&x).unwrap();
        assert_eq!(result, vec![3, 3]);
    }

    #[test]
    fn test_gaussian_naive_bayes_numerical_stability() {
        #[derive(Debug, PartialEq, Clone)]
        struct GaussianTestDistribution {
            classes: Vec<u32>,
            means: Vec<Vec<f64>>,
            variances: Vec<Vec<f64>>,
            priors: Vec<f64>,
        }

        impl NBDistribution<f64, u32> for GaussianTestDistribution {
            fn classes(&self) -> &Vec<u32> {
                &self.classes
            }

            fn prior(&self, class_index: usize) -> f64 {
                self.priors[class_index]
            }

            fn log_likelihood<'a>(
                &'a self,
                class_index: usize,
                j: &'a Box<dyn ArrayView1<f64> + 'a>,
            ) -> f64 {
                let means = &self.means[class_index];
                let variances = &self.variances[class_index];
                j.iterator(0)
                    .enumerate()
                    .map(|(i, &xi)| {
                        let mean = means[i];
                        let var = variances[i] + 1e-9; // Small smoothing for numerical stability
                        let coeff = -0.5 * (2.0 * std::f64::consts::PI * var).ln();
                        let exponent = -(xi - mean).powi(2) / (2.0 * var);
                        coeff + exponent
                    })
                    .sum()
            }
        }

        fn train_distribution(x: &DenseMatrix<f64>, y: &[u32]) -> GaussianTestDistribution {
            let mut classes: Vec<u32> = y
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<u32>>()
                .into_iter()
                .collect();
            classes.sort();
            let n_classes = classes.len();
            let n_features = x.shape().1;

            let mut means = vec![vec![0.0; n_features]; n_classes];
            let mut variances = vec![vec![0.0; n_features]; n_classes];
            let mut class_counts = vec![0; n_classes];

            // Calculate means and count samples per class
            for (sample, &class) in x.row_iter().zip(y.iter()) {
                let class_idx = classes.iter().position(|&c| c == class).unwrap();
                class_counts[class_idx] += 1;
                for (i, &value) in sample.iterator(0).enumerate() {
                    means[class_idx][i] += value;
                }
            }

            // Normalize means
            for (class_idx, mean) in means.iter_mut().enumerate() {
                for value in mean.iter_mut() {
                    *value /= class_counts[class_idx] as f64;
                }
            }

            // Calculate variances
            for (sample, &class) in x.row_iter().zip(y.iter()) {
                let class_idx = classes.iter().position(|&c| c == class).unwrap();
                for (i, &value) in sample.iterator(0).enumerate() {
                    let diff = value - means[class_idx][i];
                    variances[class_idx][i] += diff * diff;
                }
            }

            // Normalize variances and add small epsilon to avoid zero variance
            let epsilon = 1e-9;
            for (class_idx, variance) in variances.iter_mut().enumerate() {
                for value in variance.iter_mut() {
                    *value = *value / class_counts[class_idx] as f64 + epsilon;
                }
            }

            // Calculate priors
            let total_samples = y.len() as f64;
            let priors: Vec<f64> = class_counts
                .iter()
                .map(|&count| count as f64 / total_samples)
                .collect();

            GaussianTestDistribution {
                classes,
                means,
                variances,
                priors,
            }
        }

        type TestNBGaussian =
            BaseNaiveBayes<f64, u32, DenseMatrix<f64>, Vec<u32>, GaussianTestDistribution>;

        // Create a constant training dataset
        let n_samples = 1000;
        let n_features = 5;
        let n_classes = 4;

        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            for j in 0..n_features {
                x_data.push((i * j) as f64 % 10.0);
            }
            y_data.push((i % n_classes) as u32);
        }

        let x = DenseMatrix::new(n_samples, n_features, x_data, true).unwrap();
        let y = y_data;

        // Train the model
        let dist = train_distribution(&x, &y);
        let nb = TestNBGaussian::fit(dist).unwrap();

        // Create constant test data
        let n_test_samples = 100;
        let mut test_x_data = Vec::with_capacity(n_test_samples * n_features);
        for i in 0..n_test_samples {
            for j in 0..n_features {
                test_x_data.push((i * j * 2) as f64 % 15.0);
            }
        }
        let test_x = DenseMatrix::new(n_test_samples, n_features, test_x_data, true).unwrap();

        // Make predictions
        let predictions = nb
            .predict(&test_x)
            .map_err(|e| format!("Prediction failed: {}", e))
            .unwrap();

        // Check numerical stability
        assert_eq!(
            predictions.len(),
            n_test_samples,
            "Number of predictions should match number of test samples"
        );

        // Check that all predictions are valid class labels
        for &pred in predictions.iter() {
            assert!(pred < n_classes as u32, "Predicted class should be valid");
        }

        // Check consistency of predictions
        let repeated_predictions = nb
            .predict(&test_x)
            .map_err(|e| format!("Repeated prediction failed: {}", e))
            .unwrap();
        assert_eq!(
            predictions, repeated_predictions,
            "Predictions should be consistent when repeated"
        );

        // Check extreme values
        let extreme_x =
            DenseMatrix::new(2, n_features, vec![f64::MAX; n_features * 2], true).unwrap();
        let extreme_predictions = nb.predict(&extreme_x);
        assert!(
            extreme_predictions.is_err(),
            "Extreme value input should result in an error"
        );
        assert_eq!(
            extreme_predictions.unwrap_err().to_string(),
            "Predict failed: Failed to predict, all probabilities were NaN",
            "Incorrect error message for extreme values"
        );

        // Check for NaN handling
        let nan_x = DenseMatrix::new(2, n_features, vec![f64::NAN; n_features * 2], true).unwrap();
        let nan_predictions = nb.predict(&nan_x);
        assert!(
            nan_predictions.is_err(),
            "NaN input should result in an error"
        );

        // Check for very small values
        let small_x =
            DenseMatrix::new(2, n_features, vec![f64::MIN_POSITIVE; n_features * 2], true).unwrap();
        let small_predictions = nb
            .predict(&small_x)
            .map_err(|e| format!("Small value prediction failed: {}", e))
            .unwrap();
        for &pred in small_predictions.iter() {
            assert!(
                pred < n_classes as u32,
                "Predictions for very small values should be valid"
            );
        }

        // Check for values close to zero
        let near_zero_x =
            DenseMatrix::new(2, n_features, vec![1e-300; n_features * 2], true).unwrap();
        let near_zero_predictions = nb
            .predict(&near_zero_x)
            .map_err(|e| format!("Near-zero value prediction failed: {}", e))
            .unwrap();
        for &pred in near_zero_predictions.iter() {
            assert!(
                pred < n_classes as u32,
                "Predictions for near-zero values should be valid"
            );
        }

        println!("All numerical stability checks passed!");
    }

    #[test]
    fn test_gaussian_naive_bayes_numerical_stability_random_data() {
        #[derive(Debug)]
        struct MySimpleRng {
            state: u64,
        }

        impl MySimpleRng {
            fn new(seed: u64) -> Self {
                MySimpleRng { state: seed }
            }

            /// Get the next u64 in the sequence.
            fn next_u64(&mut self) -> u64 {
                // LCG parameters; these are somewhat arbitrary but commonly used.
                // Feel free to tweak the multiplier/adder etc.
                self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
                self.state
            }

            /// Get an f64 in the range [min, max).
            fn next_f64(&mut self, min: f64, max: f64) -> f64 {
                let fraction = (self.next_u64() as f64) / (u64::MAX as f64);
                min + fraction * (max - min)
            }

            /// Get a usize in the range [min, max). This floors the floating result.
            fn gen_range_usize(&mut self, min: usize, max: usize) -> usize {
                let v = self.next_f64(min as f64, max as f64);
                // Truncate into the integer range. Because of floating inexactness,
                // ensure we also clamp.
                let int_v = v.floor() as isize;
                // simple clamp to avoid any float rounding out of range
                let clamped = int_v.max(min as isize).min((max - 1) as isize);
                clamped as usize
            }
        }
        use crate::naive_bayes::gaussian::GaussianNB;
        // We will generate random data in a reproducible way (using a fixed seed).
        // We will generate random data in a reproducible way:
        let mut rng = MySimpleRng::new(42);

        let n_samples = 1000;
        let n_features = 5;
        let n_classes = 4;

        // Our feature matrix and label vector
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data = Vec::with_capacity(n_samples);

        // Fill x_data with random values and y_data with random class labels.
        for _i in 0..n_samples {
            for _j in 0..n_features {
                // We’ll pick random values in [-10, 10).
                x_data.push(rng.next_f64(-10.0, 10.0));
            }
            let class = rng.gen_range_usize(0, n_classes) as u32;
            y_data.push(class);
        }

        // Create DenseMatrix from x_data
        let x = DenseMatrix::new(n_samples, n_features, x_data, true).unwrap();

        // Train GaussianNB
        let gnb = GaussianNB::fit(&x, &y_data, Default::default())
            .expect("Fitting GaussianNB with random data failed.");

        // Predict on the same training data to verify no numerical instability
        let predictions = gnb.predict(&x).expect("Prediction on random data failed.");

        // Basic sanity checks
        assert_eq!(
            predictions.len(),
            n_samples,
            "Prediction size must match n_samples"
        );
        for &pred_class in &predictions {
            assert!(
                (pred_class as usize) < n_classes,
                "Predicted class {} is out of range [0..n_classes).",
                pred_class
            );
        }

        // If you want to compare with scikit-learn, you can do something like:
        // println!("X = {:?}", &x);
        // println!("Y = {:?}", &y_data);
        // println!("predictions = {:?}", &predictions);
        // and then in Python:
        //    import numpy as np
        //    from sklearn.naive_bayes import GaussianNB
        //    X = np.reshape(np.array(x), (1000, 5), order='F')
        //    Y = np.array(y)
        //    gnb = GaussianNB().fit(X, Y)
        //    preds = gnb.predict(X)
        //    expected = np.array(predictions)
        //    assert expected == preds
        // They should match closely (or exactly) depending on floating rounding.
    }
}
