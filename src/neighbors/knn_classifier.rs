//! # K Nearest Neighbors Classifier
//!
//! `smartcore` relies on 2 backend algorithms to speedup KNN queries:
//! * [`LinearSearch`](../../algorithm/neighbour/linear_search/index.html)
//! * [`CoverTree`](../../algorithm/neighbour/cover_tree/index.html)
//!
//! The parameter `k` controls the stability of the KNN estimate: when `k` is small the algorithm is sensitive to the noise in data. When `k` increases the estimator becomes more stable.
//! In terms of the bias variance trade-off the variance decreases with `k` and the bias is likely to increase with `k`.
//!
//! When you don't know which search algorithm and `k` value to use go with default parameters defined by `Default::default()`
//!
//! To fit the model to a 4 x 2 matrix with 4 training samples, 2 features per sample:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::neighbors::knn_classifier::*;
//! use smartcore::metrics::distance::*;
//!
//! //your explanatory variables. Each row is a training sample with 2 numerical features
//! let x = DenseMatrix::from_2d_array(&[
//!     &[1., 2.],
//!     &[3., 4.],
//!     &[5., 6.],
//!     &[7., 8.],
//! &[9., 10.]]).unwrap();
//! let y = vec![2, 2, 2, 3, 3]; //your class labels
//!
//! let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
//! let y_hat = knn.predict(&x).unwrap();
//! ```
//!
//! variable `y_hat` will hold a vector with estimates of class labels
//!
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::algorithm::neighbour::{KNNAlgorithm, KNNAlgorithmName};
use crate::api::{Predictor, SupervisedEstimator};
use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::metrics::distance::euclidian::Euclidian;
use crate::metrics::distance::{Distance, Distances};
use crate::neighbors::KNNWeightFunction;
use crate::numbers::basenum::Number;

/// `KNNClassifier` parameters. Use `Default::default()` for default values.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct KNNClassifierParameters<T: Number, D: Distance<Vec<T>>> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// a function that defines a distance between each pair of point in training data.
    /// This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    /// See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    pub distance: D,
    #[cfg_attr(feature = "serde", serde(default))]
    /// backend search algorithm. See [`knn search algorithms`](../../algorithm/neighbour/index.html). `CoverTree` is default.
    pub algorithm: KNNAlgorithmName,
    #[cfg_attr(feature = "serde", serde(default))]
    /// weighting function that is used to calculate estimated class value. Default function is `KNNWeightFunction::Uniform`.
    pub weight: KNNWeightFunction,
    #[cfg_attr(feature = "serde", serde(default))]
    /// number of training samples to consider when estimating class for new point. Default value is 3.
    pub k: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// this parameter is not used
    t: PhantomData<T>,
}

/// K Nearest Neighbors Classifier
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct KNNClassifier<
    TX: Number,
    TY: Number + Ord,
    X: Array2<TX>,
    Y: Array1<TY>,
    D: Distance<Vec<TX>>,
> {
    classes: Option<Vec<TY>>,
    y: Option<Vec<usize>>,
    knn_algorithm: Option<KNNAlgorithm<TX, D>>,
    weight: Option<KNNWeightFunction>,
    k: Option<usize>,
    _phantom_tx: PhantomData<TX>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>, D: Distance<Vec<TX>>>
    KNNClassifier<TX, TY, X, Y, D>
{
    fn classes(&self) -> &Vec<TY> {
        self.classes.as_ref().unwrap()
    }
    fn y(&self) -> &Vec<usize> {
        self.y.as_ref().unwrap()
    }
    fn knn_algorithm(&self) -> &KNNAlgorithm<TX, D> {
        self.knn_algorithm.as_ref().unwrap()
    }
    fn weight(&self) -> &KNNWeightFunction {
        self.weight.as_ref().unwrap()
    }
    fn k(&self) -> usize {
        self.k.unwrap()
    }
}

impl<T: Number, D: Distance<Vec<T>>> KNNClassifierParameters<T, D> {
    /// number of training samples to consider when estimating class for new point. Default value is 3.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
    /// a function that defines a distance between each pair of point in training data.
    /// This function should extend [`Distance`](../../math/distance/trait.Distance.html) trait.
    /// See [`Distances`](../../math/distance/struct.Distances.html) for a list of available functions.
    pub fn with_distance<DD: Distance<Vec<T>>>(
        self,
        distance: DD,
    ) -> KNNClassifierParameters<T, DD> {
        KNNClassifierParameters {
            distance,
            algorithm: self.algorithm,
            weight: self.weight,
            k: self.k,
            t: PhantomData,
        }
    }
    /// backend search algorithm. See [`knn search algorithms`](../../algorithm/neighbour/index.html). `CoverTree` is default.
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithmName) -> Self {
        self.algorithm = algorithm;
        self
    }
    /// weighting function that is used to calculate estimated class value. Default function is `KNNWeightFunction::Uniform`.
    pub fn with_weight(mut self, weight: KNNWeightFunction) -> Self {
        self.weight = weight;
        self
    }
}

impl<T: Number> Default for KNNClassifierParameters<T, Euclidian<T>> {
    fn default() -> Self {
        KNNClassifierParameters {
            distance: Distances::euclidian(),
            algorithm: KNNAlgorithmName::default(),
            weight: KNNWeightFunction::default(),
            k: 3,
            t: PhantomData,
        }
    }
}

impl<TX: Number, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>, D: Distance<Vec<TX>>> PartialEq
    for KNNClassifier<TX, TY, X, Y, D>
{
    fn eq(&self, other: &Self) -> bool {
        if self.classes().len() != other.classes().len()
            || self.k() != other.k()
            || self.y().len() != other.y().len()
        {
            false
        } else {
            for i in 0..self.classes().len() {
                if self.classes()[i] != other.classes()[i] {
                    return false;
                }
            }
            for i in 0..self.y().len() {
                if self.y().get(i) != other.y().get(i) {
                    return false;
                }
            }
            true
        }
    }
}

impl<TX: Number, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>, D: Distance<Vec<TX>>>
    SupervisedEstimator<X, Y, KNNClassifierParameters<TX, D>> for KNNClassifier<TX, TY, X, Y, D>
{
    fn new() -> Self {
        Self {
            classes: Option::None,
            y: Option::None,
            knn_algorithm: Option::None,
            weight: Option::None,
            k: Option::None,
            _phantom_tx: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        }
    }
    fn fit(x: &X, y: &Y, parameters: KNNClassifierParameters<TX, D>) -> Result<Self, Failed> {
        KNNClassifier::fit(x, y, parameters)
    }
}

impl<TX: Number, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>, D: Distance<Vec<TX>>>
    Predictor<X, Y> for KNNClassifier<TX, TY, X, Y, D>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>, D: Distance<Vec<TX>>>
    KNNClassifier<TX, TY, X, Y, D>
{
    /// Fits KNN classifier to a NxM matrix where N is number of samples and M is number of features.
    /// * `x` - training data
    /// * `y` - vector with target values (classes) of length N
    /// * `parameters` - additional parameters like search algorithm and k
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: KNNClassifierParameters<TX, D>,
    ) -> Result<KNNClassifier<TX, TY, X, Y, D>, Failed> {
        let y_n = y.shape();
        let (x_n, _) = x.shape();

        let data = x
            .row_iter()
            .map(|row| row.iterator(0).copied().collect())
            .collect();

        let mut yi: Vec<usize> = vec![0; y_n];
        let classes = y.unique();

        for (i, yi_i) in yi.iter_mut().enumerate().take(y_n) {
            let yc = *y.get(i);
            *yi_i = classes.iter().position(|c| yc == *c).unwrap();
        }

        if x_n != y_n {
            return Err(Failed::fit(&format!(
                "Size of x should equal size of y; |x|=[{x_n}], |y|=[{y_n}]"
            )));
        }

        if parameters.k <= 1 {
            return Err(Failed::fit(&format!(
                "k should be > 1, k=[{}]",
                parameters.k
            )));
        }

        Ok(KNNClassifier {
            classes: Some(classes),
            y: Some(yi),
            k: Some(parameters.k),
            knn_algorithm: Some(parameters.algorithm.fit(data, parameters.distance)?),
            weight: Some(parameters.weight),
            _phantom_tx: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        })
    }

    /// Estimates the class labels for the provided data.
    /// * `x` - data of shape NxM where N is number of data points to estimate and M is number of features.
    ///
    /// Returns a vector of size N with class estimates.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let mut result = Y::zeros(x.shape().0);

        let mut row_vec = vec![TX::zero(); x.shape().1];
        for (i, row) in x.row_iter().enumerate() {
            row.iterator(0)
                .zip(row_vec.iter_mut())
                .for_each(|(&s, v)| *v = s);
            result.set(i, self.classes()[self.predict_for_row(&row_vec)?]);
        }

        Ok(result)
    }

    /// Compute class probabilities for a single row. All the rest functions will use it
    fn predict_proba_for_row(&self, row: &Vec<TX>) -> Result<Vec<f64>, Failed> {
        let search_result = self.knn_algorithm().find(row, self.k())?;

        // Getting distances and calculating weights
        let weights = self
            .weight()
            .calc_weights(search_result.iter().map(|v| v.1).collect());

        let w_sum: f64 = weights.iter().copied().sum();

        // Additional check. If weights sum == 0, normalization is not possible
        if w_sum == 0.0 {
            return Err(Failed::because(
                FailedError::PredictFailed,
                "Sum of weights is zero; cannot compute probabilities",
            ));
        }

        // Accumulating raw weights...
        let mut class_votes = vec![0.0; self.classes().len()];
        for (r, w) in search_result.iter().zip(weights.iter()) {
            // r.0 - index of a neighbor in X
            // self.y()[r.0] - class index of this neighbor (0, 1, 2...)
            class_votes[self.y()[r.0]] += *w;
        }

        // Normalization with a bit of optimization
        let inv_sum = 1.0 / w_sum;
        for v in &mut class_votes {
            *v *= inv_sum;
        }

        Ok(class_votes)
    }

    /// Predicts class index for a single row by reusing predict_proba_for_row
    fn predict_for_row(&self, row: &Vec<TX>) -> Result<usize, Failed> {
        let proba = self.predict_proba_for_row(row)?;
        let mut max_idx = 0;
        let mut max_val = proba[0];

        for (i, &val) in proba.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx) // Goes directly to already existing predict() method
    }

    /// Predict class probabilities for the input samples.
    /// Returns a vector of probability vectors, one per sample.
    /// Each probability vector has length equal to number of classes and sums to 1.
    pub fn predict_proba(&self, x: &X) -> Result<Vec<Vec<f64>>, Failed> {
        let mut result = Vec::with_capacity(x.shape().0);
        let mut row_vec = vec![TX::zero(); x.shape().1];
        for row in x.row_iter() {
            row.iterator(0)
                .zip(row_vec.iter_mut())
                .for_each(|(&s, v)| *v = s);
            result.push(self.predict_proba_for_row(&row_vec)?);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

    /// Helper function to compare two f64 vectors with tolerance, placing it ourside of wasm_bindgen_test
    fn assert_vec_f64_eq(a: &[f64], b: &[f64], tol: f64, msg: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (va - vb).abs() < tol,
                "{}: index {} differs: {} vs {}",
                msg,
                i,
                va,
                vb
            );
        }
    }

    // Apply wasm_bindgen_test to all tests in this module
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn knn_fit_predict() {
        let x =
            DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]])
                .unwrap();
        let y = vec![2, 2, 2, 3, 3];

        let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
        let y_hat = knn.predict(&x).unwrap();

        assert_eq!(5, y_hat.len());
        assert_eq!(y, y_hat);
    }

    #[test]
    fn knn_fit_predict_weighted() {
        let x = DenseMatrix::from_2d_array(&[&[1.], &[2.], &[3.], &[4.], &[5.]]).unwrap();
        let y = vec![2, 2, 2, 3, 3];

        let knn = KNNClassifier::fit(
            &x,
            &y,
            KNNClassifierParameters::default()
                .with_k(5)
                .with_algorithm(KNNAlgorithmName::LinearSearch)
                .with_weight(KNNWeightFunction::Distance),
        )
        .unwrap();

        let y_hat = knn
            .predict(&DenseMatrix::from_2d_array(&[&[4.1]]).unwrap())
            .unwrap();
        assert_eq!(vec![3], y_hat);
    }

    // New 8 tests (2026-03-19)
    #[test]
    fn knn_predict_proba_valid() {
        // Test 1. Test that predict_proba returns valid probability distributions
        let x = DenseMatrix::from_2d_array(&[
            &[1., 2.],
            &[2., 3.],
            &[3., 4.], // class 0
            &[8., 9.],
            &[9., 10.],
            &[10., 11.], // class 1
        ])
        .unwrap();
        let y = vec![0, 0, 0, 1, 1, 1];

        let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
        let proba = knn.predict_proba(&x).unwrap();

        for (i, p) in proba.iter().enumerate() {
            // Probabilities must sum to 1.0 (with floating point tolerance)
            assert!(
                (p.iter().sum::<f64>() - 1.0).abs() < 1e-10,
                "Sample {}: probabilities don't sum to 1",
                i
            );

            // Each probability must be in [0, 1]
            for &prob in p {
                assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Sample {}: probability {} out of range",
                    i,
                    prob
                );
            }
        }
    }

    #[test]
    fn knn_predict_consistent_with_proba() {
        // Test 2. Verify that predict() and predict_proba() return consistent results
        let x = DenseMatrix::from_2d_array(&[
            &[1., 1.],
            &[2., 2.],
            &[3., 3.],
            &[8., 8.],
            &[9., 9.],
            &[10., 10.],
        ])
        .unwrap();
        let y = vec![10, 10, 10, 20, 20, 20];

        let knn = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(3)).unwrap();

        let test = DenseMatrix::from_2d_array(&[&[2.5, 2.5]]).unwrap();

        let pred_class = knn.predict(&test).unwrap();
        let pred_proba = knn.predict_proba(&test).unwrap();

        // Find class index with maximum probability

        let max_proba_idx = pred_proba[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // The class with max probability should match predict() result
        assert_eq!(
            knn.classes()[max_proba_idx],
            pred_class[0],
            "predict() and predict_proba() disagree on class"
        );
    }

    #[test]
    fn knn_predict_proba_linear_vs_cover_tree() {
        // Test 3. Verify both search algorithms produce identical probabilities
        let x = DenseMatrix::from_2d_array(&[
            &[1., 2.],
            &[2., 2.],
            &[3., 3.],
            &[8., 8.],
            &[9., 9.],
            &[10., 10.],
        ])
        .unwrap();
        let y = vec![0, 0, 0, 1, 1, 1];

        let test = DenseMatrix::from_2d_array(&[&[2.5, 2.5], &[9.5, 9.5]]).unwrap();

        let knn_linear = KNNClassifier::fit(
            &x,
            &y,
            KNNClassifierParameters::default()
                .with_algorithm(KNNAlgorithmName::LinearSearch)
                .with_k(3),
        )
        .unwrap();

        let knn_cover = KNNClassifier::fit(
            &x,
            &y,
            KNNClassifierParameters::default()
                .with_algorithm(KNNAlgorithmName::CoverTree)
                .with_k(3),
        )
        .unwrap();

        let proba_linear = knn_linear.predict_proba(&test).unwrap();
        let proba_cover = knn_cover.predict_proba(&test).unwrap();

        // Compare element-wise with tolerance for floating point differences
        for (i, (pl, pc)) in proba_linear.iter().zip(proba_cover.iter()).enumerate() {
            assert_vec_f64_eq(
                pl,
                pc,
                1e-10,
                &format!("Sample {} probability vectors differ", i),
            );
        }
    }

    #[test]
    fn knn_predict_proba_zero_weights_error() {
        // Test 4. Handling of edge case where sum of weights is zero
        let x = DenseMatrix::from_2d_array(&[&[1., 1.], &[1., 1.], &[1., 1.]]).unwrap();
        let y = vec![0, 1, 2]; // Three different classes, identical feature vectors

        let knn = KNNClassifier::fit(
            &x,
            &y,
            KNNClassifierParameters::default()
                .with_k(3)
                .with_weight(KNNWeightFunction::Distance),
        )
        .unwrap();

        let test = DenseMatrix::from_2d_array(&[&[1., 1.]]).unwrap();
        let result = knn.predict_proba(&test);

        // Should either succeed with valid probabilities or return a clear error
        match result {
            Ok(proba) => {
                assert_eq!(proba.len(), 1);
                assert!((proba[0].iter().sum::<f64>() - 1.0).abs() < 1e-10);
            }
            Err(e) => {
                // Error message should be informative
                let err_msg = format!("{:?}", e);
                assert!(
                    err_msg.contains("weight") || err_msg.contains("zero"),
                    "Error message should mention weights or zero sum: {}",
                    err_msg
                );
            }
        }
    }

    #[test]
    fn knn_predict_proba_weight_functions_differ() {
        // Test 5. Verify that different weight functions produce different probabilities
        let x = DenseMatrix::from_2d_array(&[
            &[1., 1.],   // class 0, close
            &[2., 2.],   // class 0, farther
            &[10., 10.], // class 1, far
        ])
        .unwrap();
        let y = vec![0, 0, 1];

        let test = DenseMatrix::from_2d_array(&[&[1.5, 1.5]]).unwrap();

        let knn_uniform = KNNClassifier::fit(
            &x,
            &y,
            KNNClassifierParameters::default()
                .with_k(3)
                .with_weight(KNNWeightFunction::Uniform),
        )
        .unwrap();

        let knn_distance = KNNClassifier::fit(
            &x,
            &y,
            KNNClassifierParameters::default()
                .with_k(3)
                .with_weight(KNNWeightFunction::Distance),
        )
        .unwrap();

        let proba_uniform = knn_uniform.predict_proba(&test).unwrap();
        let proba_distance = knn_distance.predict_proba(&test).unwrap();

        // Uniform and Distance weighting should produce different results (at least one probability value should differ)
        let mut differs = false;
        for (vu, vd) in proba_uniform[0].iter().zip(proba_distance[0].iter()) {
            if (vu - vd).abs() > 1e-10 {
                differs = true;
                break;
            }
        }
        assert!(
            differs,
            "Uniform and Distance weights should produce different probabilities"
        );
    }

    #[test]
    fn knn_predict_proba_extreme_k_values() {
        // Test 6. k=n: with mixed classes, no single class should have probability 1.0
        let x =
            DenseMatrix::from_2d_array(&[&[1., 1.], &[2., 2.], &[3., 3.], &[8., 8.], &[9., 9.]])
                .unwrap();
        let y = vec![0, 0, 1, 1, 1];

        let test = DenseMatrix::from_2d_array(&[&[2.5, 2.5]]).unwrap();

        let knn_kn =
            KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(5)).unwrap();
        let proba_kn = knn_kn.predict_proba(&test).unwrap();
        let max_prob = proba_kn[0].iter().copied().fold(0.0, f64::max);
        assert!(
            max_prob < 1.0 - 1e-10,
            "k=n with mixed classes should not give probability 1.0"
        );
    }

    #[test]
    fn knn_predict_proba_multiclass() {
        // Test 7. Test with more than 2 classes (using i32 labels)
        let x = DenseMatrix::from_2d_array(&[
            &[1., 1.],
            &[1.5, 1.5], // class 10
            &[4., 4.],
            &[4.5, 4.5], // class 20
            &[8., 8.],
            &[8.5, 8.5], // class 30
        ])
        .unwrap();
        let y = vec![10, 10, 20, 20, 30, 30];

        let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
        let test = DenseMatrix::from_2d_array(&[&[4.2, 4.2]]).unwrap();

        let proba = knn.predict_proba(&test).unwrap();

        assert_eq!(proba[0].len(), 3, "Should have 3 class probabilities");
        assert!((proba[0].iter().sum::<f64>() - 1.0).abs() < 1e-10);

        // Point is closest to class 20, so its probability should be highest
        let max_idx = proba[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(knn.classes()[max_idx], 20);
    }

    #[test]
    fn knn_predict_proba_batch() {
        // Test 8. Batch prediction (multiple samples at once)
        let x = DenseMatrix::from_2d_array(&[
            &[1., 1.],
            &[2., 2.],
            &[3., 3.],
            &[8., 8.],
            &[9., 9.],
            &[10., 10.],
        ])
        .unwrap();
        let y = vec![0, 0, 0, 1, 1, 1];

        let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();

        // Query multiple points simultaneously
        let test = DenseMatrix::from_2d_array(&[
            &[1.5, 1.5], // closer to class 0
            &[9.5, 9.5], // closer to class 1
            &[5., 5.],   // middle point
        ])
        .unwrap();

        let proba = knn.predict_proba(&test).unwrap();

        // Check 1
        assert_eq!(proba.len(), 3, "Should return probabilities for 3 samples");

        // Check 2: Each row must be a valid probability distribution
        for p in &proba {
            assert_eq!(p.len(), 2); // 2 classes
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        }

        // Check 3 (Intuitive checks): first sample favors class 0, second favors class 1
        assert!(
            proba[0][0] > proba[0][1],
            "First sample should favor class 0"
        );
        assert!(
            proba[1][1] > proba[1][0],
            "Second sample should favor class 1"
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x =
            DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.], &[7., 8.], &[9., 10.]])
                .unwrap();
        let y = vec![2, 2, 2, 3, 3];

        let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
        let deserialized_knn = bincode::deserialize(&bincode::serialize(&knn).unwrap()).unwrap();

        assert_eq!(knn, deserialized_knn);
    }
}
