//! # Extra Trees Regressor
//! An Extra-Trees (Extremely Randomized Trees) regressor is an ensemble learning method that fits multiple randomized
//! decision trees on the dataset and averages their predictions to improve accuracy and control over-fitting.
//!
//! It is similar to a standard Random Forest, but introduces more randomness in the way splits are chosen, which can
//! reduce the variance of the model and often make the training process faster.
//!
//! The two key differences from a standard Random Forest are:
//! 1. It uses the whole original dataset to build each tree instead of bootstrap samples.
//! 2. When splitting a node, it chooses a random split point for each feature, rather than the most optimal one.
//!
//! See [ensemble models](../index.html) for more details.
//!
//! Bigger number of estimators in general improves performance of the algorithm with an increased cost of training time.
//! The random sample of _m_ predictors is typically set to be \\(\sqrt{p}\\) from the full set of _p_ predictors.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::ensemble::extra_trees_regressor::*;
//!
//! // Longley dataset ([https://www.statsmodels.org/stable/datasets/generated/longley.html](https://www.statsmodels.org/stable/datasets/generated/longley.html))
//! let x = DenseMatrix::from_2d_array(&[
//!     &[234.289, 235.6, 159., 107.608, 1947., 60.323],
//!     &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
//!     &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
//!     &[284.599, 335.1, 165., 110.929, 1950., 61.187],
//!     &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
//!     &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
//!     &[365.385, 187., 354.7, 115.094, 1953., 64.989],
//!     &[363.112, 357.8, 335., 116.219, 1954., 63.761],
//!     &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
//!     &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
//!     &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
//!     &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
//!     &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
//!     &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
//!     &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
//!     &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
//! ]).unwrap();
//! let y = vec![
//!     83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2,
//!     104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9
//! ];
//!
//! let regressor = ExtraTreesRegressor::fit(&x, &y, Default::default()).unwrap();
//!
//! let y_hat = regressor.predict(&x).unwrap(); // use the same data for prediction
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

use std::default::Default;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::ensemble::base_forest_regressor::{BaseForestRegressor, BaseForestRegressorParameters};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;
use crate::tree::base_tree_regressor::Splitter;

/// Validates the sample weights
fn validate_sample_weights(sample_weights: &[f64], n_rows: usize) -> Result<(), Failed> {
    if sample_weights.len() != n_rows {
        return Err(Failed::fit(
            "Number of sample weights must equal number of rows in x",
        ));
    }
    if sample_weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(Failed::fit(
            "Sample weights must be finite and non-negative",
        ));
    }
    if sample_weights.iter().sum::<f64>() <= 0.0 {
        return Err(Failed::fit("Sum of sample weights must be positive"));
    }
    Ok(())
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// Parameters of the Extra Trees Regressor
/// Some parameters here are passed directly into base estimator.
#[must_use]
pub struct ExtraTreesRegressorParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Tree max depth. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub max_depth: Option<u16>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_leaf: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to split an internal node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_split: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The number of trees in the forest.
    pub n_trees: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of random sample of predictors to use as split candidates.
    pub m: Option<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub keep_samples: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Seed used for bootstrap sampling and feature selection for each tree.
    pub seed: u64,
}

/// Extra Trees Regressor
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct ExtraTreesRegressor<
    TX: Number + FloatNumber + PartialOrd,
    TY: Number,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    forest_regressor: Option<BaseForestRegressor<TX, TY, X, Y>>,
}

impl ExtraTreesRegressorParameters {
    /// Tree max depth. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub fn with_max_depth(mut self, max_depth: u16) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }
    /// The minimum number of samples required to split an internal node. See [Decision Tree Classifier](../../tree/decision_tree_classifier/index.html)
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }
    /// The number of trees in the forest.
    pub fn with_n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }
    /// Number of random sample of predictors to use as split candidates.
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = Some(m);
        self
    }

    /// Whether to keep samples used for tree generation. This is required for OOB prediction.
    pub fn with_keep_samples(mut self, keep_samples: bool) -> Self {
        self.keep_samples = keep_samples;
        self
    }

    /// Seed used for bootstrap sampling and feature selection for each tree.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}
impl Default for ExtraTreesRegressorParameters {
    fn default() -> Self {
        ExtraTreesRegressorParameters {
            max_depth: Option::None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            n_trees: 10,
            m: Option::None,
            keep_samples: false,
            seed: 0,
        }
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, ExtraTreesRegressorParameters> for ExtraTreesRegressor<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            forest_regressor: Option::None,
        }
    }

    fn fit(x: &X, y: &Y, parameters: ExtraTreesRegressorParameters) -> Result<Self, Failed> {
        ExtraTreesRegressor::fit(x, y, parameters)
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    Predictor<X, Y> for ExtraTreesRegressor<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number + FloatNumber + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    ExtraTreesRegressor<TX, TY, X, Y>
{
    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: ExtraTreesRegressorParameters,
    ) -> Result<ExtraTreesRegressor<TX, TY, X, Y>, Failed> {
        Self::fit_inner(x, y, None, parameters)
    }

    /// Build a forest of trees from the training set.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target class values
    /// * `sample_weights`: sample_weights to use during fitting
    pub fn fit_with_weights(
        x: &X,
        y: &Y,
        sample_weights: &[f64],
        parameters: ExtraTreesRegressorParameters,
    ) -> Result<ExtraTreesRegressor<TX, TY, X, Y>, Failed> {
        validate_sample_weights(sample_weights, x.shape().0)?;
        Self::fit_inner(x, y, Some(sample_weights), parameters)
    }

    fn fit_inner(
        x: &X,
        y: &Y,
        sample_weights: Option<&[f64]>,
        parameters: ExtraTreesRegressorParameters,
    ) -> Result<ExtraTreesRegressor<TX, TY, X, Y>, Failed> {
        let regressor_params = BaseForestRegressorParameters {
            max_depth: parameters.max_depth,
            min_samples_leaf: parameters.min_samples_leaf,
            min_samples_split: parameters.min_samples_split,
            n_trees: parameters.n_trees,
            m: parameters.m,
            keep_samples: parameters.keep_samples,
            seed: parameters.seed,
            bootstrap: false,
            splitter: Splitter::Random,
        };
        let forest_regressor = BaseForestRegressor::fit(x, y, sample_weights, regressor_params)?;

        Ok(ExtraTreesRegressor {
            forest_regressor: Some(forest_regressor),
        })
    }

    /// Predict class for `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        match &self.forest_regressor {
            Some(forest) => forest.predict(x),
            None => Err(Failed::predict(
                "'fit' should be called before calling 'predict'",
            )),
        }
    }

    /// Predict OOB classes for `x`. `x` is expected to be equal to the dataset used in training.
    pub fn predict_oob(&self, x: &X) -> Result<Y, Failed> {
        match &self.forest_regressor {
            Some(forest) => forest.predict_oob(x),
            None => Err(Failed::predict(
                "'fit' should be called before calling 'predict'",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::metrics::mean_squared_error;

    #[test]
    fn test_extra_trees_regressor_fit_predict() {
        // Use a simpler, more predictable dataset for unit testing.
        let x = DenseMatrix::from_2d_array(&[
            &[1., 2.],
            &[3., 4.],
            &[5., 6.],
            &[7., 8.],
            &[9., 10.],
            &[11., 12.],
            &[13., 14.],
            &[15., 16.],
        ])
        .unwrap();
        let y = vec![1., 2., 3., 4., 5., 6., 7., 8.];

        let parameters = ExtraTreesRegressorParameters::default()
            .with_n_trees(100)
            .with_seed(42);

        let regressor = ExtraTreesRegressor::fit(&x, &y, parameters).unwrap();
        let y_hat = regressor.predict(&x).unwrap();

        assert_eq!(y_hat.len(), y.len());
        // A basic check to ensure the model is learning something.
        // The error should be significantly less than the variance of y.
        let mse = mean_squared_error(&y, &y_hat);
        // With this simple dataset, the error should be very low.
        assert!(mse < 1.0);
    }

    #[test]
    fn fit_with_weights_validates_weights() {
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator((0..6).map(|i| i as f64), 3, 2, 0);
        let y = vec![1.0_f64, 2.0, 3.0];
        let parameters = ExtraTreesRegressorParameters::default()
            .with_n_trees(5)
            .with_seed(42);

        // Valid weights: a zero weight is permitted when the sum is positive
        for weights in [vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.5]] {
            assert!(
                ExtraTreesRegressor::fit_with_weights(&x, &y, &weights, parameters.clone()).is_ok(),
                "weights: {weights:?}"
            );
        }

        let wrong_length = "Number of sample weights must equal number of rows in x";
        let not_finite_or_negative = "Sample weights must be finite and non-negative";
        let zero_sum = "Sum of sample weights must be positive";
        let cases: Vec<(Vec<f64>, &str)> = vec![
            (vec![], wrong_length),
            (vec![1.0, 2.0], wrong_length),
            (vec![1.0, 2.0, 3.0, 4.0], wrong_length),
            (vec![1.0, -1.0, 3.0], not_finite_or_negative),
            (vec![1.0, f64::NAN, 3.0], not_finite_or_negative),
            (vec![1.0, f64::INFINITY, 3.0], not_finite_or_negative),
            (vec![0.0, 0.0, 0.0], zero_sum),
        ];
        for (weights, msg) in cases {
            let result =
                ExtraTreesRegressor::fit_with_weights(&x, &y, &weights, parameters.clone());
            assert_eq!(result.err(), Some(Failed::fit(msg)), "weights: {weights:?}");
        }
    }

    #[test]
    fn test_fit_predict_higher_dims() {
        // Dataset with 10 features, but y is only dependent on the 3rd feature (index 2).
        let x = DenseMatrix::from_2d_array(&[
            // The 3rd column is the important one. The rest are noise.
            &[0., 0., 10., 5., 8., 1., 4., 9., 2., 7.],
            &[0., 0., 20., 1., 2., 3., 4., 5., 6., 7.],
            &[0., 0., 30., 7., 6., 5., 4., 3., 2., 1.],
            &[0., 0., 40., 9., 2., 4., 6., 8., 1., 3.],
            &[0., 0., 55., 3., 1., 8., 6., 4., 2., 9.],
            &[0., 0., 65., 2., 4., 7., 5., 3., 1., 8.],
        ])
        .unwrap();
        let y = vec![10., 20., 30., 40., 55., 65.];

        let parameters = ExtraTreesRegressorParameters::default()
            .with_n_trees(100)
            .with_seed(42);

        let regressor = ExtraTreesRegressor::fit(&x, &y, parameters).unwrap();
        let y_hat = regressor.predict(&x).unwrap();

        assert_eq!(y_hat.len(), y.len());

        let mse = mean_squared_error(&y, &y_hat);

        // The model should be able to learn this simple relationship perfectly,
        // ignoring the noise features. The MSE should be very low.
        assert!(mse < 1.0);
    }

    #[test]
    fn test_reproducibility() {
        let x = DenseMatrix::from_2d_array(&[
            &[1., 2.],
            &[3., 4.],
            &[5., 6.],
            &[7., 8.],
            &[9., 10.],
            &[11., 12.],
        ])
        .unwrap();
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let params = ExtraTreesRegressorParameters::default().with_seed(42);

        let regressor1 = ExtraTreesRegressor::fit(&x, &y, params.clone()).unwrap();
        let y_hat1 = regressor1.predict(&x).unwrap();

        let regressor2 = ExtraTreesRegressor::fit(&x, &y, params.clone()).unwrap();
        let y_hat2 = regressor2.predict(&x).unwrap();

        assert_eq!(y_hat1, y_hat2);
    }

    #[test]
    fn fit_with_weights_predicts_approx_weighted_mean() {
        // 20 rows, 1 feature. Stumps (max_depth = 0): each tree predicts the
        // weighted mean of y over its bootstrap sample (which is also weighted)
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator((0..20).map(|i| i as f64), 20, 1, 0);
        let y: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 10.0 }).collect();
        // Rows with y = 10 have weight 9, rows with y = 0 have weight 1.
        let sample_weights: Vec<f64> = (0..20).map(|i| if i < 10 { 1.0 } else { 9.0 }).collect();

        let parameters = ExtraTreesRegressorParameters::default()
            .with_max_depth(0)
            .with_n_trees(50)
            .with_seed(42);

        let forest =
            ExtraTreesRegressor::fit_with_weights(&x, &y, &sample_weights, parameters.clone())
                .expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");

        // weighted mean is (10.0 * 0 + 90*10) / 100 = 9
        for p in y_hat.iter() {
            assert!(
                (p - 9.0f64).abs() < 1e-9,
                "expected value very close to 9, got {p}"
            );
        }

        // Without weights, the predicted value should be close to 5
        let forest = ExtraTreesRegressor::fit(&x, &y, parameters).expect("Fit should work");
        let y_hat = forest.predict(&x).expect("Predict should work");

        for p in y_hat.iter() {
            assert!(
                (p - 5.0).abs() < 1e-9,
                "expected value very close to 5, got {p}"
            );
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_without_fit_should_not_panic() {
        let forest: ExtraTreesRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            ExtraTreesRegressor::new();
        let x = DenseMatrix::from_2d_array(&[&[1.0f64]]).expect("Construction of x should work");
        let yhat = forest.predict(&x);
        assert!(yhat.is_err());
        let msg = "'fit' should be called before calling 'predict'";
        assert_eq!(yhat.err(), Some(Failed::predict(msg)));
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn predict_oob_without_fit_should_not_panic() {
        let forest: ExtraTreesRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            ExtraTreesRegressor::new();
        let x = DenseMatrix::from_2d_array(&[&[1.0f64]]).expect("Construction of x should work");
        let yhat = forest.predict_oob(&x);
        assert!(yhat.is_err());
        let msg = "'fit' should be called before calling 'predict'";
        assert_eq!(yhat.err(), Some(Failed::predict(msg)));
    }

    mod sklearn_parity {
        use super::*;
        // sklearn parity tests.
        //
        // smartcore and numpy use different RNGs.
        // However, with many trees, both forests converge to the same bagged
        // predictor. Thus we compare the predictions within a tolerance.
        //
        // Reference: sklearn 1.9.1, numpy 2.4.6. Each reference value is the mean of 10 sklearn
        // runs (random_state = 0..10). The tolerance is approximately 4 x the largest standard
        // deviation of one sklearn run, for each row.
        //
        // ```python
        // rng = np.random.default_rng(0)
        // x = np.round(rng.uniform(-1, 1, (40, 4)), 4)
        // y = np.round(x[:, 0] * x[:, 1] + np.sin(3 * x[:, 2]) + 0.1 * rng.normal(size=40), 4)
        // x_probe = np.round(rng.uniform(-1, 1, (10, 4)), 4)
        // for max_features in [1.0, 2]:
        //   train, probe = [], []
        //   for seed in range(N_SEEDS):
        //     rf = ExtraTreesRegressor(
        //         n_estimators=N_TREES,
        //         max_features=max_features,
        //         max_depth=None,
        //         min_samples_leaf=5, # so that the predictions are not equal to the training values
        //         min_samples_split=2,
        //         bootstrap=False,
        //         random_state=seed,
        //         n_jobs=-1,
        //      ).fit(x, y)
        //     train.append(rf.predict(x))
        //     probe.append(rf.predict(x_probe))
        // print(f"\nExtra Trees. === max_features={max_features}")
        // for name, runs in [("train", train), ("probe", probe)]:
        //     runs = np.array(runs)
        //     mean = runs.mean(axis=0)
        //     max_std = runs.std(axis=0, ddof=1).max()
        //     max_dev = np.abs(runs - mean).max()
        //     print(f"{name}: max_std={max_std:.5f} max_dev={max_dev:.5f}")
        //     print(f"{name}_ref:", rust_vec(mean))
        // ```
        //

        fn sklearn_parity_train_data() -> (DenseMatrix<f64>, Vec<f64>) {
            let x = DenseMatrix::from_2d_array(&[
                &[0.2739, -0.4604, -0.9181, -0.9669],
                &[0.6265, 0.8255, 0.2133, 0.459],
                &[0.0872, 0.8701, 0.6317, -0.9945],
                &[0.7148, -0.9328, 0.4593, -0.6487],
                &[0.7264, 0.0829, -0.4006, -0.1546],
                &[-0.9434, -0.7514, 0.3412, 0.2944],
                &[0.2308, -0.2326, 0.9944, 0.9617],
                &[0.3711, 0.3009, 0.3769, -0.2222],
                &[-0.7298, 0.443, 0.0507, -0.3795],
                &[-0.0283, 0.779, 0.8681, -0.2844],
                &[0.1431, -0.3563, 0.1886, -0.3242],
                &[-0.2168, 0.7805, -0.5457, 0.2464],
                &[-0.832, 0.6653, 0.5742, -0.5213],
                &[0.753, -0.8829, -0.3278, -0.6994],
                &[-0.0993, 0.5926, -0.5387, -0.896],
                &[-0.1909, -0.603, -0.8185, 0.1607],
                &[-0.4026, 0.344, -0.601, 0.8842],
                &[-0.2698, -0.789, 0.2582, 0.8543],
                &[-0.1192, 0.9092, -0.0002, -0.1495],
                &[0.2404, 0.9902, 0.8979, -0.0799],
                &[0.5155, -0.0052, 0.0586, 0.5716],
                &[-0.1707, 0.469, 0.4223, 0.8641],
                &[-0.7701, 0.458, 0.8548, 0.9359],
                &[-0.9706, 0.7273, 0.9624, 0.9144],
                &[-0.7025, 0.9453, 0.7799, 0.6447],
                &[-0.04, -0.5353, 0.6038, 0.8471],
                &[-0.4677, 0.0779, -0.1145, 0.862],
                &[-0.919, 0.464, 0.2287, -0.9433],
                &[0.4384, -0.968, 0.5159, 0.0255],
                &[0.8582, -0.8678, 0.6826, -0.8666],
                &[-0.3114, -0.1394, 0.9321, 0.1245],
                &[-0.4823, -0.5166, 0.7762, -0.5483],
                &[-0.7509, -0.4233, 0.1722, 0.1082],
                &[0.6194, 0.121, -0.4232, -0.1742],
                &[0.6362, 0.253, 0.9182, -0.2612],
                &[0.1052, 0.1878, 0.6966, -0.7091],
                &[-0.187, 0.8199, -0.9139, 0.6454],
                &[-0.1692, 0.6596, -0.9801, -0.2699],
                &[-0.8427, 0.3052, -0.4523, 0.4053],
                &[0.8876, -0.7464, 0.7296, -0.8811],
            ])
            .unwrap();
            let y = vec![
                -0.5144, 0.9957, 0.7839, 0.3660, -0.9022, 1.5099, 0.0804, 1.1980, -0.1768, 0.4984,
                0.3364, -1.0023, 0.5267, -1.3905, -1.0531, -0.4267, -1.0746, 0.9736, -0.1242,
                0.5237, 0.2751, 0.6806, 0.1690, -0.4747, -0.0497, 1.0539, -0.3933, 0.1634, 0.6273,
                0.0960, 0.5208, 1.0106, 0.7643, -1.0745, 0.4076, 0.9968, -0.5477, -0.3399, -1.0701,
                0.0243,
            ];
            (x, y)
        }

        fn sklearn_parity_probe_data() -> DenseMatrix<f64> {
            DenseMatrix::from_2d_array(&[
                &[-0.3606, -0.625, 0.3451, -0.6098],
                &[0.1554, 0.2045, 0.9248, -0.8555],
                &[-0.0001, 0.4882, -0.6455, -0.2239],
                &[-0.8742, 0.4518, -0.8245, -0.2098],
                &[0.747, -0.0554, 0.8252, 0.5318],
                &[0.8306, -0.7452, -0.8529, -0.8593],
                &[0.7377, 0.2681, -0.0069, -0.6729],
                &[0.3475, -0.364, 0.4218, -0.0793],
                &[0.0149, 0.5793, -0.8145, 0.1575],
                &[-0.6055, 0.6163, -0.0223, 0.9774],
            ])
            .unwrap()
        }

        fn sklearn_sample_weights() -> Vec<f64> {
            (0..40).into_iter().map(|i| ((i % 4) + 1) as f64).collect()
        }

        fn assert_close_to_sklearn(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
            assert_eq!(actual.len(), expected.len(), "{label}: length");
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() <= tol,
                    "{label}, row {i}: smartcore {a}, sklearn {e}, tol {tol}"
                );
            }
        }

        /// Fits the extra trees regressor with 2000 trees and compares the predictions with sklearn.
        fn check_sklearn_parity(
            m: usize,
            train_ref: &[f64],
            probe_ref: &[f64],
            tol: f64,
            sample_weights: Option<&[f64]>,
        ) {
            let (x, y) = sklearn_parity_train_data();
            let x_probe = sklearn_parity_probe_data();

            // Use `min_samples_leaf`= 5 to prevent the tree from predicting all training examples correct
            let parameters = ExtraTreesRegressorParameters::default()
                .with_n_trees(2000)
                .with_m(m)
                .with_min_samples_leaf(5)
                .with_min_samples_split(2)
                .with_keep_samples(true)
                .with_seed(42);
            let forest = match sample_weights {
                None => ExtraTreesRegressor::fit(&x, &y, parameters).unwrap(),
                Some(sample_weights) => {
                    ExtraTreesRegressor::fit_with_weights(&x, &y, sample_weights, parameters)
                        .unwrap()
                }
            };

            let y_hat: Vec<f64> = forest.predict(&x).unwrap();
            assert_close_to_sklearn(&y_hat, train_ref, tol, "train");

            let y_hat_probe: Vec<f64> = forest.predict(&x_probe).unwrap();
            assert_close_to_sklearn(&y_hat_probe, probe_ref, tol, "probe");
        }

        #[test]
        fn sklearn_parity_all_features() {
            /*
             * train: max_std=0.01081 max_dev=0.02053
             * probe: max_std=0.01127 max_dev=0.02510
             */
            let train_ref = [
                -0.514400, 0.350098, 0.500618, 0.370298, -0.582169, 0.630296, 0.362747, 0.528070,
                0.027928, 0.409103, 0.376290, -0.672295, 0.408925, -0.406530, -0.656713, -0.433254,
                -0.656863, 0.572460, -0.005012, 0.398251, 0.134855, 0.457975, 0.232323, 0.203198,
                0.255614, 0.580654, -0.171093, 0.266703, 0.538009, 0.339515, 0.425172, 0.512873,
                0.452168, -0.597154, 0.382057, 0.522851, -0.636264, -0.652577, -0.614220, 0.327194,
            ];
            let probe_ref = [
                0.538464, 0.434785, -0.641271, -0.638714, 0.324906, -0.499588, -0.042825, 0.545153,
                -0.637955, -0.073064,
            ];
            check_sklearn_parity(4, &train_ref, &probe_ref, 0.045, None);
        }

        #[test]
        fn sklearn_parity_with_weights() {
            /*
             * train: max_std=0.01332 max_dev=0.02559
             * probe: max_std=0.01166 max_dev=0.01867
             */
            let train_ref = [
                -0.494622, 0.335181, 0.532771, 0.398018, -0.535612, 0.545873, 0.337625, 0.588692,
                0.030596, 0.431604, 0.376213, -0.686286, 0.400965, -0.400551, -0.658719, -0.481374,
                -0.649386, 0.477041, -0.017966, 0.424634, 0.126455, 0.419039, 0.196019, 0.148182,
                0.238049, 0.512727, -0.195074, 0.255755, 0.513608, 0.384085, 0.438159, 0.544918,
                0.374897, -0.552449, 0.433162, 0.583000, -0.636756, -0.643531, -0.627507, 0.372639,
            ];

            let probe_ref = [
                0.530162, 0.489443, -0.611812, -0.626195, 0.352540, -0.502707, 0.001104, 0.551536,
                -0.621779, -0.120032,
            ];

            check_sklearn_parity(
                4,
                &train_ref,
                &probe_ref,
                0.05,
                Some(&sklearn_sample_weights()),
            );
        }
    }
}
