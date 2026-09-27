//! # Decision Tree Regressor
//!
//! The process of building a decision tree can be simplified to these two steps:
//!
//! 1. Divide the predictor space \\(X\\) into K distinct and non-overlapping regions, \\(R_1, R_2, ..., R_K\\).
//! 1. For every observation that falls into the region \\(R_k\\), we make the same prediction, which is simply the mean of the response values for the training observations in \\(R_k\\).
//!
//! Regions \\(R_1, R_2, ..., R_K\\) are build in such a way that minimizes the residual sum of squares (RSS) given by
//!
//! \\[RSS = \sum_{k=1}^K\sum_{i \in R_k} (y_i - \hat{y}_{Rk})^2\\]
//!
//! where \\(\hat{y}_{Rk}\\) is the mean response for the training observations withing region _k_.
//!
//! `smartcore` uses recursive binary splitting approach to build \\(R_1, R_2, ..., R_K\\) regions. The approach begins at the top of the tree and then successively splits the predictor space
//! one predictor at a time. At each step of the tree-building process, the best split is made at that particular step, rather than looking ahead and picking a split that will lead to a better
//! tree in some future step.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::tree::decision_tree_regressor::*;
//!
//! // Longley dataset (https://www.statsmodels.org/stable/datasets/generated/longley.html)
//! let x = DenseMatrix::from_2d_array(&[
//!             &[234.289, 235.6, 159., 107.608, 1947., 60.323],
//!             &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
//!             &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
//!             &[284.599, 335.1, 165., 110.929, 1950., 61.187],
//!             &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
//!             &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
//!             &[365.385, 187., 354.7, 115.094, 1953., 64.989],
//!             &[363.112, 357.8, 335., 116.219, 1954., 63.761],
//!             &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
//!             &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
//!             &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
//!             &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
//!             &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
//!             &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
//!             &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
//!             &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
//!        ]).unwrap();
//! let y: Vec<f64> = vec![
//!             83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0,
//!             101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9,
//!        ];
//!
//! let tree = DecisionTreeRegressor::fit(&x, &y, Default::default()).unwrap();
//!
//! let y_hat = tree.predict(&x).unwrap(); // use the same data for prediction
//! ```
//!
//! ## References:
//!
//! * ["Classification and regression trees", Breiman, L, Friedman, J H, Olshen, R A, and Stone, C J, 1984](https://www.sciencebase.gov/catalog/item/545d07dfe4b0ba8303f728c1)
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., Chapter 8](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

use std::default::Default;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::base_tree_regressor::{BaseTreeRegressor, BaseTreeRegressorParameters, Splitter};
use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::basenum::Number;

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
/// Parameters of Regression Tree
#[must_use]
pub struct DecisionTreeRegressorParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// The maximum depth of the tree.
    pub max_depth: Option<u16>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to be at a leaf node.
    pub min_samples_leaf: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to split an internal node.
    pub min_samples_split: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Controls the randomness of the estimator
    pub seed: Option<u64>,
}

/// Regression Tree
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct DecisionTreeRegressor<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
{
    tree_regressor: Option<BaseTreeRegressor<TX, TY, X, Y>>,
}

impl DecisionTreeRegressorParameters {
    /// The maximum depth of the tree.
    pub fn with_max_depth(mut self, max_depth: u16) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
    /// The minimum number of samples required to be at a leaf node.
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }
    /// The minimum number of samples required to split an internal node.
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }
}

impl Default for DecisionTreeRegressorParameters {
    fn default() -> Self {
        DecisionTreeRegressorParameters {
            max_depth: Option::None,
            min_samples_leaf: 1,
            min_samples_split: 2,
            seed: Option::None,
        }
    }
}

/// DecisionTreeRegressor grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
#[must_use]
pub struct DecisionTreeRegressorSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Tree max depth. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub max_depth: Vec<Option<u16>>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to be at a leaf node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_leaf: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// The minimum number of samples required to split an internal node. See [Decision Tree Regressor](../../tree/decision_tree_regressor/index.html)
    pub min_samples_split: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Controls the randomness of the estimator
    pub seed: Vec<Option<u64>>,
}

/// DecisionTreeRegressor grid search iterator
pub struct DecisionTreeRegressorSearchParametersIterator {
    decision_tree_regressor_search_parameters: DecisionTreeRegressorSearchParameters,
    current_max_depth: usize,
    current_min_samples_leaf: usize,
    current_min_samples_split: usize,
    current_seed: usize,
}

impl IntoIterator for DecisionTreeRegressorSearchParameters {
    type Item = DecisionTreeRegressorParameters;
    type IntoIter = DecisionTreeRegressorSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        DecisionTreeRegressorSearchParametersIterator {
            decision_tree_regressor_search_parameters: self,
            current_max_depth: 0,
            current_min_samples_leaf: 0,
            current_min_samples_split: 0,
            current_seed: 0,
        }
    }
}

impl Iterator for DecisionTreeRegressorSearchParametersIterator {
    type Item = DecisionTreeRegressorParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_max_depth
            == self
                .decision_tree_regressor_search_parameters
                .max_depth
                .len()
            && self.current_min_samples_leaf
                == self
                    .decision_tree_regressor_search_parameters
                    .min_samples_leaf
                    .len()
            && self.current_min_samples_split
                == self
                    .decision_tree_regressor_search_parameters
                    .min_samples_split
                    .len()
            && self.current_seed == self.decision_tree_regressor_search_parameters.seed.len()
        {
            return None;
        }

        let next = DecisionTreeRegressorParameters {
            max_depth: self.decision_tree_regressor_search_parameters.max_depth
                [self.current_max_depth],
            min_samples_leaf: self
                .decision_tree_regressor_search_parameters
                .min_samples_leaf[self.current_min_samples_leaf],
            min_samples_split: self
                .decision_tree_regressor_search_parameters
                .min_samples_split[self.current_min_samples_split],
            seed: self.decision_tree_regressor_search_parameters.seed[self.current_seed],
        };

        if self.current_max_depth + 1
            < self
                .decision_tree_regressor_search_parameters
                .max_depth
                .len()
        {
            self.current_max_depth += 1;
        } else if self.current_min_samples_leaf + 1
            < self
                .decision_tree_regressor_search_parameters
                .min_samples_leaf
                .len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf += 1;
        } else if self.current_min_samples_split + 1
            < self
                .decision_tree_regressor_search_parameters
                .min_samples_split
                .len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split += 1;
        } else if self.current_seed + 1 < self.decision_tree_regressor_search_parameters.seed.len()
        {
            self.current_max_depth = 0;
            self.current_min_samples_leaf = 0;
            self.current_min_samples_split = 0;
            self.current_seed += 1;
        } else {
            self.current_max_depth += 1;
            self.current_min_samples_leaf += 1;
            self.current_min_samples_split += 1;
            self.current_seed += 1;
        }

        Some(next)
    }
}

impl Default for DecisionTreeRegressorSearchParameters {
    fn default() -> Self {
        let default_params = DecisionTreeRegressorParameters::default();

        DecisionTreeRegressorSearchParameters {
            max_depth: vec![default_params.max_depth],
            min_samples_leaf: vec![default_params.min_samples_leaf],
            min_samples_split: vec![default_params.min_samples_split],
            seed: vec![default_params.seed],
        }
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for DecisionTreeRegressor<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        self.tree_regressor == other.tree_regressor
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, DecisionTreeRegressorParameters>
    for DecisionTreeRegressor<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            tree_regressor: None,
        }
    }

    fn fit(x: &X, y: &Y, parameters: DecisionTreeRegressorParameters) -> Result<Self, Failed> {
        DecisionTreeRegressor::fit(x, y, parameters)
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> Predictor<X, Y>
    for DecisionTreeRegressor<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    DecisionTreeRegressor<TX, TY, X, Y>
{
    /// Build a decision tree regressor from the training data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target values
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: DecisionTreeRegressorParameters,
    ) -> Result<DecisionTreeRegressor<TX, TY, X, Y>, Failed> {
        Self::fit_inner(x, y, None, parameters)
    }

    /// Build a decision tree regressor from the training data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - the target values
    /// * `sample_weights` - weights to use during fitting
    pub fn fit_with_weights(
        x: &X,
        y: &Y,
        sample_weights: &[f64],
        parameters: DecisionTreeRegressorParameters,
    ) -> Result<DecisionTreeRegressor<TX, TY, X, Y>, Failed> {
        validate_sample_weights(sample_weights, x.shape().0)?;
        Self::fit_inner(x, y, Some(sample_weights), parameters)
    }

    fn fit_inner(
        x: &X,
        y: &Y,
        sample_weights: Option<&[f64]>,
        parameters: DecisionTreeRegressorParameters,
    ) -> Result<DecisionTreeRegressor<TX, TY, X, Y>, Failed> {
        let tree_parameters = BaseTreeRegressorParameters {
            max_depth: parameters.max_depth,
            min_samples_leaf: parameters.min_samples_leaf,
            min_samples_split: parameters.min_samples_split,
            seed: parameters.seed,
            splitter: Splitter::Best,
        };
        let tree = BaseTreeRegressor::fit_inner(x, y, sample_weights, tree_parameters)?;
        Ok(Self {
            tree_regressor: Some(tree),
        })
    }

    /// Predict regression value for `x`.
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.tree_regressor.as_ref().unwrap().predict(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn search_parameters() {
        let parameters = DecisionTreeRegressorSearchParameters {
            max_depth: vec![Some(10), Some(100)],
            min_samples_split: vec![1, 2],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.max_depth, Some(10));
        assert_eq!(next.min_samples_split, 1);
        let next = iter.next().unwrap();
        assert_eq!(next.max_depth, Some(100));
        assert_eq!(next.min_samples_split, 1);
        let next = iter.next().unwrap();
        assert_eq!(next.max_depth, Some(10));
        assert_eq!(next.min_samples_split, 2);
        let next = iter.next().unwrap();
        assert_eq!(next.max_depth, Some(100));
        assert_eq!(next.min_samples_split, 2);
        assert!(iter.next().is_none());
    }

    #[test]
    fn fit_with_weights_validates_weights() {
        let x: DenseMatrix<f64> = DenseMatrix::from_iterator((0..6).map(|i| i as f64), 3, 2, 0);
        let y = vec![1.0_f64, 2.0, 3.0];
        let parameters = DecisionTreeRegressorParameters::default();

        // Valid weights: a zero weight is permitted when the sum is positive
        for weights in [vec![1.0, 2.0, 3.0], vec![0.0, 0.0, 0.5]] {
            assert!(
                DecisionTreeRegressor::fit_with_weights(&x, &y, &weights, parameters.clone())
                    .is_ok(),
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
                DecisionTreeRegressor::fit_with_weights(&x, &y, &weights, parameters.clone());
            assert_eq!(result.err(), Some(Failed::fit(msg)), "weights: {weights:?}");
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fit_longley() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159., 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165., 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            &[365.385, 187., 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335., 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ])
        .unwrap();
        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let y_hat = DecisionTreeRegressor::fit(&x, &y, Default::default())
            .and_then(|t| t.predict(&x))
            .unwrap();

        for i in 0..y_hat.len() {
            assert!((y_hat[i] - y[i]).abs() < 0.1);
        }

        let expected_y = [
            87.3, 87.3, 87.3, 87.3, 98.9, 98.9, 98.9, 98.9, 98.9, 107.9, 107.9, 107.9, 114.85,
            114.85, 114.85, 114.85,
        ];
        let y_hat = DecisionTreeRegressor::fit(
            &x,
            &y,
            DecisionTreeRegressorParameters {
                max_depth: Option::None,
                min_samples_leaf: 2,
                min_samples_split: 6,
                seed: Option::None,
            },
        )
        .and_then(|t| t.predict(&x))
        .unwrap();

        for i in 0..y_hat.len() {
            assert!((y_hat[i] - expected_y[i]).abs() < 0.1);
        }

        let expected_y = [
            83.0, 88.35, 88.35, 89.5, 97.15, 97.15, 99.5, 99.5, 101.2, 104.6, 109.6, 109.6, 113.4,
            113.4, 116.30, 116.30,
        ];
        let y_hat = DecisionTreeRegressor::fit(
            &x,
            &y,
            DecisionTreeRegressorParameters {
                max_depth: Option::None,
                min_samples_leaf: 1,
                min_samples_split: 3,
                seed: Option::None,
            },
        )
        .and_then(|t| t.predict(&x))
        .unwrap();

        for i in 0..y_hat.len() {
            assert!((y_hat[i] - expected_y[i]).abs() < 0.1);
        }
    }

    #[test]
    fn fit_with_weights_matches_sklearn() {
        // Reference: sklearn DecisionTreeRegressor(max_depth=3, random_state=0)
        // fitted with the same sample_weight. It gives this tree:
        //
        // |--- x1 <= 2.50
        // |   |--- x2 <= 0.50
        // |   |   |--- value: [2.00]
        // |   |--- x2 >  0.50
        // |   |   |--- x1 <= 1.50
        // |   |   |   |--- value: [3.00]
        // |   |   |--- x1 >  1.50
        // |   |   |   |--- value: [2.50]
        // |--- x1 >  2.50
        // |   |--- x1 <= 6.50
        // |   |   |--- x2 <= 0.50
        // |   |   |   |--- value: [6.00]
        // |   |   |--- x2 >  0.50
        // |   |   |   |--- value: [7.50]
        // |   |--- x1 >  6.50
        // |   |   |--- x2 <= 0.50
        // |   |   |   |--- value: [9.00]
        // |   |   |--- x2 >  0.50
        // |   |   |   |--- value: [9.80]
        let x = DenseMatrix::from_2d_array(&[
            &[1., 0.],
            &[1., 1.],
            &[2., 1.],
            &[3., 0.],
            &[3., 0.],
            &[3., 1.],
            &[5., 0.],
            &[5., 1.],
            &[6., 1.],
            &[7., 0.],
            &[8., 1.],
            &[8., 1.],
        ])
        .unwrap();
        let y: Vec<f64> = vec![2.0, 3.0, 2.5, 6.0, 5.0, 7.0, 6.5, 8.0, 7.5, 9.0, 10.0, 9.5];
        let sample_weights = [2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0, 1.0, 2.0, 1.0, 3.0, 2.0];

        // smartcore counts the root as level 1, so sklearn max_depth=3 is max_depth=4 here.
        let parameters = DecisionTreeRegressorParameters::default().with_max_depth(4);
        let tree = DecisionTreeRegressor::fit_with_weights(&x, &y, &sample_weights, parameters)
            .expect("Fit should work");

        // Each training row gets the weighted mean of its leaf.
        let y_hat = tree.predict(&x).unwrap();
        let y_expected = [2.0, 3.0, 2.5, 6.0, 6.0, 7.5, 6.0, 7.5, 7.5, 9.0, 9.8, 9.8];
        for i in 0..y_expected.len() {
            assert!(
                (y_hat[i] - y_expected[i]).abs() < 1e-9,
                "row {i}: got {}, expected {}",
                y_hat[i],
                y_expected[i]
            );
        }

        // Probe points on each side of each threshold check the split features and the
        // midpoint thresholds. At the node x1 > 6.5, the splits x2 <= 0.5 and x1 <= 7.5 give
        // the same partition and the same gain. sklearn picks x2 because of its random
        // feature order. Thus we probe only points where both splits give the same value.
        let probes = DenseMatrix::from_2d_array(&[
            // root: x1 <= 2.5
            &[2.4, 0.],
            &[2.6, 0.],
            &[2.4, 1.],
            &[2.6, 1.],
            // left: x2 <= 0.5
            &[1.0, 0.4],
            &[1.0, 0.6],
            // left, x2 > 0.5: x1 <= 1.5
            &[1.4, 1.],
            &[1.6, 1.],
            // right: x1 <= 6.5
            &[6.4, 0.],
            &[6.6, 0.],
            &[6.4, 1.],
            // right, x1 <= 6.5: x2 <= 0.5
            &[4.0, 0.4],
            &[4.0, 0.6],
            // right, x1 > 6.5: tied split
            &[7.0, 0.4],
            &[8.0, 0.6],
            // out of the training range
            &[0., 0.],
            &[10., 1.],
        ])
        .unwrap();
        let probes_hat = tree.predict(&probes).unwrap();
        let probes_expected = [
            2.0, 6.0, 2.5, 7.5, 2.0, 3.0, 3.0, 2.5, 6.0, 9.0, 7.5, 6.0, 7.5, 9.0, 9.8, 2.0, 9.8,
        ];
        for i in 0..probes_expected.len() {
            assert!(
                (probes_hat[i] - probes_expected[i]).abs() < 1e-9,
                "probe {i}: got {}, expected {}",
                probes_hat[i],
                probes_expected[i]
            );
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159., 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165., 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            &[365.385, 187., 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335., 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.95, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ])
        .unwrap();
        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let tree = DecisionTreeRegressor::fit(&x, &y, Default::default()).unwrap();

        let deserialized_tree: DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            postcard::from_bytes(&postcard::to_allocvec(&tree).unwrap()).unwrap();

        assert_eq!(tree, deserialized_tree);
    }
}
