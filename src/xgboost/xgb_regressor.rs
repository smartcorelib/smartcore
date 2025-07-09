//! # Extreme Gradient Boosting (XGBoost)
//!
//! XGBoost is a highly efficient and effective implementation of the gradient boosting framework.
//! Like other boosting models, it builds an ensemble of sequential decision trees, where each new tree
//! is trained to correct the errors of the previous ones.
//!
//! What makes XGBoost powerful is its use of both the first and second derivatives (gradient and hessian)
//! of the loss function, which allows for more accurate approximations and faster convergence. It also
//! includes built-in regularization techniques (L1/`alpha` and L2/`lambda`) to prevent overfitting.
//!
//! This implementation was ported to Rust from the concepts and algorithm explained in the blog post
//! ["XGBoost from Scratch"](https://randomrealizations.com/posts/xgboost-from-scratch/). It is designed
//! to be a general-purpose regressor that can be used with any objective function that provides a gradient
//! and a hessian.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::xgboost::{XGRegressor, XGRegressorParameters};
//!
//! // Simple dataset: predict y = 2*x
//! let x = DenseMatrix::from_2d_array(&[
//!     &[1.0], &[2.0], &[3.0], &[4.0], &[5.0]
//! ]).unwrap();
//! let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
//!
//! // Use default parameters, but set a few for demonstration
//! let parameters = XGRegressorParameters::default()
//!     .with_n_estimators(50)
//!     .with_max_depth(3)
//!     .with_learning_rate(0.1);
//!
//! // Train the model
//! let model = XGRegressor::fit(&x, &y, parameters).unwrap();
//!
//! // Make predictions
//! let x_test = DenseMatrix::from_2d_array(&[&[6.0], &[7.0]]).unwrap();
//! let y_hat = model.predict(&x_test).unwrap();
//!
//! // y_hat should be close to [12.0, 14.0]
//! ```
//!

use rand::{seq::SliceRandom, Rng};
use std::{iter::zip, marker::PhantomData};

use crate::{
    api::{PredictorBorrow, SupervisedEstimatorBorrow},
    error::{Failed, FailedError},
    linalg::basic::arrays::{Array1, Array2},
    numbers::basenum::Number,
    rand_custom::get_rng_impl,
};

/// Defines the objective function to be optimized.
/// The objective function provides the loss, gradient (first derivative), and
/// hessian (second derivative) required for the XGBoost algorithm.
#[derive(Clone, Debug)]
pub enum Objective {
    /// The objective for regression tasks using Mean Squared Error.
    /// Loss: 0.5 * (y_true - y_pred)^2
    MeanSquaredError,
}

impl Objective {
    /// Calculates the loss for each sample given the true and predicted values.
    ///
    /// # Arguments
    /// * `y_true` - A vector of the true target values.
    /// * `y_pred` - A vector of the predicted values.
    ///
    /// # Returns
    /// The mean of the calculated loss values.
    pub fn loss_function<TY: Number, Y: Array1<TY>>(&self, y_true: &Y, y_pred: &Vec<f64>) -> f64 {
        match self {
            Objective::MeanSquaredError => {
                zip(y_true.iterator(0), y_pred)
                    .map(|(true_val, pred_val)| {
                        0.5 * (true_val.to_f64().unwrap() - pred_val).powi(2)
                    })
                    .sum::<f64>()
                    / y_true.shape() as f64
            }
        }
    }

    /// Calculates the gradient (first derivative) of the loss function.
    ///
    /// # Arguments
    /// * `y_true` - A vector of the true target values.
    /// * `y_pred` - A vector of the predicted values.
    ///
    /// # Returns
    /// A vector of gradients for each sample.
    pub fn gradient<TY: Number, Y: Array1<TY>>(&self, y_true: &Y, y_pred: &Vec<f64>) -> Vec<f64> {
        match self {
            Objective::MeanSquaredError => zip(y_true.iterator(0), y_pred)
                .map(|(true_val, pred_val)| (*pred_val - true_val.to_f64().unwrap()))
                .collect(),
        }
    }

    /// Calculates the hessian (second derivative) of the loss function.
    ///
    /// # Arguments
    /// * `y_true` - A vector of the true target values.
    /// * `y_pred` - A vector of the predicted values.
    ///
    /// # Returns
    /// A vector of hessians for each sample.
    #[allow(unused_variables)]
    pub fn hessian<TY: Number, Y: Array1<TY>>(&self, y_true: &Y, y_pred: &[f64]) -> Vec<f64> {
        match self {
            Objective::MeanSquaredError => vec![1.0; y_true.shape()],
        }
    }
}

/// Represents a single decision tree in the XGBoost ensemble.
///
/// This is a recursive data structure where each `TreeRegressor` is a node
/// that can have a left and a right child, also of type `TreeRegressor`.
#[allow(dead_code)]
struct TreeRegressor<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    left: Option<Box<TreeRegressor<TX, TY, X, Y>>>,
    right: Option<Box<TreeRegressor<TX, TY, X, Y>>>,
    /// The output value of this node. If it's a leaf, this is the final prediction.
    value: f64,
    /// The feature value threshold used to split this node.
    threshold: f64,
    /// The index of the feature used for splitting.
    split_feature_idx: usize,
    /// The gain in score achieved by this split.
    split_score: f64,
    _phantom_tx: PhantomData<TX>,
    _phantom_ty: PhantomData<TY>,
    _phantom_x: PhantomData<X>,
    _phantom_y: PhantomData<Y>,
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    TreeRegressor<TX, TY, X, Y>
{
    /// Recursively builds a decision tree (a `TreeRegressor` node).
    ///
    /// This function determines the optimal split for the given set of samples (`idxs`)
    /// and then recursively calls itself to build the left and right child nodes.
    ///
    /// # Arguments
    /// * `data` - The full training dataset.
    /// * `g` - Gradients for all samples.
    /// * `h` - Hessians for all samples.
    /// * `idxs` - The indices of the samples belonging to the current node.
    /// * `max_depth` - The maximum remaining depth for this branch.
    /// * `min_child_weight` - The minimum sum of hessians required in a child node.
    /// * `lambda` - L2 regularization term on weights.
    /// * `gamma` - Minimum loss reduction required to make a further partition.
    pub fn fit(
        data: &X,
        g: &Vec<f64>,
        h: &Vec<f64>,
        idxs: &[usize],
        max_depth: u16,
        min_child_weight: f64,
        lambda: f64,
        gamma: f64,
    ) -> Self {
        let g_sum = idxs.iter().map(|&i| g[i]).sum::<f64>();
        let h_sum = idxs.iter().map(|&i| h[i]).sum::<f64>();
        let value = -g_sum / (h_sum + lambda);

        let mut best_feature_idx = usize::MAX;
        let mut best_split_score = 0.0;
        let mut best_threshold = 0.0;
        let mut left = Option::None;
        let mut right = Option::None;

        if max_depth > 0 {
            Self::insert_child_nodes(
                data,
                g,
                h,
                idxs,
                &mut best_feature_idx,
                &mut best_split_score,
                &mut best_threshold,
                &mut left,
                &mut right,
                max_depth,
                min_child_weight,
                lambda,
                gamma,
            );
        }

        Self {
            left,
            right,
            value,
            threshold: best_threshold,
            split_feature_idx: best_feature_idx,
            split_score: best_split_score,
            _phantom_tx: PhantomData,
            _phantom_ty: PhantomData,
            _phantom_x: PhantomData,
            _phantom_y: PhantomData,
        }
    }

    /// Finds the best split and creates child nodes if a valid split is found.
    fn insert_child_nodes(
        data: &X,
        g: &Vec<f64>,
        h: &Vec<f64>,
        idxs: &[usize],
        best_feature_idx: &mut usize,
        best_split_score: &mut f64,
        best_threshold: &mut f64,
        left: &mut Option<Box<Self>>,
        right: &mut Option<Box<Self>>,
        max_depth: u16,
        min_child_weight: f64,
        lambda: f64,
        gamma: f64,
    ) {
        let (_, n_features) = data.shape();
        for i in 0..n_features {
            Self::find_best_split(
                data,
                g,
                h,
                idxs,
                i,
                best_feature_idx,
                best_split_score,
                best_threshold,
                min_child_weight,
                lambda,
                gamma,
            );
        }

        // A split is only valid if it results in a positive gain.
        if *best_split_score > 0.0 {
            let mut left_idxs = Vec::new();
            let mut right_idxs = Vec::new();
            for idx in idxs.iter() {
                if data.get((*idx, *best_feature_idx)).to_f64().unwrap() <= *best_threshold {
                    left_idxs.push(*idx);
                } else {
                    right_idxs.push(*idx);
                }
            }

            *left = Some(Box::new(TreeRegressor::fit(
                data,
                g,
                h,
                &left_idxs,
                max_depth - 1,
                min_child_weight,
                lambda,
                gamma,
            )));
            *right = Some(Box::new(TreeRegressor::fit(
                data,
                g,
                h,
                &right_idxs,
                max_depth - 1,
                min_child_weight,
                lambda,
                gamma,
            )));
        }
    }

    /// Iterates through a single feature to find the best possible split point.
    fn find_best_split(
        data: &X,
        g: &[f64],
        h: &[f64],
        idxs: &[usize],
        feature_idx: usize,
        best_feature_idx: &mut usize,
        best_split_score: &mut f64,
        best_threshold: &mut f64,
        min_child_weight: f64,
        lambda: f64,
        gamma: f64,
    ) {
        let mut sorted_idxs = idxs.to_owned();
        sorted_idxs.sort_by(|a, b| {
            data.get((*a, feature_idx))
                .partial_cmp(data.get((*b, feature_idx)))
                .unwrap()
        });

        let sum_g = sorted_idxs.iter().map(|&i| g[i]).sum::<f64>();
        let sum_h = sorted_idxs.iter().map(|&i| h[i]).sum::<f64>();

        let mut sum_g_right = sum_g;
        let mut sum_h_right = sum_h;
        let mut sum_g_left = 0.0;
        let mut sum_h_left = 0.0;

        for i in 0..sorted_idxs.len() - 1 {
            let idx = sorted_idxs[i];
            let next_idx = sorted_idxs[i + 1];

            let g_i = g[idx];
            let h_i = h[idx];
            let x_i = data.get((idx, feature_idx)).to_f64().unwrap();
            let x_i_next = data.get((next_idx, feature_idx)).to_f64().unwrap();

            sum_g_left += g_i;
            sum_h_left += h_i;
            sum_g_right -= g_i;
            sum_h_right -= h_i;

            if sum_h_left < min_child_weight || x_i == x_i_next {
                continue;
            }
            if sum_h_right < min_child_weight {
                break;
            }

            let gain = 0.5
                * ((sum_g_left * sum_g_left / (sum_h_left + lambda))
                    + (sum_g_right * sum_g_right / (sum_h_right + lambda))
                    - (sum_g * sum_g / (sum_h + lambda)))
                - gamma;

            if gain > *best_split_score {
                *best_split_score = gain;
                *best_threshold = (x_i + x_i_next) / 2.0;
                *best_feature_idx = feature_idx;
            }
        }
    }

    /// Predicts the output values for a dataset.
    pub fn predict(&self, data: &X) -> Vec<f64> {
        let (n_samples, n_features) = data.shape();
        (0..n_samples)
            .map(|i| {
                self.predict_for_row(&Vec::from_iterator(
                    data.get_row(i).iterator(0).copied(),
                    n_features,
                ))
            })
            .collect()
    }

    /// Predicts the output value for a single row of data by traversing the tree.
    pub fn predict_for_row(&self, row: &Vec<TX>) -> f64 {
        // A leaf node is identified by having no children.
        if self.left.is_none() {
            return self.value;
        }

        // Recurse down the appropriate branch.
        let child = if row[self.split_feature_idx].to_f64().unwrap() <= self.threshold {
            self.left.as_ref().unwrap()
        } else {
            self.right.as_ref().unwrap()
        };

        child.predict_for_row(row)
    }
}

/// Parameters for the `jRegressor` model.
///
/// This struct holds all the hyperparameters that control the training process.
#[derive(Clone, Debug)]
pub struct XGRegressorParameters {
    /// The number of boosting rounds or trees to build.
    pub n_estimators: usize,
    /// The maximum depth of each tree.
    pub max_depth: u16,
    /// Step size shrinkage used to prevent overfitting.
    pub learning_rate: f64,
    /// Minimum sum of instance weight (hessian) needed in a child.
    pub min_child_weight: usize,
    /// L2 regularization term on weights.
    pub lambda: f64,
    /// Minimum loss reduction required to make a further partition on a leaf node.
    pub gamma: f64,
    /// The initial prediction score for all instances.
    pub base_score: f64,
    /// The fraction of samples to be used for fitting the individual base learners.
    pub subsample: f64,
    /// The seed for the random number generator for reproducibility.
    pub seed: u64,
    /// The objective function to be optimized.
    pub objective: Objective,
}

impl Default for XGRegressorParameters {
    /// Creates a new set of `XGRegressorParameters` with default values.
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.3,
            max_depth: 6,
            min_child_weight: 1,
            lambda: 1.0,
            gamma: 0.0,
            base_score: 0.5,
            subsample: 1.0,
            seed: 0,
            objective: Objective::MeanSquaredError,
        }
    }
}

// Builder pattern for XGRegressorParameters
impl XGRegressorParameters {
    /// Sets the number of boosting rounds or trees to build.
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Sets the step size shrinkage used to prevent overfitting.
    ///
    /// Also known as `eta`. A smaller value makes the model more robust by preventing
    /// too much weight being given to any single tree.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the maximum depth of each individual tree.
    // A lower value helps prevent overfitting.*
    pub fn with_max_depth(mut self, max_depth: u16) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the minimum sum of instance weight (hessian) needed in a child node.
    ///
    /// If the tree partition step results in a leaf node with the sum of
    // instance weight less than `min_child_weight`, then the building process*
    /// will give up further partitioning.
    pub fn with_min_child_weight(mut self, min_child_weight: usize) -> Self {
        self.min_child_weight = min_child_weight;
        self
    }

    /// Sets the L2 regularization term on weights (`lambda`).
    ///
    /// Increasing this value will make the model more conservative.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Sets the minimum loss reduction required to make a further partition on a leaf node.
    ///
    /// The larger `gamma` is, the more conservative the algorithm will be.
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Sets the initial prediction score for all instances.
    pub fn with_base_score(mut self, base_score: f64) -> Self {
        self.base_score = base_score;
        self
    }

    /// Sets the fraction of samples to be used for fitting individual base learners.
    ///
    /// A value of less than 1.0 introduces randomness and helps prevent overfitting.
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample;
        self
    }

    /// Sets the seed for the random number generator for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the objective function to be optimized during training.
    pub fn with_objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }
}

/// An Extreme Gradient Boosting (XGBoost) model for regression and classification tasks.
pub struct XGRegressor<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> {
    regressors: Option<Vec<TreeRegressor<TX, TY, X, Y>>>,
    parameters: Option<XGRegressorParameters>,
    _phantom_ty: PhantomData<TY>,
    _phantom_tx: PhantomData<TX>,
    _phantom_y: PhantomData<Y>,
    _phantom_x: PhantomData<X>,
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> XGRegressor<TX, TY, X, Y> {
    /// Fits the XGBoost model to the training data.
    pub fn fit(data: &X, y: &Y, parameters: XGRegressorParameters) -> Result<Self, Failed> {
        if parameters.subsample > 1.0 || parameters.subsample <= 0.0 {
            return Err(Failed::because(
                FailedError::ParametersError,
                "Subsample ratio must be in (0, 1].",
            ));
        }

        let (n_samples, _) = data.shape();
        let learning_rate = parameters.learning_rate;
        let mut predictions = vec![parameters.base_score; n_samples];

        let mut regressors = Vec::new();
        let mut rng = get_rng_impl(Some(parameters.seed));

        for _ in 0..parameters.n_estimators {
            let gradients = parameters.objective.gradient(y, &predictions);
            let hessians = parameters.objective.hessian(y, &predictions);

            let sample_idxs = if parameters.subsample < 1.0 {
                Self::sample_without_replacement(n_samples, parameters.subsample, &mut rng)
            } else {
                (0..n_samples).collect::<Vec<usize>>()
            };

            let regressor = TreeRegressor::fit(
                data,
                &gradients,
                &hessians,
                &sample_idxs,
                parameters.max_depth,
                parameters.min_child_weight as f64,
                parameters.lambda,
                parameters.gamma,
            );

            let corrections = regressor.predict(data);
            predictions = zip(predictions, corrections)
                .map(|(pred, correction)| pred + (learning_rate * correction))
                .collect();

            regressors.push(regressor);
        }

        Ok(Self {
            regressors: Some(regressors),
            parameters: Some(parameters),
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
            _phantom_tx: PhantomData,
            _phantom_x: PhantomData,
        })
    }

    /// Predicts target values for the given input data.
    pub fn predict(&self, data: &X) -> Result<Vec<TX>, Failed> {
        let (n_samples, _) = data.shape();

        let parameters = self.parameters.as_ref().unwrap();
        let mut predictions = vec![parameters.base_score; n_samples];
        let regressors = self.regressors.as_ref().unwrap();

        for regressor in regressors.iter() {
            let corrections = regressor.predict(data);
            predictions = zip(predictions, corrections)
                .map(|(pred, correction)| pred + (parameters.learning_rate * correction))
                .collect();
        }

        Ok(predictions
            .into_iter()
            .map(|p| TX::from_f64(p).unwrap())
            .collect())
    }

    /// Creates a random sample of indices without replacement.
    fn sample_without_replacement(
        population_size: usize,
        subsample_ratio: f64,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..population_size).collect();
        indices.shuffle(rng);
        indices.truncate((population_size as f64 * subsample_ratio) as usize);
        indices
    }
}

// Boilerplate implementation for the smartcore traits
impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimatorBorrow<'_, X, Y, XGRegressorParameters> for XGRegressor<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            regressors: None,
            parameters: None,
            _phantom_ty: PhantomData,
            _phantom_y: PhantomData,
            _phantom_tx: PhantomData,
            _phantom_x: PhantomData,
        }
    }

    fn fit(x: &X, y: &Y, parameters: &XGRegressorParameters) -> Result<Self, Failed> {
        XGRegressor::fit(x, y, parameters.clone())
    }
}

impl<TX: Number + PartialOrd, TY: Number, X: Array2<TX>, Y: Array1<TY>> PredictorBorrow<'_, X, TX>
    for XGRegressor<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Vec<TX>, Failed> {
        self.predict(x)
    }
}

// ------------------- TESTS -------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::{arrays::Array, matrix::DenseMatrix};

    /// Tests the gradient and hessian calculations for MeanSquaredError.
    #[test]
    fn test_mse_objective() {
        let objective = Objective::MeanSquaredError;
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.5, 2.5, 2.5];

        let gradients = objective.gradient(&y_true, &y_pred);
        let hessians = objective.hessian(&y_true, &y_pred);

        // Gradients should be (pred - true)
        assert_eq!(gradients, vec![0.5, 0.5, -0.5]);
        // Hessians should be all 1.0 for MSE
        assert_eq!(hessians, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_find_best_split_multidimensional() {
        // Data has two features. The second feature is a better predictor.
        let data = vec![
            vec![1.0, 10.0], // g = -0.5
            vec![1.0, 20.0], // g = -1.0
            vec![1.0, 30.0], // g = 1.0
            vec![1.0, 40.0], // g = 1.5
        ];
        let data = DenseMatrix::from_2d_vec(&data).unwrap();
        let g = vec![-0.5, -1.0, 1.0, 1.5];
        let h = vec![1.0, 1.0, 1.0, 1.0];
        let idxs = (0..4).collect::<Vec<usize>>();

        let mut best_feature_idx = usize::MAX;
        let mut best_split_score = 0.0;
        let mut best_threshold = 0.0;

        // Manually calculated expected gain for the best split (on feature 1, with lambda=1.0).
        // G_left = -1.5, H_left = 2.0
        // G_right = 2.5, H_right = 2.0
        // G_total = 1.0, H_total = 4.0
        // Gain = 0.5 * (G_l^2/(H_l+λ) + G_r^2/(H_r+λ) - G_t^2/(H_t+λ))
        // Gain = 0.5 * ((-1.5)^2/(2+1) + (2.5)^2/(2+1) - (1.0)^2/(4+1))
        // Gain = 0.5 * (2.25/3 + 6.25/3 - 1.0/5) = 0.5 * (0.75 + 2.0833 - 0.2) = 1.3166...
        let expected_gain = 1.3166666666666667;

        // Search both features. The algorithm must find the best split on feature 1.
        let (_, n_features) = data.shape();
        for i in 0..n_features {
            TreeRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::find_best_split(
                &data,
                &g,
                &h,
                &idxs,
                i,
                &mut best_feature_idx,
                &mut best_split_score,
                &mut best_threshold,
                1.0,
                1.0,
                0.0,
            );
        }

        assert_eq!(best_feature_idx, 1); // Should choose the second feature
        assert!((best_split_score - expected_gain).abs() < 1e-9);
        assert_eq!(best_threshold, 25.0); // (20 + 30) / 2
    }

    /// Tests that the TreeRegressor can build a simple one-level tree on multidimensional data.
    #[test]
    fn test_tree_regressor_fit_multidimensional() {
        let data = vec![
            vec![1.0, 10.0],
            vec![1.0, 20.0],
            vec![1.0, 30.0],
            vec![1.0, 40.0],
        ];
        let data = DenseMatrix::from_2d_vec(&data).unwrap();
        let g = vec![-0.5, -1.0, 1.0, 1.5];
        let h = vec![1.0, 1.0, 1.0, 1.0];
        let idxs = (0..4).collect::<Vec<usize>>();

        let tree = TreeRegressor::<f64, f64, DenseMatrix<f64>, Vec<f64>>::fit(
            &data, &g, &h, &idxs, 2, 1.0, 1.0, 0.0,
        );

        // Check that the root node was split on the correct feature
        assert!(tree.left.is_some());
        assert!(tree.right.is_some());
        assert_eq!(tree.split_feature_idx, 1); // Should split on the second feature
        assert_eq!(tree.threshold, 25.0);

        // Check leaf values (G/H+lambda)
        // Left leaf: G = -1.5, H = 2.0 => value = -(-1.5)/(2+1) = 0.5
        // Right leaf: G = 2.5, H = 2.0 => value = -(2.5)/(2+1) = -0.8333
        assert!((tree.left.unwrap().value - 0.5).abs() < 1e-9);
        assert!((tree.right.unwrap().value - (-0.833333333)).abs() < 1e-9);
    }

    /// A "smoke test" to ensure the main XGRegressor can fit and predict on multidimensional data.
    #[test]
    fn test_xgregressor_fit_predict_multidimensional() {
        // Simple 2D data where y is roughly 2*x1 + 3*x2
        let x_vec = vec![
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 2.0],
        ];
        let x = DenseMatrix::from_2d_vec(&x_vec).unwrap();
        let y = vec![5.0, 7.0, 8.0, 10.0];

        let params = XGRegressorParameters::default()
            .with_n_estimators(10)
            .with_max_depth(2);

        let fit_result = XGRegressor::fit(&x, &y, params);
        assert!(
            fit_result.is_ok(),
            "Fit failed with error: {:?}",
            fit_result.err()
        );

        let model = fit_result.unwrap();
        let predict_result = model.predict(&x);
        assert!(
            predict_result.is_ok(),
            "Predict failed with error: {:?}",
            predict_result.err()
        );

        let predictions = predict_result.unwrap();
        assert_eq!(predictions.len(), 4);
    }
}
