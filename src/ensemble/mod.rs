//! # Ensemble Methods
//!
//! Combining predictions of several base estimators is a general-purpose procedure for reducing the variance of a statistical learning method.
//! When combined with bagging, ensemble models achive superior performance to individual estimators.
//!
//! The main idea behind bagging (or bootstrap aggregation) is to fit the same base model to a big number of random subsets of the original training
//! set and then aggregate their individual predictions to form a final prediction. In classification setting the overall prediction is the most commonly
//! occurring majority class among the individual predictions.
//!
//! In SmartCore you will find implementation of RandomForest - a popular averaging algorithms based on randomized [decision trees](../tree/index.html).
//! Random forests provide an improvement over bagged trees by way of a small tweak that decorrelates the trees. As in bagging, we build a number of
//! decision trees on bootstrapped training samples. But when building these decision trees, each time a split in a tree is considered,
//! a random sample of _m_ predictors is chosen as split candidates from the full set of _p_ predictors.
//!
//! ## References:
//!
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 8.2 Bagging, Random Forests, Boosting](http://faculty.marshall.usc.edu/gareth-james/ISL/)

/// Random forest classifier
pub mod random_forest_classifier;
/// Random forest regressor
pub mod random_forest_regressor;
