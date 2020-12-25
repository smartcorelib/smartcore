//! # Common Interfaces and API
//!
//! This module provides interfaces and uniform API with simple conventions
//! that are used in other modules for supervised and unsupervised learning.

use crate::error::Failed;

/// An estimator for unsupervised learning, that provides method `fit` to learn from data
pub trait UnsupervisedEstimator<X, P> {
    /// Fit a model to a training dataset, estimate model's parameters.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `parameters` - hyperparameters of an algorithm
    fn fit(x: &X, parameters: P) -> Result<Self, Failed>
    where
        Self: Sized,
        P: Clone;
}

/// An estimator for supervised learning, , that provides method `fit` to learn from data and training values
pub trait SupervisedEstimator<X, Y, P> {
    /// Fit a model to a training dataset, estimate model's parameters.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target training values of size _N_.
    /// * `parameters` - hyperparameters of an algorithm
    fn fit(x: &X, y: &Y, parameters: P) -> Result<Self, Failed>
    where
        Self: Sized,
        P: Clone;
}

/// Implements method predict that estimates target value from new data
pub trait Predictor<X, Y> {
    /// Estimate target values from new data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    fn predict(&self, x: &X) -> Result<Y, Failed>;
}

/// Implements method transform that filters or modifies input data
pub trait Transformer<X> {
    /// Transform data by modifying or filtering it
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    fn transform(&self, x: &X) -> Result<X, Failed>;
}
