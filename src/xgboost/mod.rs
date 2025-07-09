//! # XGBoost
//!
//! XGBoost, which stands for Extreme Gradient Boosting, is a powerful and efficient implementation of the gradient boosting framework. Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
//!
//! The core idea of boosting is to build the model in a stage-wise fashion. It learns from its mistakes by sequentially adding new models that correct the errors of the previous ones. Unlike bagging, which trains models in parallel, boosting is a sequential process. Each new tree is fit on a modified version of the original data set, specifically focusing on the instances where the previous models performed poorly.
//!
//! XGBoost enhances this process through several key innovations. It employs a more regularized model formalization to control over-fitting, which gives it better performance. It also has a highly optimized and parallelized tree construction process, making it significantly faster and more scalable than traditional gradient boosting implementations.
//!
//! ## References:
//!
//! * "Elements of Statistical Learning", Hastie T., Tibshirani R., Friedman J., 10. Boosting and Additive Trees
//! *  XGBoost: A Scalable Tree Boosting System, Chen T., Guestrin C.

// xgboost implementation
pub mod xgb_regressor;
pub use xgb_regressor::{XGRegressor, XGRegressorParameters};
