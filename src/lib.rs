#![allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::unnecessary_wraps,
    clippy::upper_case_acronyms
)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]

//! # SmartCore
//!
//! Welcome to SmartCore, the most advanced machine learning library in Rust!
//!
//! SmartCore features various classification, regression and clustering algorithms including support vector machines, random forests, k-means and DBSCAN,
//! as well as tools for model selection and model evaluation.
//!
//! SmartCore is well integrated with a with wide variaty of libraries that provide support for large, multi-dimensional arrays and matrices. At this moment,
//! all Smartcore's algorithms work with ordinary Rust vectors, as well as matrices and vectors defined in these packages:
//! * [ndarray](https://docs.rs/ndarray)
//! * [nalgebra](https://docs.rs/nalgebra/)
//!
//! ## Getting Started
//!
//! To start using SmartCore simply add the following to your Cargo.toml file:
//! ```ignore
//! [dependencies]
//! smartcore = "0.2.0"
//! ```
//!
//! All machine learning algorithms in SmartCore are grouped into these broad categories:
//! * [Clustering](cluster/index.html), unsupervised clustering of unlabeled data.
//! * [Matrix Decomposition](decomposition/index.html), various methods for matrix decomposition.
//! * [Linear Models](linear/index.html), regression and classification methods where output is assumed to have linear relation to explanatory variables
//! * [Ensemble Models](ensemble/index.html), variety of regression and classification ensemble models
//! * [Tree-based Models](tree/index.html), classification and regression trees
//! * [Nearest Neighbors](neighbors/index.html), K Nearest Neighbors for classification and regression
//! * [Naive Bayes](naive_bayes/index.html), statistical classification technique based on Bayes Theorem
//! * [SVM](svm/index.html), support vector machines
//!
//!
//! For example, you can use this code to fit a [K Nearest Neighbors classifier](neighbors/knn_classifier/index.html) to a dataset that is defined as standard Rust vector:
//!
//! ```
//! // DenseMatrix defenition
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! // KNNClassifier
//! use smartcore::neighbors::knn_classifier::*;
//! // Various distance metrics
//! use smartcore::metrics::distance::*;
//!
//! // Turn Rust vectors with samples into a matrix
//! let x = DenseMatrix::from_2d_array(&[
//!    &[1., 2.],
//!    &[3., 4.],
//!    &[5., 6.],
//!    &[7., 8.],
//!    &[9., 10.]]);
//! // Our classes are defined as a Vector
//! let y = vec![2, 2, 2, 3, 3];
//!
//! // Train classifier
//! let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
//!
//! // Predict classes
//! let y_hat = knn.predict(&x).unwrap();
//! ```

/// Foundamental numbers traits
pub mod numbers;

/// Various algorithms and helper methods that are used elsewhere in SmartCore
pub mod algorithm;
pub mod api;

// /// Algorithms for clustering of unlabeled data
// pub mod cluster;
// /// Various datasets
#[cfg(feature = "datasets")]
pub mod dataset;
// /// Matrix decomposition algorithms
// pub mod decomposition;
// /// Ensemble methods, including Random Forest classifier and regressor
// pub mod ensemble;

pub mod error;
/// Diverse collection of linear algebra abstractions and methods that power SmartCore algorithms
pub mod linalg;
/// Supervised classification and regression models that assume linear relationship between dependent and explanatory variables.
pub mod linear;
/// Functions for assessing prediction error.
pub mod metrics;
// pub mod model_selection;
// ///  Supervised learning algorithms based on applying the Bayes theorem with the independence assumptions between predictors
// pub mod naive_bayes;
/// Supervised neighbors-based learning methods
pub mod neighbors;

pub(crate) mod optimization;
// /// Preprocessing utilities
// pub mod preprocessing;
// /// Reading in Data.
// pub mod readers;
// /// Support Vector Machines
// pub mod svm;
// /// Supervised tree-based learning methods
// pub mod tree;
#[cfg(test)]
pub mod utils;

pub(crate) mod rand_custom;
