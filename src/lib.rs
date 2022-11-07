#![allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::unnecessary_wraps,
    clippy::upper_case_acronyms
)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]

//! # smartcore
//!
//! Welcome to `smartcore`, machine learning in Rust!
//!
//! `smartcore` features various classification, regression and clustering algorithms including support vector machines, random forests, k-means and DBSCAN,
//! as well as tools for model selection and model evaluation.
//!
//! `smartcore` provides its own traits system that extends Rust standard library, to deal with linear algebra and common
//! computational models. Its API is designed using well recognizable patterns. Extra features (like support for [ndarray](https://docs.rs/ndarray)
//! structures) is available via optional features.
//!
//! ## Getting Started
//!
//! To start using `smartcore` latest stable version simply add the following to your `Cargo.toml` file:
//! ```ignore
//! [dependencies]
//! smartcore = "*"
//! ```
//!
//! To start using smartcore development version with latest unstable additions:
//! ```ignore
//! [dependencies]
//! smartcore = { git = "https://github.com/smartcorelib/smartcore", branch = "development" }
//! ```
//!
//! There are different features that can be added to the base library, for example to add sample datasets:
//! ```ignore
//! [dependencies]
//! smartcore = { git = "https://github.com/smartcorelib/smartcore", features = ["datasets"] }
//! ```
//! Check `smartcore`'s `Cargo.toml` for available features.
//!
//! ## Using Jupyter
//! For quick introduction, Jupyter Notebooks are available [here](https://github.com/smartcorelib/smartcore-jupyter/tree/main/notebooks).
//! You can set up a local environment to run Rust notebooks using [EVCXR](https://github.com/google/evcxr)
//! following [these instructions](https://depth-first.com/articles/2020/09/21/interactive-rust-in-a-repl-and-jupyter-notebook-with-evcxr/).
//!
//!
//! ## First Example
//! For example, you can use this code to fit a [K Nearest Neighbors classifier](neighbors/knn_classifier/index.html) to a dataset that is defined as standard Rust vector:
//!
//! ```
//! // DenseMatrix definition
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! // KNNClassifier
//! use smartcore::neighbors::knn_classifier::*;
//! // Various distance metrics
//! use smartcore::metrics::distance::*;
//!
//! // Turn Rust vector-slices with samples into a matrix
//! let x = DenseMatrix::from_2d_array(&[
//!    &[1., 2.],
//!    &[3., 4.],
//!    &[5., 6.],
//!    &[7., 8.],
//!    &[9., 10.]]);
//! // Our classes are defined as a vector
//! let y = vec![2, 2, 2, 3, 3];
//!
//! // Train classifier
//! let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
//!
//! // Predict classes
//! let y_hat = knn.predict(&x).unwrap();
//! ```
//!
//! ## Overview
//! 
//! ### Supported algorithms
//! All machine learning algorithms are grouped into these broad categories:
//! * [Clustering](cluster/index.html), unsupervised clustering of unlabeled data.
//! * [Matrix Decomposition](decomposition/index.html), various methods for matrix decomposition.
//! * [Linear Models](linear/index.html), regression and classification methods where output is assumed to have linear relation to explanatory variables
//! * [Ensemble Models](ensemble/index.html), variety of regression and classification ensemble models
//! * [Tree-based Models](tree/index.html), classification and regression trees
//! * [Nearest Neighbors](neighbors/index.html), K Nearest Neighbors for classification and regression
//! * [Naive Bayes](naive_bayes/index.html), statistical classification technique based on Bayes Theorem
//! * [SVM](svm/index.html), support vector machines
//! 
//! ### Linear Algebra traits system
//! For an introduction to `smartcore`'s traits system see [this notebook](https://github.com/smartcorelib/smartcore-jupyter/blob/5523993c53c6ec1fd72eea130ef4e7883121c1ea/notebooks/01-A-little-bit-about-numbers.ipynb)

/// Foundamental numbers traits
pub mod numbers;

/// Various algorithms and helper methods that are used elsewhere in smartcore
pub mod algorithm;
pub mod api;

/// Algorithms for clustering of unlabeled data
pub mod cluster;
/// Various datasets
#[cfg(feature = "datasets")]
pub mod dataset;
/// Matrix decomposition algorithms
pub mod decomposition;
/// Ensemble methods, including Random Forest classifier and regressor
pub mod ensemble;
pub mod error;
/// Diverse collection of linear algebra abstractions and methods that power smartcore algorithms
pub mod linalg;
/// Supervised classification and regression models that assume linear relationship between dependent and explanatory variables.
pub mod linear;
/// Functions for assessing prediction error.
pub mod metrics;
/// TODO: add docstring for model_selection
pub mod model_selection;
///  Supervised learning algorithms based on applying the Bayes theorem with the independence assumptions between predictors
pub mod naive_bayes;
/// Supervised neighbors-based learning methods
pub mod neighbors;
/// Optimization procedures
pub mod optimization;
/// Preprocessing utilities
pub mod preprocessing;
/// Reading in data from serialized formats
#[cfg(feature = "serde")]
pub mod readers;
/// Support Vector Machines
pub mod svm;
/// Supervised tree-based learning methods
pub mod tree;

pub(crate) mod rand_custom;
