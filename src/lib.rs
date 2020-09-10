#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]

//! # SmartCore
//!
//! Welcome to SmartCore library, the most complete machine learning library for Rust!
//!
//! In SmartCore you will find implementation of these ML algorithms:
//! * __Regression__: Linear Regression (OLS), Decision Tree Regressor, Random Forest Regressor, K Nearest Neighbors
//! * __Classification__: Logistic Regressor, Decision Tree Classifier, Random Forest Classifier, Supervised Nearest Neighbors (KNN)
//! * __Clustering__: K-Means
//! * __Matrix Decomposition__: PCA, LU, QR, SVD, EVD
//! * __Distance Metrics__: Euclidian, Minkowski, Manhattan, Hamming, Mahalanobis
//! * __Evaluation Metrics__: Accuracy, AUC, Recall, Precision, F1, Mean Absolute Error, Mean Squared Error, R2
//!
//! Most of algorithms implemented in SmartCore operate on n-dimentional arrays. While you can use Rust vectors with all functions defined in this library
//! we do recommend to go with one of the popular linear algebra libraries available in Rust. At this moment we support these packages:
//! * [ndarray](https://docs.rs/ndarray)
//! * [nalgebra](https://docs.rs/nalgebra/)
//!
//! ## Getting Started
//!
//! To start using SmartCore simply add the following to your Cargo.toml file:
//! ```ignore
//! [dependencies]
//! smartcore = "0.1.0"
//! ```
//!
//! All ML algorithms in SmartCore are grouped into these generic categories:
//! * [Clustering](cluster/index.html), unsupervised clustering of unlabeled data.
//! * [Martix Decomposition](decomposition/index.html), various methods for matrix decomposition.
//! * [Linear Models](linear/index.html), regression and classification methods where output is assumed to have linear relation to explanatory variables
//! * [Ensemble Models](ensemble/index.html), variety of regression and classification ensemble models
//! * [Tree-based Models](tree/index.html), classification and regression trees
//! * [Nearest Neighbors](neighbors/index.html), K Nearest Neighbors for classification and regression
//!
//! Each category is assigned to a separate module.
//!
//! For example, KNN classifier is defined in [smartcore::neighbors::knn_classifier](neighbors/knn_classifier/index.html). To train and run it using standard Rust vectors you will
//! run this code:
//!
//! ```
//! // DenseMatrix defenition
//! use smartcore::linalg::naive::dense_matrix::*;
//! // KNNClassifier
//! use smartcore::neighbors::knn_classifier::*;
//! // Various distance metrics
//! use smartcore::math::distance::*;
//!
//! // Turn Rust vectors with samples into a matrix
//! let x = DenseMatrix::from_2d_array(&[
//!    &[1., 2.],
//!    &[3., 4.],
//!    &[5., 6.],
//!    &[7., 8.],
//!    &[9., 10.]]);
//! // Our classes are defined as a Vector
//! let y = vec![2., 2., 2., 3., 3.];
//!
//! // Train classifier
//! let knn = KNNClassifier::fit(&x, &y, Distances::euclidian(), Default::default());
//!
//! // Predict classes
//! let y_hat = knn.predict(&x);
//! ```

/// Various algorithms and helper methods that are used elsewhere in SmartCore
pub mod algorithm;
/// Algorithms for clustering of unlabeled data
pub mod cluster;
/// Various datasets
#[cfg(feature = "datasets")]
pub mod dataset;
/// Matrix decomposition algorithms
pub mod decomposition;
/// Ensemble methods, including Random Forest classifier and regressor
pub mod ensemble;
/// Diverse collection of linear algebra abstractions and methods that power SmartCore algorithms
pub mod linalg;
/// Supervised classification and regression models that assume linear relationship between dependent and explanatory variables.
pub mod linear;
/// Helper methods and classes, including definitions of distance metrics
pub mod math;
/// Functions for assessing prediction error.
pub mod metrics;
/// Supervised neighbors-based learning methods
pub mod neighbors;
pub(crate) mod optimization;
/// Supervised tree-based learning methods
pub mod tree;
