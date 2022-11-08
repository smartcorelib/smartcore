# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3] - 2022-11 

## Added
- WARNING: Breaking changes!
- Seeds to multiple algorithims that depend on random number generation.
- Drop `nalgebra-bindings` feature
- Complete refactoring with **extensive API changes** that includes:
    * moving to a new traits system, less structs more traits
    * adapting all the modules to the new traits system
    * moving to Rust 2021, in particular the use of `dyn` and `as_ref`
    * reorganization of the code base, trying to eliminate duplicates
- usage of `serde` is now optional, use the `serde` feature
- default feature is now Wasm-/Wasi-first for minimal binary size

## BREAKING CHANGE
- Added a new parameter to `train_test_split` to define the seed.

## [0.2.1] - 2021-05-10

## Added
- L2 regularization penalty to the Logistic Regression
- Getters for the naive bayes structs
- One hot encoder
- Make moons data generator
- Support for WASM.

## Changed
- Make serde optional

## [0.2.0] - 2021-01-03

### Added
- DBSCAN
- Epsilon-SVR, SVC
- Ridge, Lasso, ElasticNet
- Bernoulli, Gaussian, Categorical and Multinomial Naive Bayes
- K-fold Cross Validation
- Singular value decomposition
- New api module
- Integration with Clippy
- Cholesky decomposition

### Changed
- ndarray upgraded to 0.14
- smartcore::error:FailedError is now non-exhaustive
- K-Means
- PCA
- Random Forest
- Linear and Logistic Regression
- KNN
- Decision Tree

## [0.1.0] - 2020-09-25

### Added
- First release of smartcore.
- KNN + distance metrics (Euclidian, Minkowski, Manhattan, Hamming, Mahalanobis)
- Linear Regression (OLS)
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- PCA
- K-Means
- Integrated with ndarray
- Abstract linear algebra methods
- RandomForest Regressor
- Decision Tree Regressor
- Serde integration
- Integrated with nalgebra
- LU, QR, SVD, EVD
- Evaluation Metrics
