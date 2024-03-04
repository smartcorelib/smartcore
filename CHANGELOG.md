# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2023-04-05

## Added
- WARNING: Breaking changes!
- `DenseMatrix` constructor now returns `Result` to avoid user instantiating inconsistent rows/cols count. Their return values need to be unwrapped with `unwrap()`, see tests

## [0.3.0] - 2022-11-09 

## Added
- WARNING: Breaking changes!
- Complete refactoring with **extensive API changes** that includes:
    * moving to a new traits system, less structs more traits
    * adapting all the modules to the new traits system
    * moving to Rust 2021, use of object-safe traits and `as_ref`
    * reorganization of the code base, eliminate duplicates
- implements `readers` (needs "serde" feature) for read/write CSV file, extendible to other formats
- default feature is now Wasm-/Wasi-first

## Changed
- WARNING: Breaking changes!
- Seeds to multiple algorithims that depend on random number generation
- Added a new parameter to `train_test_split` to define the seed
- changed use of "serde" feature

## Dropped
- WARNING: Breaking changes!
- Drop `nalgebra-bindings` feature, only `ndarray` as supported library

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
