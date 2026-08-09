<p align="center">
  <a href="https://smartcorelib.github.io/">
    <img src="smartcore.svg" width="450" alt="smartcore">    
  </a>  
</p>
<p align = "center">
    <strong>
        <a href="https://smartcorelib.github.io/">User guide</a> | <a href="https://docs.rs/smartcore/">API</a> | <a href="https://github.com/smartcorelib/smartcore-jupyter">Notebooks</a>
    </strong>
</p>

-----

<p align = "center">
<b>Machine Learning in Rust</b>
</p>

-----
[![CI](https://github.com/smartcorelib/smartcore/actions/workflows/ci.yml/badge.svg)](https://github.com/smartcorelib/smartcore/actions/workflows/ci.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17219259.svg)](https://doi.org/10.5281/zenodo.17219259)

To start getting familiar with the smartcore API, there is now available a [**Jupyter Notebook environment repository**](https://github.com/smartcorelib/smartcore-jupyter). Please see instructions there, contributions welcome see [CONTRIBUTING](.github/CONTRIBUTING.md).

smartcore is a fast, ergonomic machine learning library for Rust, covering classical supervised and unsupervised methods with a modular linear algebra abstraction and optional ndarray support. It aims to provide production-friendly APIs, strong typing, and good defaults while remaining flexible for research and experimentation.


## Highlights

- Broad algorithm coverage: linear models, tree-based methods, ensembles, SVMs, neighbors, clustering, decomposition, and preprocessing.
- Strong linear algebra traits with optional ndarray integration for users who prefer array-first workflows.
- WASM-first defaults with attention to portability; features such as serde and datasets are opt-in.
- Practical utilities for model selection, evaluation, readers (CSV), dataset generators, and built-in sample datasets.


## Install

Add to Cargo.toml:

```toml
[dependencies]
smartcore = "^0.6"
```

For the latest development branch:

```toml
[dependencies]
smartcore = { git = "https://github.com/smartcorelib/smartcore", branch = "development" }
```

Optional features (examples):

- datasets
- serde
- ndarray-bindings (deprecated in favor of ndarray-only support per recent changes)

Check Cargo.toml for available features and compatibility notes.

## Quick start

Here is a minimal example fitting a KNN classifier from native Rust vectors using DenseMatrix:

```rust
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::KNNClassifier;

// Turn vector slices into a matrix
let x = DenseMatrix::from_2d_array(&[
    &[1., 2.],
    &[3., 4.],
    &[5., 6.],
    &[7., 8.],
    &[9., 10.],
]).unwrap();

// Class labels
let y = vec![2, 2, 2, 3, 3];

// Train classifier
let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();

// Predict
let yhat = knn.predict(&x).unwrap();
```

This example mirrors the “First Example” section of the crate docs and demonstrates smartcore’s ergonomic API surface.

## Algorithms

smartcore organizes algorithms into clear modules with consistent traits:

- Clustering: K-Means, DBSCAN, agglomerative (including single-linkage), with K-Means++ initialization and utilities.
- Matrix decomposition: SVD, EVD, Cholesky, LU, QR, plus related linear algebra helpers.
- Linear models: OLS, Ridge, Lasso, ElasticNet, Logistic Regression.
- Ensemble and tree-based: Random Forest (classifier and regressor), Extra Trees, shared reusable components across trees and forests.
- SVM: SVC/SVR with kernel enum support and multiclass extensions.
- Neighbors: KNN classification and regression with distance metrics and fast selection helpers.
- Naive Bayes: Gaussian, Bernoulli, Categorical, Multinomial.
- Preprocessing: encoders, split utilities, and common transforms.
- Model selection and metrics: K-fold, search parameters, and evaluation utilities.

Recent refactors emphasize reusable components in trees/forests and expanded multiclass SVM capabilities. XGBoost-style regression and single-linkage clustering have been added. See CHANGELOG for API changes and migration notes.

## Data access and readers

- CSV readers: Read matrices from CSV with configurable delimiter and header rows, with helpful error messages and testing utilities (including non-IO reader abstractions).
- Dataset generators: make_blobs, make_circles, make_moons for quick experiments.
- Built-in datasets (feature-gated): digits, diabetes, breast cancer, boston, with serialization utilities to persist or refresh .xy bundles.


## WebAssembly and portability

smartcore adopts a WASM/WASI-first posture in defaults to ease browser and embedded deployments. Some file-system operations are restricted in wasm targets; tests and IO utilities are structured to avoid unsupported calls where possible. Enable features like serde selectively to minimize footprint. Consult module-level docs and CHANGELOG for target-specific caveats.

## Notebooks

A curated set of Jupyter notebooks is available via the [companion repository to explore smartcore interactively](https://github.com/smartcorelib/smartcore-jupyter). To run locally, use EVCXR to enable Rust notebooks. This is the recommended path to quickly experiment with the smartcore API.

## Roadmap and recent changes

- Trait-system refactor, fewer structs and more object-safe traits, large codebase reorganization.
- Move to Rust 2024 edition (MSRV 1.85) and cleanup of duplicate code paths.
- Seeds and deterministic controls across algorithms using RNG plumbing.
- Search parameter API for hyperparameter exploration in K-Means and SVM families.
- Tree and forest components refactored for reuse; Extra Trees added.
- SVM multiclass support; SVR kernel enum and related improvements.
- XGBoost-style regression introduced; single-linkage clustering implemented.
- Classification metrics hardened: multiclass macro F1 now averages per-class
  F-measures (matching sklearn), and Precision/Recall/F1 share a single
  per-class confusion-counts helper so the per-class bookkeeping lives in one
  place.

See CHANGELOG.md for precise details, deprecations, and breaking changes. Some features like nalgebra-bindings have been dropped in favor of ndarray-only paths. Default features are tuned for WASM/WASI builds; enable serde/datasets as needed.

## Contributing

Contributions are welcome:

- Open an issue describing the change and link it in the PR.
- Keep PRs in sync with the development branch and ensure tests pass on stable Rust (MSRV 1.85, edition 2024).
- Provide or update tests; run clippy and apply formatting. Coverage and linting are part of the workflow.
- Use the provided PR and issue templates to describe behavior changes, new features, and expectations.

If adding IO, prefer abstractions that make non-IO testing straightforward (see readers/iotesting). For datasets, keep serialization helpers in tests gated appropriately to avoid unintended file writes in wasm targets.

## License

smartcore is open source under a permissive license; see Cargo.toml and LICENSE for details. The crate metadata identifies “smartcore Developers” as authors; community contributions are credited via Git history and releases.

## Acknowledgments

smartcore’s design incorporates well-known ML patterns while staying idiomatic to Rust. Thanks to all contributors who have helped expand algorithms, improve docs, modernize traits, and harden the codebase for production.
