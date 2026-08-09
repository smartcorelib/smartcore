# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.3]
### Changed
- Replaced the remaining 4 `unsafe {}` raw-pointer blocks in `linalg/basic/matrix.rs::iterator_mut` / `DenseMatrixMutView::iter_mut` with a safe `split_first_mut`-based helper `ordered_iter_mut` (#368). The traversal order and offset formula are identical to the previous raw-pointer implementation; only the borrow-proving mechanism changed — eliminating `unsafe` from library code entirely.
- **Performance note**: the cross-axis path (axis ≠ natural storage order) now extracts the needed refs via `split_first_mut` in sorted-offset order and reorders, introducing a small allocation. The fast path (axis matches storage order) shortcuts to `values.iter_mut().take(n)` with zero overhead. Benchmarks to quantify the cross-axis delta are tracked in #407.

## [0.6.2]
### Changed
- Ported the crate from Rust edition 2021 to edition 2024 (#401, #402). `cargo fix --edition` made no auto-edits; the only behavioral-adjacent change is `linalg/basic/arrays.rs::approximate_eq`, rewritten to the 2024-safe tail-expr drop-order form (bind the owned intermediate before the borrowing iterator) — numerical logic unchanged.
- Declared `rust-version = "1.85"` (MSRV) in `Cargo.toml` and added an `msrv` CI job that builds with `dtolnay/rust-toolchain@1.85.0` to verify the claim (#404).
- Migrated lint suppressions: `#[allow(...)]` → `#[expect(...)]` at sites where the lint still fires under `--all-features`; `#[allow]` retained where the lint genuinely does not fire (avoids `unfulfilled_lint_expectations`) (#403).
- Added `[lints.rust] unexpected_cfgs` `check-cfg` table in `Cargo.toml` for `cfg(coverage, coverage_nightly)` and `cfg(tarpaulin)` (edition-2024 `unexpected_cfgs` lint).
- `AGENTS.md`: documented the edition-2024 invariants (no RPIT, explicit `dyn Trait + 'a`, tail-expr drop-order, lint-suppression policy, `unsafe` stance) and the "preserve bespoke numerical-system logic and performance" constraint for non-behavioral refactors.

### Fixed
- `svm/svc.rs`: removed two redundant `let svc = ...; svc` tail expressions surfaced by the edition-2024 `clippy::let_and_return` lint.
- `preprocessing/categorical.rs`: kept the nested-`if` form (annotated `#[allow(clippy::collapsible_if)]`) because collapsing to a let-chain requires let-chains, unstable until Rust 1.88 — incompatible with the declared MSRV 1.85.

## [0.6.1]
### Added
- Stage 1 test-coverage push (#392): tests for previously-untested modules.
  - `linalg/traits/high_order.rs`: implemented the `/* TODO */` test module — all 4 `ab` transpose-flag branches, non-square inputs, and a matmul/transpose equivalence check.
  - `linear/lasso_optimizer.rs`: direct tests for `InteriorPointOptimizer` (`new` shape of `ata`, known-answer l1-regularized least squares with `lambda → 0`).
  - `error/mod.rs`: tests for all 6 `Failed` constructors, all 8 `FailedError` variants, both `Display` impls, both `PartialEq` impls, and the `Error` trait impl.
  - `rand_custom.rs`: seeded-RNG determinism and `None`-seed usability tests.

### Changed
- Revived 6 previously-commented-out serde round-trip tests (migrated to `postcard`, the post-#390 serialization backend) for `LinearRegression`, `RidgeRegression`, `Lasso`, `ElasticNet`, `PCA`, `SVD`.
- Fixed a latent type mismatch in the revived `SVD` serde test: the original commented-out code deserialized into `SVD<f32, DenseMatrix<f32>>` but `SVD::fit` on the `f64` iris literals produces `SVD<f64, ...>` — corrected to `SVD<f64, DenseMatrix<f64>>`.
- Renamed two copy-paste-misnamed tests: `dataset::diabetes::boston_dataset` → `diabetes_dataset`; `algorithm::sort::quick_sort::with_capacity` → `quick_argsort`.

## [0.6.0]
### Changed
- CI coverage workflow now includes doctests and enforces a strict 44% line-coverage gate via cargo-tarpaulin (#399).

## [0.5.3]
### Changed
- Classification metrics refactored: `Precision`, `Recall`, and `F1` now derive per-class scores from a single shared `ConfusionCounts` helper (`src/metrics/confusion.rs`) instead of each re-implementing the per-class tp/predicted/support bookkeeping. `Precision` and `Recall` expose a crate-private `per_class_scores_from_counts` used by `F1`'s multiclass path.
- `Precision` and `Recall` now early-return `0.0` on empty input and drop the unreachable `classes == 0` / `support.is_empty()` branches.
- Multiclass macro `F1` (landed in #382, cleaned up in #383) is unchanged behaviourally; it now consumes `Precision`/`Recall::per_class_scores_from_counts` instead of its own `HashMap` bookkeeping.

## [0.4.8] - 2025-11-29
- WARNING: Breaking changes!
- `LassoParameters` and `LassoSearchParameters` have a new field `fit_intercept`. When it is set to false, the `beta_0` term in the formula will be forced to zero, and `intercept` field in `Lasso` will be set to `None`.


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
