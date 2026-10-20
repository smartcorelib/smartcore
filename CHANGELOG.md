# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- In the `...Regressor`s the features are always shuffled even when `mtry` equals the number of attributes. This changes the constructed trees. 
- `tree`: `BaseTreeRegressor` (used by `DecisionTreeRegressor`, `RandomForestRegressor` and `ExtraTreesRegressor`) uses a new internal `Criterion` trait with `MSECriterion`. The MSE split improvement uses a numerically stable formula.  The selected splits do not change when a constant is added to y. Exactly tied splits can be broken in a different way, so fitted trees can differ slightly from earlier versions.
- `tree`: in the `...Regressor`s, rows with sample weight 0 are removed before the tree grows. They no longer count toward `min_samples_leaf` and `min_samples_split`, and they no longer move a split threshold. A tree fitted with a zero-weight row is the same as a tree fitted without that row.
- `tree`, `ensemble`: `DecisionTreeRegressor`, `RandomForestRegressor` and `ExtraTreesRegressor` get a `compute_feature_importances` method. 
- `tree`, `ensemble`: the fitted `DecisionTreeRegressor`, `RandomForestRegressor` and `ExtraTreesRegressor` store new fields that the feature importances use. Models serialized with an earlier version cannot be deserialized. Fit these models again.

## [0.6.16]
### Fixed
- `predict`, `predict_proba`, `predict_oob`, `predict_matrix` and `decision_function` now return `Err(Failed)` instead of panicking when `fit` has not run (#469, #470). Models return the same unfitted error even after deserialization, when the state fields are `None`. The guarded methods cover the decision trees, the random forests, the extra-trees regressor, the KNN classifier and regressor, `SVC`, `MultiClassSVC`, `SVR`, the linear models, the naive Bayes classifiers and `XGRegressor`.
- `KNNClassifier::fit` now accepts `k = 1`, as `KNNRegressor` already did (#476). `k = 0` is still rejected.
- The ndarray implementation of `Array2::from_iterator` now preserves column-major input order, including for rectangular matrices that previously caused a panic. Column-major input produces a column-major array (#482).
- `ensemble`: `RandomForestRegressor` and `RandomForestClassifier` now give each tree its own seed (`seed + tree index`). Before this change, all trees of a forest used the same seed, so all trees picked the same feature samples (#475). A forest fitted with a fixed `seed` now returns different predictions, because the trees no longer share one feature sample. Fits with the same seed stay reproducible.
- `ensemble`: sample weights now follow the scikit-learn rules. With `bootstrap = true`, the weights are used in the bootstrap phase only. With `bootstrap = false`, the weights go to each tree (#478). New tests compare `RandomForestRegressor` and `ExtraTreesRegressor` predictions (train, probe and OOB) with reference values from scikit-learn 1.9.1.

### Changed
- The error text for an unfitted `MultiClassSVC::predict` changed from "MultiClassSVC is not fitted" to the common "'fit' should be called before calling 'predict'". Code that matched the old string must be updated.
- The error text for an invalid `k` in `KNNClassifier::fit` changed from "k should be > 1" to "k should be > 0", the same text `KNNRegressor` uses. Code that matched the old string must be updated.
- `tree`: `BaseTreeRegressor` (used by `DecisionTreeRegressor`, `RandomForestRegressor` and `ExtraTreesRegressor`) runs faster: less allocation, less sorting, node rows stored as `(start, end)` spans, and no partition step for child nodes that cannot split (#481). Split values no longer fall between adjacent floating-point numbers, so fitted values can differ slightly from earlier versions. New tests check parity of `RandomForestRegressor` with scikit-learn.
- `linear`: the `test_logit` fixture now uses a 10000 x 94 random matrix instead of 52181 x 94. The class ratio stays the same (418 of 10000 rows are positive). The test checks the same property and runs much faster, so it stays inside the CI coverage time budget.
- CI: the coverage job runs cargo-tarpaulin with the LLVM engine, in release mode and with a 600 s timeout, instead of the default engine (#485, fixes #484).

## [0.6.15]
### Fixed
- `tree`: regression tests for the tree growth fix from #464. New `min_samples_split_boundary` tests pin the split rule for `DecisionTreeClassifier` and `BaseTreeRegressor`: a node that holds exactly `min_samples_split` samples must still split, and a node with fewer samples must stay a leaf. The `full_depth` tests now also assert the tree `depth` (3), which guards the public `DecisionTreeClassifier::depth` accessor against silent regressions. Library code is unchanged.

## [0.6.14]
### Added
- `decomposition/lda.rs`: `LDA`, linear discriminant analysis for supervised dimensionality reduction (#136). It projects the data onto the directions that best separate the classes, keeping `min(n_classes - 1, n_features)` components by default, and implements the `Transformer` interface next to `PCA`. Directions match scikit-learn's `LinearDiscriminantAnalysis(solver="eigen")` up to sign.

## [0.6.13]
### Fixed
- `xgboost/xgb_regressor.rs`: `XGRegressor::fit` no longer panics when `subsample` is less than 1.0 on a small dataset (#444). The sample for each tree now keeps a minimum of one row, as scikit-learn does for its own `subsample` parameter. Sample sizes of one row or more are unchanged.

## [0.6.11]
### Fixed
- `algorithm/neighbour/cosinepair.rs`: `CosinePair::query_row_top_k` now returns exact nearest neighbours whenever `approximate` is `false` (the default). Previously the query always sampled only `top_k` evenly strided candidate rows without documentation, and the bounded candidate heap evicted its closest entry, so the method could return the farthest of the sampled rows (#442). Strided sampling is now gated behind `CosinePairParameters { approximate: true, .. }` and is documented as approximate.
- `algorithm/neighbour/cosinepair.rs`: `CosinePair` construction now evaluates each unordered row pair once (symmetric half-scan), precomputes row norms once in O(n·d), and scores pairs through zero-copy row views instead of materialising two `Vec`s per pair (#442). Distances are unchanged (bit-identical formula and operation order as `Cosine::new().distance(...)`); measured build time drops ~3x on a 1500x64 input. Construction remains Theta(n^2) dot products — `top_k` does not make it sub-quadratic; module and method docs now state both facts.

### Changed
- **Breaking**: `CosinePair` gained a private `row_norms` field holding the precomputed row norms. Construct the structure through `new` / `with_top_k` / `with_parameters` instead of struct literals.

## [0.6.10]
### Fixed
- `model_selection`: pinned the `KFold` seed in the `test_cross_val_predict_knn` and `test_cross_validate_knn` unit tests. Under `--all-features` (`std_rand`) an unseeded `KFold` draws OS entropy, so each CI run shuffled the folds differently; a sweep of 20 000 seeds showed 0.17% of shuffles violate the `MAE < 10.0` assertion (worst 12.81) and 0.01% violate `train_score < test_score`, making CI flaky. Library behaviour is unchanged; the entropy-seeded path stays covered by the `rand_custom` tests.

## [0.6.9]
### Fixed
- `metrics`: `precision`, `recall`, and `f1` (the free functions, the `Precision` / `Recall` / `F1` metric structs, and the matching `ClassificationMetrics` entry points) now accept any label type that implements `Number`, including ordered integers such as `u16` or `i32`; labels no longer need to implement `RealNumber` or `FloatNumber` (#322). The same integer labels can now feed `RandomForestClassifier::fit` and classification metrics inside `model_selection::cross_validate`. Class keys are derived through a shared `f64` conversion instead of raw float bit transmutation; scores for float inputs are unchanged.

## [0.6.8]
### Fixed
- `linear/linear_regression.rs`: `LinearRegression::fit` / `fit_matrix` now return `Err(Failed::fit(...))` instead of panicking when the intercept-augmented system is underdetermined, i.e. `n_features + 1 > n_samples` (#435). Both the default SVD solver and the QR solver are covered.
- `linalg/traits/svd.rs`: `svd_solve` / `svd_solve_mut` reject systems where _A_ has more columns than rows with `FailedError::SolutionFailed` instead of writing the solution out of `b`'s bounds; `SVD::solve` gained a matching defensive guard.
- `linalg/traits/qr.rs`: `qr_solve_mut` returns `Err(Failed)` ("Matrix is rank deficient.") for rank-deficient systems (duplicate/collinear columns, underdetermined shapes) instead of panicking.

## [0.6.7]
### Added
- `linear/linear_regression.rs`: native multi-output matrix support (#432, #433). `LinearRegression::fit_matrix` fits _N x K_ targets directly (both QR and SVD solvers); `predict_matrix` returns the _K_-column prediction matrix; `intercept_matrix` exposes the _1 x K_ intercept row.

### Changed
- **Breaking**: `LinearRegression` field `intercept` changed from `Option<TX>` to `Option<X>` (a _1 x K_ matrix) and the struct gained `PhantomData<TX>`. Previously serialized models will not deserialize; re-fit or migrate saved models. `intercept()` still returns a scalar for single-output callers.

## [0.6.4]
### Added
- Stage 2 test-coverage push (#393): `proptest` dev-dependency + property-based invariant tests and linalg edge cases.
  - `linalg/basic/arrays.rs`: proptest invariants — transpose involution `(A^T)^T == A`, matmul with identity `A*I == A`, matmul associativity `(AB)C ≈ A(BC)` (approximate comparison for FP), `(AB)^T == B^T A^T`, reshape preserves element count. Edge cases — 1x1 matmul, row×col matmul, shape-mismatch panic, reshape-incompatible panic, 1xN transpose.
  - `algorithm/sort/quick_sort.rs`: proptest — `quick_argsort` produces a valid permutation (all indices present exactly once, values non-decreasing in permutation order).
  - `metrics/distance/euclidian.rs`: proptest — `d(a,a) == 0`, symmetry `d(a,b) == d(b,a)`, triangle inequality `d(a,c) ≤ d(a,b) + d(b,c)`.

### Changed
- Added `proptest = "1.5"` to `[dev-dependencies]`.

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
