# AGENTS.md

Agent-focused guidance for working on the `smartcore` Rust machine-learning library.

## Project basics

- **Language / edition**: Rust 2024 (MSRV 1.85 — verified by the `msrv` CI job).
- **Repository**: https://github.com/smartcorelib/smartcore
- **Default branch**: `main`. All changes should target `main` first.
- **License**: Apache-2.0.
- **Authors**: "smartcore Developers".

Always use ASD-STE100 Simplified Technical English

## Build and test

Common commands used in this codebase:

```bash
# Build default (no features)
cargo build

# Build with optional ndarray support
cargo build --features ndarray-bindings

# Build everything
cargo build --all-features

# Run tests
cargo test
cargo test --features ndarray-bindings
cargo test --all-features

# Formatting (enforced in CI)
cargo fmt --all -- --check

# Linting (enforced in CI)
cargo clippy --all-features -- -Drust-2018-idioms -Dwarnings

# Generate and review docs
cargo doc --no-deps --open
```

## Cargo features

Key features defined in `Cargo.toml`:

- `ndarray-bindings` — optional `ndarray` integration.
- `serde` — serialization support (also pulls in `typetag`).
- `datasets` — built-in sample datasets; implies `std_rand` and `serde`.
- `std_rand` — enables standard RNG facilities in `rand`.
- `js` — for `wasm32-unknown-unknown` in-browser usage.

When touching feature-gated code, run at least `cargo build --all-features` and `cargo test --all-features`.

## Code conventions

- Follow the existing **sklearn-inspired API** where possible for a frictionless user experience.
- Keep the library code **pure Rust**. Unsafe code is strongly discouraged; limited low-level exceptions are allowed only with clear justification.
- **Do not use macros in library code**. Prefer explicit, readable implementations.
- Always use **zero-copy data access**. Traverse and transform matrices/vectors through the `iterator(...)` method and the view traits (`ArrayView*`, `MutArrayView*`) instead of cloning into temporary collections.
- Always rely on the bespoke **`numbers/` abstraction** (`Number`, `RealNumber`, `FloatNumber`, ...) for numeric logic; do not hard-code primitive arithmetic on concrete types or pull in external numeric crates.
- Target small/average datasets with a limited memory footprint rather than big-data optimizations.
- Every public module should:
  - Start with a `//!` doc comment that includes references to scientific literature relating the code to research.
  - Provide Rust **doctests** that demonstrate usage.
  - Provide comprehensive unit tests in a `mod tests {}` submodule at the end of the file.
- IO-related code should prefer abstractions that make non-IO testing straightforward (see `readers/iotesting`).
- Dataset serialization helpers should be gated so they do not trigger unintended file writes on wasm targets.

## Pull request workflow

- Open an issue describing the change before starting significant work.
- Search open and closed issues/PRs for related discussion.
- Open PRs against the `main` branch.
- Use the PR template (`.github/PULL_REQUEST_TEMPLATE.md`) and erase sections that do not apply.
- Update `CHANGELOG.md` for breaking changes, new environment variables, exposed ports, useful file locations, and container parameters.
- Ensure CI checks pass:
  - `cargo fmt --all -- --check`
  - `cargo clippy --all-features -- -Drust-2018-idioms -Dwarnings`
  - Full test suite on relevant targets
- A PR requires sign-off from at least one other developer before merging.

## Code structure

High-level layout:

- `src/numbers/` — foundational numeric traits built on `num-traits`.
- `src/linalg/basic/` — core linear-algebra traits:
  - `arrays` — `Array`, `Array1`, `Array2`, view traits (`ArrayView*`, `MutArrayView*`).
  - `matrix` — `DenseMatrix`, the main instantiable matrix type.
  - `vector` — convenience implementations for `std::Vec`.
- `src/linalg/traits/` — theoretical linear-algebra capability traits (`QRDecomposable`, `SVDDecomposable`, `CholeskyDecomposable`, etc.).
- `src/metrics/` — classification, regression, clustering metrics and distance measures.
- `src/linear/`, `src/tree/`, `src/ensemble/`, `src/svm/`, `src/neighbors/`, `src/naive_bayes/`, `src/clustering/`, `src/decomposition/`, `src/preprocessing/` — algorithm modules.
- `src/model_selection/` — cross-validation, search parameters.
- `src/readers/` — CSV and dataset readers.

Most algorithm code is generic over the `numbers` and `linalg` traits rather than concrete types.

## Edition 2024 invariants

The crate was ported from edition 2021 to edition 2024 (#401). The port relies on the following invariants — when touching this code, keep them intact so the edition-2024 lint group (`rust-2024-compatibility`) stays clean:

- **No return-position `impl Trait` (RPIT)**. The codebase returns zero `-> impl Trait` items today. Introducing one would opt into the 2024 lifetime over-capture rules; gate any new RPIT with an explicit lifetime bound and re-run `cargo clippy --all-features -- -Drust-2024-compatibility` before landing.
- **`dyn Trait` bounds carry explicit lifetimes**. All `Box<dyn Trait>` returns are written as `Box<dyn Trait + 'a>` (see `linalg/basic/arrays.rs` and `vector.rs` iterators/views). Do not drop the `+ 'a`; edition 2024 changes the default elision and a bare `dyn Trait` may not mean what you intend.
- **Tail-expression drop order matters**. Edition 2024 reorders temporaries in tail position. When a tail expression owns a temporary that a borrow extends through (e.g. `(self.sub(other)).iterator(0).all(...)`), bind the owned intermediate to a local first:
  ```rust
  // 2024-safe form
  let diff = self.sub(other);
  diff.iterator(0).all(|v| v.abs() <= error)
  ```
  A regression here triggers `tail_expr_drop_order` warnings under `rust-2024-compatibility`.
- **Lint suppressions use `#[expect(...)]` where the lint actually fires, `#[allow(...)]` otherwise**. The crate migrated `#[allow]` → `#[expect]` where clippy confirmed the lint still fires under `--all-features`; sites where the lint does *not* fire (e.g. some `clippy::ptr_arg`/`upper_case_acronyms`/`dead_code` suppressions) remain `#[allow]` to avoid `unfulfilled_lint_expectations` errors. `#[expect]` errors if its lint later stops firing — prefer it for new suppressions where you've verified the lint fires; fall back to `#[allow]` when a lint genuinely doesn't fire and you still want to document intent. Re-run `cargo clippy --all-features -- -Drust-2018-idioms -Drust-2024-compatibility -Dwarnings` before landing.
- **No `unsafe` in library code**. The four remaining `unsafe {}` blocks in `linalg/basic/matrix.rs` (`iterator_mut` raw-pointer traversal for `DenseMatrixMutView`) are tracked by #368 and slated for a safe `split_at_mut` rewrite in a dedicated PR. Until that lands, do not add new `unsafe`; any new `unsafe` requires clear justification and must not trigger `unsafe_op_in_unsafe_fn` (default-warn in 2024) or other unsafe-attr lints.
- **Preserve the bespoke numerical-system logic and performance.** The numeric/linalg traits and their concrete impls (`numbers/`, `linalg/basic/arrays.rs`, `linalg/basic/matrix.rs`, the `HighOrderOperations`/`matmul`/`iterator` paths) are hand-tuned for correctness and speed. When refactoring (e.g. for edition-2024 tail-expr drop order, for #368 `unsafe` removal, or for any other non-behavioral change), **keep the numerical logic intact and do not regress performance**: change only the structural form (bind intermediates, swap the unsafe mechanism), never the math, indexing scheme, traversal order, or allocation strategy. Verify a refactor is behavior-preserving by re-running `cargo test --all-features` (known-answer tests must still pass) before landing.
- **Coverage cfg is not referenced in source**. We never `cfg(coverage)` or `cfg(tarpaulin)` in `src/`, so the 2024 `unexpected_cfgs` lint stays silent. If you add such a cfg, declare it in the `Cargo.toml` `[lints.rust]` `check-cfg` table to avoid the warning.

### Commands for edition-2024 health

```bash
# The rust-2018-idioms gate is still enforced; add the 2024 group locally:
cargo clippy --all-features -- -Drust-2018-idioms -Drust-2024-compatibility -Dwarnings

# MSRV check (mirrors the msrv CI job):
cargo +1.85.0 build --all-features
```

## Conduct

This project follows the [Contributor Covenant Code of Conduct](.github/CODE_OF_CONDUCT.md). Interactions should be respectful and harassment-free.
