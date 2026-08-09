# AGENTS.md

Agent-focused guidance for working on the `smartcore` Rust machine-learning library.

## Project basics

- **Language / edition**: Rust 2021.
- **Repository**: https://github.com/smartcorelib/smartcore
- **Default branch**: `development`. All changes should target `development` first.
- **License**: Apache-2.0.
- **Authors**: "smartcore Developers".

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
- Open PRs against the `development` branch.
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

## Conduct

This project follows the [Contributor Covenant Code of Conduct](.github/CODE_OF_CONDUCT.md). Interactions should be respectful and harassment-free.
