/// basic data structures for linear algebra constructs: arrays and views
pub mod basic;

/// traits associated to algebraic constructs
pub mod traits;

#[cfg(feature = "ndarray-bindings")]
/// ndarray bindings
pub mod ndarray;
