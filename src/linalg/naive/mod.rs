//! # Simple Dense Matrix
//!
//! Implements [`BaseMatrix`](../../trait.BaseMatrix.html) and [`BaseVector`](../../trait.BaseVector.html) for [Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html).
//! Data is stored in dense format with [column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//!
//! // 3x3 matrix
//! let A = DenseMatrix::from_2d_array(&[
//!            &[0.9000, 0.4000, 0.7000],
//!            &[0.4000, 0.5000, 0.3000],
//!            &[0.7000, 0.3000, 0.8000],
//!         ]);
//!
//! // row vector
//! let B = DenseMatrix::from_array(1, 3, &[0.9, 0.4, 0.7]);
//!
//! // column vector
//! let C = DenseMatrix::from_vec(3, 1, &vec!(0.9, 0.4, 0.7));
//! ```

/// Add this module to use Dense Matrix
pub mod dense_matrix;
