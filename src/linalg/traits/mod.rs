#![allow(clippy::wrong_self_convention)]

pub mod cholesky;
/// The matrix is represented in terms of its eigenvalues and eigenvectors.
pub mod evd;
pub mod high_order;
/// Factors a matrix as the product of a lower triangular matrix and an upper triangular matrix.
pub mod lu;

/// QR factorization that factors a matrix into a product of an orthogonal matrix and an upper triangular matrix.
pub mod qr;
/// statistacal tools for DenseMatrix
pub mod stats;
/// Singular value decomposition.
pub mod svd;
