#![allow(clippy::wrong_self_convention)]
//! # Linear Algebra and Matrix Decomposition
//!
//! Most machine learning algorithms in SmartCore depend on linear algebra and matrix decomposition methods from this module.
//!
//! Traits [`BaseMatrix`](trait.BaseMatrix.html), [`Matrix`](trait.Matrix.html) and [`BaseVector`](trait.BaseVector.html) define
//! abstract methods that can be implemented for any two-dimensional and one-dimentional arrays (matrix and vector).
//! Functions from these traits are designed for SmartCore machine learning algorithms and should not be used directly in your code.
//! If you still want to use functions from `BaseMatrix`, `Matrix` and `BaseVector` please be aware that methods defined in these
//! traits might change in the future.
//!
//! One reason why linear algebra traits are public is to allow for different types of matrices and vectors to be plugged into SmartCore.
//! Once all methods defined in `BaseMatrix`, `Matrix` and `BaseVector` are implemented for your favourite type of matrix and vector you
//! should be able to run SmartCore algorithms on it. Please see `nalgebra_bindings` and `ndarray_bindings` modules for an example of how
//! it is done for other libraries.
//!
//! You will also find verious matrix decomposition methods that work for any matrix that extends [`Matrix`](trait.Matrix.html).
//! For example, to decompose matrix defined as [Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html):
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::linalg::svd::*;
//!
//! let A = DenseMatrix::from_2d_array(&[
//!            &[0.9000, 0.4000, 0.7000],
//!            &[0.4000, 0.5000, 0.3000],
//!            &[0.7000, 0.3000, 0.8000],
//!         ]);
//!
//! let svd = A.svd().unwrap();
//!
//! let s: Vec<f64> = svd.s;
//! let v: DenseMatrix<f64> = svd.V;
//! let u: DenseMatrix<f64> = svd.U;
//! ```

pub mod cholesky;
/// The matrix is represented in terms of its eigenvalues and eigenvectors.
pub mod evd;
pub mod high_order;
/// Factors a matrix as the product of a lower triangular matrix and an upper triangular matrix.
pub mod lu;
/// Dense matrix with column-major order that wraps [Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html).
pub mod naive;
/// [nalgebra](https://docs.rs/nalgebra/) bindings.
#[cfg(feature = "nalgebra-bindings")]
pub mod nalgebra_bindings;
/// [ndarray](https://docs.rs/ndarray) bindings.
#[cfg(feature = "ndarray-bindings")]
pub mod ndarray_bindings;
/// QR factorization that factors a matrix into a product of an orthogonal matrix and an upper triangular matrix.
pub mod qr;
pub mod stats;
/// Singular value decomposition.
pub mod svd;

use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Range;

use crate::math::num::RealNumber;
use cholesky::CholeskyDecomposableMatrix;
use evd::EVDDecomposableMatrix;
use high_order::HighOrderOperations;
use lu::LUDecomposableMatrix;
use qr::QRDecomposableMatrix;
use stats::{MatrixPreprocessing, MatrixStats};
use std::fs;
use svd::SVDDecomposableMatrix;

use crate::readers;

/// Column or row vector
pub trait BaseVector<T: RealNumber>: Clone + Debug {
    /// Get an element of a vector
    /// * `i` - index of an element
    fn get(&self, i: usize) -> T;

    /// Set an element at `i` to `x`
    /// * `i` - index of an element
    /// * `x` - new value
    fn set(&mut self, i: usize, x: T);

    /// Get number of elevemnt in the vector
    fn len(&self) -> usize;

    /// Returns true if the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a new vector from a &[T]
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    /// let a: [f64; 5] = [0., 0.5, 2., 3., 4.];
    /// let v: Vec<f64> = BaseVector::from_array(&a);
    /// assert_eq!(v, vec![0., 0.5, 2., 3., 4.]);
    /// ```
    fn from_array(f: &[T]) -> Self {
        let mut v = Self::zeros(f.len());
        for (i, elem) in f.iter().enumerate() {
            v.set(i, *elem);
        }
        v
    }

    /// Return a vector with the elements of the one-dimensional array.
    fn to_vec(&self) -> Vec<T>;

    /// Create new vector with zeros of size `len`.
    fn zeros(len: usize) -> Self;

    /// Create new vector with ones of size `len`.
    fn ones(len: usize) -> Self;

    /// Create new vector of size `len` where each element is set to `value`.
    fn fill(len: usize, value: T) -> Self;

    /// Vector dot product
    fn dot(&self, other: &Self) -> T;

    /// Returns True if matrices are element-wise equal within a tolerance `error`.
    fn approximate_eq(&self, other: &Self, error: T) -> bool;

    /// Returns [L2 norm] of the vector(https://en.wikipedia.org/wiki/Matrix_norm).
    fn norm2(&self) -> T;

    /// Returns [vectors norm](https://en.wikipedia.org/wiki/Matrix_norm) of order `p`.
    fn norm(&self, p: T) -> T;

    /// Divide single element of the vector by `x`, write result to original vector.
    fn div_element_mut(&mut self, pos: usize, x: T);

    /// Multiply single element of the vector by `x`, write result to original vector.
    fn mul_element_mut(&mut self, pos: usize, x: T);

    /// Add single element of the vector to `x`, write result to original vector.
    fn add_element_mut(&mut self, pos: usize, x: T);

    /// Subtract `x` from single element of the vector, write result to original vector.
    fn sub_element_mut(&mut self, pos: usize, x: T);

    /// Subtract scalar
    fn sub_scalar_mut(&mut self, x: T) -> &Self {
        for i in 0..self.len() {
            self.set(i, self.get(i) - x);
        }
        self
    }

    /// Subtract scalar
    fn add_scalar_mut(&mut self, x: T) -> &Self {
        for i in 0..self.len() {
            self.set(i, self.get(i) + x);
        }
        self
    }

    /// Subtract scalar
    fn mul_scalar_mut(&mut self, x: T) -> &Self {
        for i in 0..self.len() {
            self.set(i, self.get(i) * x);
        }
        self
    }

    /// Subtract scalar
    fn div_scalar_mut(&mut self, x: T) -> &Self {
        for i in 0..self.len() {
            self.set(i, self.get(i) / x);
        }
        self
    }

    /// Add vectors, element-wise
    fn add_scalar(&self, x: T) -> Self {
        let mut r = self.clone();
        r.add_scalar_mut(x);
        r
    }

    /// Subtract vectors, element-wise
    fn sub_scalar(&self, x: T) -> Self {
        let mut r = self.clone();
        r.sub_scalar_mut(x);
        r
    }

    /// Multiply vectors, element-wise
    fn mul_scalar(&self, x: T) -> Self {
        let mut r = self.clone();
        r.mul_scalar_mut(x);
        r
    }

    /// Divide vectors, element-wise
    fn div_scalar(&self, x: T) -> Self {
        let mut r = self.clone();
        r.div_scalar_mut(x);
        r
    }

    /// Add vectors, element-wise, overriding original vector with result.
    fn add_mut(&mut self, other: &Self) -> &Self;

    /// Subtract vectors, element-wise, overriding original vector with result.
    fn sub_mut(&mut self, other: &Self) -> &Self;

    /// Multiply vectors, element-wise, overriding original vector with result.
    fn mul_mut(&mut self, other: &Self) -> &Self;

    /// Divide vectors, element-wise, overriding original vector with result.
    fn div_mut(&mut self, other: &Self) -> &Self;

    /// Add vectors, element-wise
    fn add(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.add_mut(other);
        r
    }

    /// Subtract vectors, element-wise
    fn sub(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.sub_mut(other);
        r
    }

    /// Multiply vectors, element-wise
    fn mul(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.mul_mut(other);
        r
    }

    /// Divide vectors, element-wise
    fn div(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.div_mut(other);
        r
    }

    /// Calculates sum of all elements of the vector.
    fn sum(&self) -> T;

    /// Returns unique values from the vector.
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    /// let a = vec!(1., 2., 2., -2., -6., -7., 2., 3., 4.);
    ///
    ///assert_eq!(a.unique(), vec![-7., -6., -2., 1., 2., 3., 4.]);
    /// ```
    fn unique(&self) -> Vec<T>;

    /// Computes the arithmetic mean.
    fn mean(&self) -> T {
        self.sum() / T::from_usize(self.len()).unwrap()
    }
    /// Computes variance.
    fn var(&self) -> T {
        let n = self.len();

        let mut mu = T::zero();
        let mut sum = T::zero();
        let div = T::from_usize(n).unwrap();
        for i in 0..n {
            let xi = self.get(i);
            mu += xi;
            sum += xi * xi;
        }
        mu /= div;
        sum / div - mu.powi(2)
    }
    /// Computes the standard deviation.
    fn std(&self) -> T {
        self.var().sqrt()
    }

    /// Copies content of `other` vector.
    fn copy_from(&mut self, other: &Self);

    /// Take elements from an array.
    fn take(&self, index: &[usize]) -> Self {
        let n = index.len();

        let mut result = Self::zeros(n);

        for (i, idx) in index.iter().enumerate() {
            result.set(i, self.get(*idx));
        }

        result
    }
}

/// Generic matrix type.
pub trait BaseMatrix<T: RealNumber>: Clone + Debug {
    /// Row vector that is associated with this matrix type,
    /// e.g. if we have an implementation of sparce matrix
    /// we should have an associated sparce vector type that
    /// represents a row in this matrix.
    type RowVector: BaseVector<T> + Clone + Debug;

    /// Create a matrix from a csv file.
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::DenseMatrix;
    /// use smartcore::linalg::BaseMatrix;
    /// use smartcore::readers::csv;
    /// use std::fs;
    ///
    /// fs::write("identity.csv", "header\n1.0,0.0\n0.0,1.0");
    /// assert_eq!(
    ///     DenseMatrix::<f64>::from_csv("identity.csv", csv::CSVDefinition::default()).unwrap(),
    ///     DenseMatrix::from_row_vectors(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap()
    /// );
    /// fs::remove_file("identity.csv");
    /// ```
    fn from_csv(
        path: &str,
        definition: readers::csv::CSVDefinition<'_>,
    ) -> Result<Self, readers::ReadingError> {
        readers::csv::matrix_from_csv_source(fs::File::open(path)?, definition)
    }

    /// Transforms row vector `vec` into a 1xM matrix.
    fn from_row_vector(vec: Self::RowVector) -> Self;

    /// Transforms Vector of n rows with dimension m into
    /// a matrix nxm.
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::DenseMatrix;
    /// use crate::smartcore::linalg::BaseMatrix;
    ///
    /// let eye = DenseMatrix::from_row_vectors(vec![vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.]])
    ///     .unwrap();
    ///
    /// assert_eq!(
    ///    eye,
    ///    DenseMatrix::from_2d_vec(&vec![
    ///        vec![1.0, 0.0, 0.0],
    ///        vec![0.0, 1.0, 0.0],
    ///        vec![0.0, 0.0, 1.0],
    ///    ])
    /// );
    fn from_row_vectors(rows: Vec<Self::RowVector>) -> Option<Self> {
        if rows.is_empty() {
            return None;
        }
        let n = rows.len();
        let m = rows[0].len();

        let mut result = Self::zeros(n, m);

        for (row_idx, row) in rows.into_iter().enumerate() {
            result.set_row(row_idx, row);
        }

        Some(result)
    }

    /// Transforms 1-d matrix of 1xM into a row vector.
    fn to_row_vector(self) -> Self::RowVector;

    /// Get an element of the matrix.
    /// * `row` - row number
    /// * `col` - column number
    fn get(&self, row: usize, col: usize) -> T;

    /// Get a vector with elements of the `row`'th row
    /// * `row` - row number
    fn get_row_as_vec(&self, row: usize) -> Vec<T>;

    /// Get the `row`'th row
    /// * `row` - row number
    fn get_row(&self, row: usize) -> Self::RowVector;

    /// Copies a vector with elements of the `row`'th row into `result`
    /// * `row` - row number
    /// * `result` - receiver for the row
    fn copy_row_as_vec(&self, row: usize, result: &mut Vec<T>);

    /// Set row vector at row `row_idx`.
    fn set_row(&mut self, row_idx: usize, row: Self::RowVector) {
        for (col_idx, val) in row.to_vec().into_iter().enumerate() {
            self.set(row_idx, col_idx, val);
        }
    }

    /// Get a vector with elements of the `col`'th column
    /// * `col` - column number
    fn get_col_as_vec(&self, col: usize) -> Vec<T>;

    /// Copies a vector with elements of the `col`'th column into `result`
    /// * `col` - column number
    /// * `result` - receiver for the col
    fn copy_col_as_vec(&self, col: usize, result: &mut Vec<T>);

    /// Set an element at `col`, `row` to `x`
    fn set(&mut self, row: usize, col: usize, x: T);

    /// Create an identity matrix of size `size`
    fn eye(size: usize) -> Self;

    /// Create new matrix with zeros of size `nrows` by `ncols`.
    fn zeros(nrows: usize, ncols: usize) -> Self;

    /// Create new matrix with ones of size `nrows` by `ncols`.
    fn ones(nrows: usize, ncols: usize) -> Self;

    /// Create new matrix of size `nrows` by `ncols` where each element is set to `value`.
    fn fill(nrows: usize, ncols: usize, value: T) -> Self;

    /// Return the shape of an array.
    fn shape(&self) -> (usize, usize);

    /// Stack arrays in sequence vertically (row wise).
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    ///
    /// let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]);
    /// let b = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.]]);
    /// let expected = DenseMatrix::from_2d_array(&[
    ///     &[1., 2., 3., 1., 2.],
    ///     &[4., 5., 6., 3., 4.]
    /// ]);
    ///
    /// assert_eq!(a.h_stack(&b), expected);
    /// ```
    fn h_stack(&self, other: &Self) -> Self;

    /// Stack arrays in sequence horizontally (column wise).
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    ///
    /// let a = DenseMatrix::from_array(1, 3, &[1., 2., 3.]);
    /// let b = DenseMatrix::from_array(1, 3, &[4., 5., 6.]);
    /// let expected = DenseMatrix::from_2d_array(&[
    ///     &[1., 2., 3.],
    ///     &[4., 5., 6.]
    /// ]);
    ///
    /// assert_eq!(a.v_stack(&b), expected);
    /// ```
    fn v_stack(&self, other: &Self) -> Self;

    /// Matrix product.
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    ///
    /// let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.]]);
    /// let expected = DenseMatrix::from_2d_array(&[
    ///     &[7., 10.],
    ///     &[15., 22.]
    /// ]);
    ///
    /// assert_eq!(a.matmul(&a), expected);
    /// ```
    fn matmul(&self, other: &Self) -> Self;

    /// Vector dot product
    /// Both matrices should be of size _1xM_
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    ///
    /// let a = DenseMatrix::from_array(1, 3, &[1., 2., 3.]);
    /// let b = DenseMatrix::from_array(1, 3, &[4., 5., 6.]);
    ///
    /// assert_eq!(a.dot(&b), 32.);
    /// ```
    fn dot(&self, other: &Self) -> T;

    /// Return a slice of the matrix.
    /// * `rows` - range of rows to return
    /// * `cols` - range of columns to return
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    ///
    /// let m = DenseMatrix::from_2d_array(&[
    ///             &[1., 2., 3., 1.],
    ///             &[4., 5., 6., 3.],
    ///             &[7., 8., 9., 5.]
    ///         ]);
    /// let expected = DenseMatrix::from_2d_array(&[&[2., 3.], &[5., 6.]]);
    /// let result = m.slice(0..2, 1..3);
    /// assert_eq!(result, expected);
    /// ```
    fn slice(&self, rows: Range<usize>, cols: Range<usize>) -> Self;

    /// Returns True if matrices are element-wise equal within a tolerance `error`.
    fn approximate_eq(&self, other: &Self, error: T) -> bool;

    /// Add matrices, element-wise, overriding original matrix with result.
    fn add_mut(&mut self, other: &Self) -> &Self;

    /// Subtract matrices, element-wise, overriding original matrix with result.
    fn sub_mut(&mut self, other: &Self) -> &Self;

    /// Multiply matrices, element-wise, overriding original matrix with result.
    fn mul_mut(&mut self, other: &Self) -> &Self;

    /// Divide matrices, element-wise, overriding original matrix with result.
    fn div_mut(&mut self, other: &Self) -> &Self;

    /// Divide single element of the matrix by `x`, write result to original matrix.
    fn div_element_mut(&mut self, row: usize, col: usize, x: T);

    /// Multiply single element of the matrix by `x`, write result to original matrix.
    fn mul_element_mut(&mut self, row: usize, col: usize, x: T);

    /// Add single element of the matrix to `x`, write result to original matrix.
    fn add_element_mut(&mut self, row: usize, col: usize, x: T);

    /// Subtract `x` from single element of the matrix, write result to original matrix.
    fn sub_element_mut(&mut self, row: usize, col: usize, x: T);

    /// Add matrices, element-wise
    fn add(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.add_mut(other);
        r
    }

    /// Subtract matrices, element-wise
    fn sub(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.sub_mut(other);
        r
    }

    /// Multiply matrices, element-wise
    fn mul(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.mul_mut(other);
        r
    }

    /// Divide matrices, element-wise
    fn div(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.div_mut(other);
        r
    }

    /// Add `scalar` to the matrix, override original matrix with result.
    fn add_scalar_mut(&mut self, scalar: T) -> &Self;

    /// Subtract `scalar` from the elements of matrix, override original matrix with result.
    fn sub_scalar_mut(&mut self, scalar: T) -> &Self;

    /// Multiply `scalar` by the elements of matrix, override original matrix with result.
    fn mul_scalar_mut(&mut self, scalar: T) -> &Self;

    /// Divide elements of the matrix by `scalar`, override original matrix with result.
    fn div_scalar_mut(&mut self, scalar: T) -> &Self;

    /// Add `scalar` to the matrix.
    fn add_scalar(&self, scalar: T) -> Self {
        let mut r = self.clone();
        r.add_scalar_mut(scalar);
        r
    }

    /// Subtract `scalar` from the elements of matrix.
    fn sub_scalar(&self, scalar: T) -> Self {
        let mut r = self.clone();
        r.sub_scalar_mut(scalar);
        r
    }

    /// Multiply `scalar` by the elements of matrix.
    fn mul_scalar(&self, scalar: T) -> Self {
        let mut r = self.clone();
        r.mul_scalar_mut(scalar);
        r
    }

    /// Divide elements of the matrix by `scalar`.
    fn div_scalar(&self, scalar: T) -> Self {
        let mut r = self.clone();
        r.div_scalar_mut(scalar);
        r
    }

    /// Reverse or permute the axes of the matrix, return new matrix.
    fn transpose(&self) -> Self;

    /// Create new `nrows` by `ncols` matrix and populate it with random samples from a uniform distribution over [0, 1).
    fn rand(nrows: usize, ncols: usize) -> Self;

    /// Returns [L2 norm](https://en.wikipedia.org/wiki/Matrix_norm).
    fn norm2(&self) -> T;

    /// Returns [matrix norm](https://en.wikipedia.org/wiki/Matrix_norm) of order `p`.
    fn norm(&self, p: T) -> T;

    /// Returns the average of the matrix columns.
    fn column_mean(&self) -> Vec<T>;

    /// Numerical negative, element-wise. Overrides original matrix.
    fn negative_mut(&mut self);

    /// Numerical negative, element-wise.
    fn negative(&self) -> Self {
        let mut result = self.clone();
        result.negative_mut();
        result
    }

    /// Returns new matrix of shape `nrows` by `ncols` with data copied from original matrix.
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    ///
    /// let a = DenseMatrix::from_array(1, 6, &[1., 2., 3., 4., 5., 6.]);
    /// let expected = DenseMatrix::from_2d_array(&[
    ///             &[1., 2., 3.],
    ///             &[4., 5., 6.]
    ///         ]);
    ///
    /// assert_eq!(a.reshape(2, 3), expected);
    /// ```
    fn reshape(&self, nrows: usize, ncols: usize) -> Self;

    /// Copies content of `other` matrix.
    fn copy_from(&mut self, other: &Self);

    /// Calculate the absolute value element-wise. Overrides original matrix.
    fn abs_mut(&mut self) -> &Self;

    /// Calculate the absolute value element-wise.
    fn abs(&self) -> Self {
        let mut result = self.clone();
        result.abs_mut();
        result
    }

    /// Calculates sum of all elements of the matrix.
    fn sum(&self) -> T;

    /// Calculates max of all elements of the matrix.
    fn max(&self) -> T;

    /// Calculates min of all elements of the matrix.
    fn min(&self) -> T;

    /// Calculates max(|a - b|) of two matrices
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    ///
    /// let a = DenseMatrix::from_array(2, 3, &[1., 2., 3., 4., -5., 6.]);
    /// let b = DenseMatrix::from_array(2, 3, &[2., 3., 4., 1., 0., -12.]);
    ///
    /// assert_eq!(a.max_diff(&b), 18.);
    /// assert_eq!(b.max_diff(&b), 0.);
    /// ```
    fn max_diff(&self, other: &Self) -> T {
        self.sub(other).abs().max()
    }

    /// Calculates [Softmax function](https://en.wikipedia.org/wiki/Softmax_function). Overrides the matrix with result.
    fn softmax_mut(&mut self);

    /// Raises elements of the matrix to the power of `p`
    fn pow_mut(&mut self, p: T) -> &Self;

    /// Returns new matrix with elements raised to the power of `p`
    fn pow(&mut self, p: T) -> Self {
        let mut result = self.clone();
        result.pow_mut(p);
        result
    }

    /// Returns the indices of the maximum values in each row.
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    /// let a = DenseMatrix::from_array(2, 3, &[1., 2., 3., -5., -6., -7.]);
    ///
    /// assert_eq!(a.argmax(), vec![2, 0]);
    /// ```
    fn argmax(&self) -> Vec<usize>;

    /// Returns vector with unique values from the matrix.
    /// ```
    /// use smartcore::linalg::naive::dense_matrix::*;
    /// let a = DenseMatrix::from_array(3, 3, &[1., 2., 2., -2., -6., -7., 2., 3., 4.]);
    ///
    ///assert_eq!(a.unique(), vec![-7., -6., -2., 1., 2., 3., 4.]);
    /// ```
    fn unique(&self) -> Vec<T>;

    /// Calculates the covariance matrix
    fn cov(&self) -> Self;

    /// Take elements from an array along an axis.
    fn take(&self, index: &[usize], axis: u8) -> Self {
        let (n, p) = self.shape();

        let k = match axis {
            0 => p,
            _ => n,
        };

        let mut result = match axis {
            0 => Self::zeros(index.len(), p),
            _ => Self::zeros(n, index.len()),
        };

        for (i, idx) in index.iter().enumerate() {
            for j in 0..k {
                match axis {
                    0 => result.set(i, j, self.get(*idx, j)),
                    _ => result.set(j, i, self.get(j, *idx)),
                };
            }
        }

        result
    }
    /// Take an individual column from the matrix.
    fn take_column(&self, column_index: usize) -> Self {
        self.take(&[column_index], 1)
    }
}

/// Generic matrix with additional mixins like various factorization methods.
pub trait Matrix<T: RealNumber>:
    BaseMatrix<T>
    + SVDDecomposableMatrix<T>
    + EVDDecomposableMatrix<T>
    + QRDecomposableMatrix<T>
    + LUDecomposableMatrix<T>
    + CholeskyDecomposableMatrix<T>
    + MatrixStats<T>
    + MatrixPreprocessing<T>
    + HighOrderOperations<T>
    + PartialEq
    + Display
{
}

pub(crate) fn row_iter<F: RealNumber, M: BaseMatrix<F>>(m: &M) -> RowIter<'_, F, M> {
    RowIter {
        m,
        pos: 0,
        max_pos: m.shape().0,
        phantom: PhantomData,
    }
}

pub(crate) struct RowIter<'a, T: RealNumber, M: BaseMatrix<T>> {
    m: &'a M,
    pos: usize,
    max_pos: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: RealNumber, M: BaseMatrix<T>> Iterator for RowIter<'a, T, M> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        let res = if self.pos < self.max_pos {
            Some(self.m.get_row_as_vec(self.pos))
        } else {
            None
        };
        self.pos += 1;
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    use crate::linalg::BaseMatrix;
    use crate::linalg::BaseVector;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mean() {
        let m = vec![1., 2., 3.];

        assert_eq!(m.mean(), 2.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn std() {
        let m = vec![1., 2., 3.];

        assert!((m.std() - 0.81f64).abs() < 1e-2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn var() {
        let m = vec![1., 2., 3., 4.];

        assert!((m.var() - 1.25f64).abs() < std::f64::EPSILON);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vec_take() {
        let m = vec![1., 2., 3., 4., 5.];

        assert_eq!(m.take(&vec!(0, 0, 4, 4)), vec![1., 1., 5., 5.]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn take() {
        let m = DenseMatrix::from_2d_array(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
            &[7.0, 8.0],
            &[9.0, 10.0],
        ]);

        let expected_0 = DenseMatrix::from_2d_array(&[&[3.0, 4.0], &[3.0, 4.0], &[7.0, 8.0]]);

        let expected_1 = DenseMatrix::from_2d_array(&[
            &[2.0, 1.0],
            &[4.0, 3.0],
            &[6.0, 5.0],
            &[8.0, 7.0],
            &[10.0, 9.0],
        ]);

        assert_eq!(m.take(&vec!(1, 1, 3), 0), expected_0);
        assert_eq!(m.take(&vec!(1, 0), 1), expected_1);
    }

    #[test]
    fn take_second_column_from_matrix() {
        let four_columns: DenseMatrix<f64> = DenseMatrix::from_2d_array(&[
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, 1.0, 2.0, 3.0],
        ]);

        let second_column = four_columns.take_column(1);
        assert_eq!(
            second_column,
            DenseMatrix::from_2d_array(&[&[1.0], &[1.0], &[1.0], &[1.0]]),
            "The second column was not extracted correctly"
        );
    }

    #[test]
    fn test_from_row_vectors_simple() {
        let eye = DenseMatrix::from_row_vectors(vec![
            vec![1., 0., 0.],
            vec![0., 1., 0.],
            vec![0., 0., 1.],
        ])
        .unwrap();
        assert_eq!(
            eye,
            DenseMatrix::from_2d_vec(&vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ])
        );
    }

    #[test]
    fn test_from_row_vectors_large() {
        let eye = DenseMatrix::from_row_vectors(vec![vec![4.25; 5000]; 5000]).unwrap();

        assert_eq!(eye.shape(), (5000, 5000));
        assert_eq!(eye.get_row(5), vec![4.25; 5000]);
    }
    mod matrix_from_csv {

        use crate::linalg::naive::dense_matrix::DenseMatrix;
        use crate::linalg::BaseMatrix;
        use crate::readers::csv;
        use crate::readers::io_testing;
        use crate::readers::ReadingError;

        #[test]
        fn simple_read_default_csv() {
            let test_csv_file = io_testing::TemporaryTextFile::new(
                "'sepal.length','sepal.width','petal.length','petal.width'\n\
                5.1,3.5,1.4,0.2\n\
                4.9,3,1.4,0.2\n\
                4.7,3.2,1.3,0.2",
            );

            assert_eq!(
                DenseMatrix::<f64>::from_csv(
                    test_csv_file
                        .expect("Temporary file could not be written.")
                        .path(),
                    csv::CSVDefinition::default()
                ),
                Ok(DenseMatrix::from_2d_array(&[
                    &[5.1, 3.5, 1.4, 0.2],
                    &[4.9, 3.0, 1.4, 0.2],
                    &[4.7, 3.2, 1.3, 0.2],
                ]))
            )
        }

        #[test]
        fn non_existant_input_file() {
            let potential_error =
                DenseMatrix::<f64>::from_csv("/invalid/path", csv::CSVDefinition::default());
            // The exact message is operating system dependant, therefore, I only test that the correct type
            // error was returned.
            assert_eq!(
                potential_error.clone(),
                Err(ReadingError::CouldNotReadFileSystem {
                    msg: String::from(potential_error.err().unwrap().message().unwrap())
                })
            )
        }
    }
}
