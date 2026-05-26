use std::fmt;
use std::fmt::{Debug, Display};
use std::ops::Range;
use std::slice::Iter;

use approx::{AbsDiffEq, RelativeEq};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::{
    Array, Array2, ArrayView1, ArrayView2, MutArray, MutArrayView2,
};
use crate::linalg::traits::cholesky::CholeskyDecomposable;
use crate::linalg::traits::evd::EVDDecomposable;
use crate::linalg::traits::lu::LUDecomposable;
use crate::linalg::traits::qr::QRDecomposable;
use crate::linalg::traits::stats::{MatrixPreprocessing, MatrixStats};
use crate::linalg::traits::svd::SVDDecomposable;
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;

use crate::error::Failed;

/// Dense matrix
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DenseMatrix<T> {
    ncols: usize,
    nrows: usize,
    values: Vec<T>,
    column_major: bool,
}

/// View on dense matrix
#[derive(Debug, Clone)]
pub struct DenseMatrixView<'a, T: Debug + Display + Copy + Sized> {
    values: &'a [T],
    stride: usize,
    nrows: usize,
    ncols: usize,
    column_major: bool,
}

/// Mutable view on dense matrix
#[derive(Debug)]
pub struct DenseMatrixMutView<'a, T: Debug + Display + Copy + Sized> {
    values: &'a mut [T],
    stride: usize,
    nrows: usize,
    ncols: usize,
    column_major: bool,
}

impl<'a, T: Debug + Display + Copy + Sized> DenseMatrixView<'a, T> {
    fn new(
        m: &'a DenseMatrix<T>,
        vrows: Range<usize>,
        vcols: Range<usize>,
    ) -> Result<Self, Failed> {
        if !m.is_valid_view(m.shape().0, m.shape().1, &vrows, &vcols) {
            Err(Failed::input(
                "The specified view is outside of the matrix range",
            ))
        } else {
            let (start, end, stride) =
                m.stride_range(m.shape().0, m.shape().1, &vrows, &vcols, m.column_major);

            Ok(DenseMatrixView {
                values: &m.values[start..end],
                stride,
                nrows: vrows.end - vrows.start,
                ncols: vcols.end - vcols.start,
                column_major: m.column_major,
            })
        }
    }

    fn iter<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );

        let contiguous_row_major = !self.column_major && axis == 0 && self.stride == self.ncols;
        let contiguous_col_major = self.column_major && axis == 1 && self.stride == self.nrows;
        if contiguous_row_major || contiguous_col_major {
            return Box::new(self.values.iter());
        }
        match axis {
            0 => Box::new(
                (0..self.nrows).flat_map(move |r| (0..self.ncols).map(move |c| self.get((r, c)))),
            ),
            _ => Box::new(
                (0..self.ncols).flat_map(move |c| (0..self.nrows).map(move |r| self.get((r, c)))),
            ),
        }
    }
}

impl<T: Debug + Display + Copy + Sized> fmt::Display for DenseMatrixView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DenseMatrix: nrows: {:?}, ncols: {:?}",
            self.nrows, self.ncols
        )?;
        writeln!(f, "column_major: {:?}", self.column_major)?;
        self.display(f)
    }
}

impl<'a, T: Debug + Display + Copy + Sized> DenseMatrixMutView<'a, T> {
    fn new(
        m: &'a mut DenseMatrix<T>,
        vrows: Range<usize>,
        vcols: Range<usize>,
    ) -> Result<Self, Failed> {
        if !m.is_valid_view(m.shape().0, m.shape().1, &vrows, &vcols) {
            Err(Failed::input(
                "The specified view is outside of the matrix range",
            ))
        } else {
            let (start, end, stride) =
                m.stride_range(m.shape().0, m.shape().1, &vrows, &vcols, m.column_major);

            Ok(DenseMatrixMutView {
                values: &mut m.values[start..end],
                stride,
                nrows: vrows.end - vrows.start,
                ncols: vcols.end - vcols.start,
                column_major: m.column_major,
            })
        }
    }

    fn iter<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );

        // Fast path: when the view is contiguous in the requested traversal order,
        // return a plain slice iterator with no per-element dispatch overhead.
        let contiguous_row_major = !self.column_major && axis == 0 && self.stride == self.ncols;
        let contiguous_col_major = self.column_major && axis == 1 && self.stride == self.nrows;
        if contiguous_row_major || contiguous_col_major {
            return Box::new(self.values.iter());
        }

        match axis {
            0 => Box::new(
                (0..self.nrows).flat_map(move |r| (0..self.ncols).map(move |c| self.get((r, c)))),
            ),
            _ => Box::new(
                (0..self.ncols).flat_map(move |c| (0..self.nrows).map(move |r| self.get((r, c)))),
            ),
        }
    }

    fn iter_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        let nrows = self.nrows;
        let ncols = self.ncols;

        if ncols == 0 || nrows == 0 {
            return Box::new(std::iter::empty());
        }

        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );

        let column_major = self.column_major;
        let stride = self.stride;

        // Eagerly validate that each strided chunk is large enough before returning iterator
        match (column_major, axis) {
            (true, 1) => assert!(
                stride >= nrows,
                "iter_mut: chunk size ({}) < take ({}): view layout is inconsistent",
                stride,
                nrows
            ),
            (false, 0) => assert!(
                stride >= ncols,
                "iter_mut: chunk size ({}) < take ({}): view layout is inconsistent",
                stride,
                ncols
            ),
            _ => {}
        }

        make_iter_mut(self.values, column_major, axis, stride, nrows, ncols)
    }
}

/// Shared mutable iterator logic for both `DenseMatrix::iterator_mut` and
/// `DenseMatrixMutView::iter_mut`
fn make_iter_mut<'a, T: Debug + Display + Copy + Sized>(
    slice: &'a mut [T],
    column_major: bool,
    axis: u8,
    stride: usize,
    nrows: usize,
    ncols: usize,
) -> Box<dyn Iterator<Item = &'a mut T> + 'a> {
    match (column_major, axis) {
        // Case B: column-major, col-by-col
        (true, 1) => Box::new(strided_iter_mut(slice, ncols, stride, nrows)),

        // Case A: column-major, row-by-row
        (true, _) => Box::new(TransposedIterMut::new(slice, stride, nrows, ncols)),

        // Case C: row-major, row-by-row
        (false, 0) => Box::new(strided_iter_mut(slice, nrows, stride, ncols)),

        // Case D: row-major, col-by-col
        (false, _) => Box::new(TransposedIterMut::new(slice, stride, ncols, nrows)),
    }
}

/// Returns a lazy iterator over the first `take` elements of each
/// chunk of size `chunk_size` from `slice`, using `chunks_mut` + `flat_map`
/// to maintain lazy evaluation without any up-front allocation
fn strided_iter_mut<T>(
    slice: &mut [T],
    _chunks: usize,
    chunk_size: usize,
    take: usize,
) -> impl Iterator<Item = &mut T> {
    slice.chunks_mut(chunk_size).flat_map(move |chunk| {
        assert!(
            chunk.len() >= take,
            "iter_mut: chunk size ({}) < take ({}): view layout is inconsistent",
            chunk.len(),
            take
        );
        chunk[..take].iter_mut()
    })
}

struct TransposedIterMut<'a, T> {
    chunks: Vec<&'a mut [T]>,
    outer_count: usize,
    outer_idx: usize,
    inner_idx: usize,
}

impl<'a, T> TransposedIterMut<'a, T> {
    fn new(slice: &'a mut [T], stride: usize, outer_count: usize, inner_count: usize) -> Self {
        let mut chunks: Vec<&'a mut [T]> = Vec::with_capacity(inner_count);
        let mut remaining = slice;
        for i in 0..inner_count {
            let chunk_len = if i < inner_count - 1 {
                stride
            } else {
                remaining.len()
            };
            let (head, tail) = remaining.split_at_mut(chunk_len);
            chunks.push(head);
            remaining = tail;
        }
        TransposedIterMut {
            chunks,
            outer_count,
            outer_idx: 0,
            inner_idx: 0,
        }
    }
}

impl<'a, T> Iterator for TransposedIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.outer_idx >= self.outer_count {
                return None;
            }

            if self.inner_idx >= self.chunks.len() {
                self.outer_idx += 1;
                self.inner_idx = 0;
                continue;
            }

            let chunk: &'a mut [T] = std::mem::take(&mut self.chunks[self.inner_idx]);
            match chunk.split_first_mut() {
                Some((first, rest)) => {
                    self.chunks[self.inner_idx] = rest;
                    self.inner_idx += 1;
                    return Some(first);
                }
                None => {
                    // Empty chunk (shouldn't happen with valid inputs), skip it
                    self.inner_idx += 1;
                }
            }
        }
    }
}

impl<T: Debug + Display + Copy + Sized> fmt::Display for DenseMatrixMutView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DenseMatrix: nrows: {:?}, ncols: {:?}",
            self.nrows, self.ncols
        )?;
        writeln!(f, "column_major: {:?}", self.column_major)?;
        self.display(f)
    }
}

impl<T: Debug + Display + Copy + Sized> DenseMatrix<T> {
    /// Create new instance of `DenseMatrix` without copying data.
    /// `values` should be in column-major order.
    pub fn new(
        nrows: usize,
        ncols: usize,
        values: Vec<T>,
        column_major: bool,
    ) -> Result<Self, Failed> {
        let data_len = values.len();
        if nrows * ncols != values.len() {
            Err(Failed::input(&format!(
                "The specified shape: (cols: {ncols}, rows: {nrows}) does not align with data len: {data_len}"
            )))
        } else {
            Ok(DenseMatrix {
                ncols,
                nrows,
                values,
                column_major,
            })
        }
    }

    /// New instance of `DenseMatrix` from 2d array.
    pub fn from_2d_array(values: &[&[T]]) -> Result<Self, Failed> {
        if values.is_empty() || values[0].is_empty() {
            return Err(Failed::input(
                "The 2d vec provided is empty; cannot instantiate the matrix",
            ));
        }

        let nrows = values.len();
        let ncols = values[0].len();

        for (i, row) in values.iter().enumerate() {
            if row.len() != ncols {
                return Err(Failed::input(&format!(
                    "Row {i} has length {} but row 0 has length {ncols}; \
                     jagged arrays are not supported",
                    row.len()
                )));
            }
        }

        let mut m_values = Vec::with_capacity(nrows * ncols);
        for c in 0..ncols {
            for r in values.iter() {
                m_values.push(r[c]);
            }
        }

        DenseMatrix::new(nrows, ncols, m_values, true)
    }

    /// New instance of `DenseMatrix` from 2d vector.
    ///
    /// Returns `Err` if the input is empty **or** if any row has a different
    /// length than the first row (jagged / ragged arrays are not supported).
    #[allow(clippy::ptr_arg)]
    pub fn from_2d_vec(values: &Vec<Vec<T>>) -> Result<Self, Failed> {
        if values.is_empty() || values[0].is_empty() {
            return Err(Failed::input(
                "The 2d vec provided is empty; cannot instantiate the matrix",
            ));
        }

        let nrows = values.len();
        let ncols = values[0].len();

        // Reject jagged arrays: every row must have exactly `ncols` elements.
        for (i, row) in values.iter().enumerate() {
            if row.len() != ncols {
                return Err(Failed::input(&format!(
                    "Row {i} has length {} but row 0 has length {ncols}; \
                     jagged arrays are not supported",
                    row.len()
                )));
            }
        }

        // Build column-major storage: for each column, push that column's
        // element from every row.  Using a temporary row-major buffer and
        // then transposing in bulk is not faster here because the input is
        // already row-slices; we iterate column-by-column to match the
        // column-major layout expected by `DenseMatrix::new(..., true)`.
        let mut m_values = Vec::with_capacity(nrows * ncols);

        for c in 0..ncols {
            for r in values.iter() {
                m_values.push(r[c]);
            }
        }

        DenseMatrix::new(nrows, ncols, m_values, true)
    }

    /// Iterate over values of matrix
    pub fn iter(&self) -> Iter<'_, T> {
        self.values.iter()
    }

    /// Returns the full backing slice of matrix values.
    ///
    /// The layout is determined by `column_major`: column-major if `true`, row-major if `false`.
    /// Use this for zero-overhead bulk access when you know the storage order.
    #[inline]
    pub fn values_slice(&self) -> &[T] {
        &self.values
    }

    /// Returns a slice of the elements in row `row`.
    ///
    /// For row-major storage this is a single contiguous slice of `ncols` elements.
    /// For column-major storage the elements are not contiguous, so `None` is returned.
    #[inline]
    pub fn row_slice(&self, row: usize) -> Option<&[T]> {
        if row >= self.nrows {
            return None;
        }
        if !self.column_major {
            let start = row * self.ncols;
            Some(&self.values[start..start + self.ncols])
        } else {
            None
        }
    }

    /// Returns a slice of the elements in column `col`.
    ///
    /// For column-major storage this is a single contiguous slice of `nrows` elements.
    /// For row-major storage the elements are not contiguous, so `None` is returned.
    #[inline]
    pub fn col_slice(&self, col: usize) -> Option<&[T]> {
        if col >= self.ncols {
            return None;
        }
        if self.column_major {
            let start = col * self.nrows;
            Some(&self.values[start..start + self.nrows])
        } else {
            None
        }
    }

    /// Check if the size of the requested view is bounded to matrix rows/cols count.
    fn is_valid_view(
        &self,
        n_rows: usize,
        n_cols: usize,
        vrows: &Range<usize>,
        vcols: &Range<usize>,
    ) -> bool {
        vrows.start <= vrows.end
            && vcols.start <= vcols.end
            && vrows.end <= n_rows
            && vcols.end <= n_cols
    }

    /// Compute the range of the requested view: start, end, size of the slice.
    fn stride_range(
        &self,
        n_rows: usize,
        n_cols: usize,
        vrows: &Range<usize>,
        vcols: &Range<usize>,
        column_major: bool,
    ) -> (usize, usize, usize) {
        let (start, end, stride) = if column_major {
            let start = vrows
                .start
                .checked_add(
                    vcols
                        .start
                        .checked_mul(n_rows)
                        .expect("stride_range: integer overflow in start (column_major)"),
                )
                .expect("stride_range: integer overflow in start (column_major)");

            let end = if vcols.is_empty() || vrows.is_empty() {
                start
            } else {
                vrows
                    .end
                    .checked_add(
                        vcols
                            .end
                            .checked_sub(1)
                            .expect("stride_range: vcols.end underflow (column_major)")
                            .checked_mul(n_rows)
                            .expect("stride_range: integer overflow in end (column_major)"),
                    )
                    .expect("stride_range: integer overflow in end (column_major)")
            };
            (start, end, n_rows)
        } else {
            let start = vrows
                .start
                .checked_mul(n_cols)
                .expect("stride_range: integer overflow in start (row_major)")
                .checked_add(vcols.start)
                .expect("stride_range: integer overflow in start (row_major)");

            let end = if vrows.is_empty() || vcols.is_empty() {
                start
            } else {
                vrows
                    .end
                    .checked_sub(1)
                    .expect("stride_range: vrows.end underflow (row_major)")
                    .checked_mul(n_cols)
                    .expect("stride_range: integer overflow in end (row_major)")
                    .checked_add(vcols.end)
                    .expect("stride_range: integer overflow in end (row_major)")
            };
            (start, end, n_cols)
        };
        (start, end, stride)
    }
}

impl<T: Debug + Display + Copy + Sized> fmt::Display for DenseMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "DenseMatrix: nrows: {:?}, ncols: {:?}",
            self.nrows, self.ncols
        )?;
        writeln!(f, "column_major: {:?}", self.column_major)?;
        self.display(f)
    }
}

impl<T: Debug + Display + Copy + Sized + PartialEq> PartialEq for DenseMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            return false;
        }

        let len = self.values.len();
        let other_len = other.values.len();

        if len != other_len {
            return false;
        }

        match self.column_major == other.column_major {
            true => self
                .values
                .iter()
                .zip(other.values.iter())
                .all(|(&v1, v2)| v1.eq(v2)),
            false => self
                .iterator(0)
                .zip(other.iterator(0))
                .all(|(&v1, v2)| v1.eq(v2)),
        }
    }
}

impl<T: Number + RealNumber + AbsDiffEq> AbsDiffEq for DenseMatrix<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            false
        } else {
            self.values
                .iter()
                .zip(other.values.iter())
                .all(|(v1, v2)| T::abs_diff_eq(v1, v2, epsilon))
        }
    }
}

impl<T: Number + RealNumber + RelativeEq> RelativeEq for DenseMatrix<T>
where
    T::Epsilon: Copy,
{
    fn default_max_relative() -> T::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(&self, other: &Self, epsilon: T::Epsilon, max_relative: T::Epsilon) -> bool {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            false
        } else {
            self.iterator(0)
                .zip(other.iterator(0))
                .all(|(v1, v2)| T::relative_eq(v1, v2, epsilon, max_relative))
        }
    }
}

impl<T: Debug + Display + Copy + Sized> Array<T, (usize, usize)> for DenseMatrix<T> {
    fn get(&self, pos: (usize, usize)) -> &T {
        let (row, col) = pos;

        if row >= self.nrows || col >= self.ncols {
            panic!(
                "Invalid index ({},{}) for {}x{} matrix",
                row, col, self.nrows, self.ncols
            );
        }
        if self.column_major {
            &self.values[col * self.nrows + row]
        } else {
            &self.values[col + self.ncols * row]
        }
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn is_empty(&self) -> bool {
        self.ncols < 1 || self.nrows < 1
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );

        // Fast path: a non-view DenseMatrix is always fully contiguous
        // column-major storage is default col-by-col (axis == 1)
        // row-major storage is by default row-by-row (axis == 0)
        // In both matching cases we can return a plain slice iterator
        let natural_order = (self.column_major && axis == 1) || (!self.column_major && axis == 0);
        if natural_order {
            return Box::new(self.values.iter());
        }

        match axis {
            0 => Box::new(
                (0..self.nrows).flat_map(move |r| (0..self.ncols).map(move |c| self.get((r, c)))),
            ),
            _ => Box::new(
                (0..self.ncols).flat_map(move |c| (0..self.nrows).map(move |r| self.get((r, c)))),
            ),
        }
    }
}

impl<T: Debug + Display + Copy + Sized> MutArray<T, (usize, usize)> for DenseMatrix<T> {
    fn set(&mut self, pos: (usize, usize), x: T) {
        if self.column_major {
            self.values[pos.1 * self.nrows + pos.0] = x;
        } else {
            self.values[pos.1 + pos.0 * self.ncols] = x;
        }
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        let (nrows, ncols) = self.shape();

        if ncols == 0 || nrows == 0 {
            return Box::new(std::iter::empty());
        }

        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );

        let column_major = self.column_major;

        let natural_order = (column_major && axis == 1) || (!column_major && axis == 0);
        if natural_order {
            return Box::new(self.values.iter_mut());
        }

        let stride = if column_major { nrows } else { ncols };

        make_iter_mut(&mut self.values, column_major, axis, stride, nrows, ncols)
    }
}

impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for DenseMatrix<T> {}

impl<T: Debug + Display + Copy + Sized> MutArrayView2<T> for DenseMatrix<T> {}

impl<T: Debug + Display + Copy + Sized> Array2<T> for DenseMatrix<T> {
    fn get_row<'a>(&'a self, row: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(DenseMatrixView::new(self, row..row + 1, 0..self.ncols).unwrap())
    }

    fn get_col<'a>(&'a self, col: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(DenseMatrixView::new(self, 0..self.nrows, col..col + 1).unwrap())
    }

    fn slice<'a>(&'a self, rows: Range<usize>, cols: Range<usize>) -> Box<dyn ArrayView2<T> + 'a> {
        Box::new(DenseMatrixView::new(self, rows, cols).unwrap())
    }

    fn slice_mut<'a>(
        &'a mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Box<dyn MutArrayView2<T> + 'a>
    where
        Self: Sized,
    {
        Box::new(DenseMatrixMutView::new(self, rows, cols).unwrap())
    }

    fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        DenseMatrix::new(nrows, ncols, vec![value; nrows * ncols], true).unwrap()
    }

    fn from_iterator<I: Iterator<Item = T>>(iter: I, nrows: usize, ncols: usize, axis: u8) -> Self {
        DenseMatrix::new(nrows, ncols, iter.collect(), axis != 0).unwrap()
    }

    fn transpose(&self) -> Self {
        let mut m = self.clone();
        m.ncols = self.nrows;
        m.nrows = self.ncols;
        m.column_major = !self.column_major;
        m
    }
}

impl<T: Number + RealNumber> QRDecomposable<T> for DenseMatrix<T> {}
impl<T: Number + RealNumber> CholeskyDecomposable<T> for DenseMatrix<T> {}
impl<T: Number + RealNumber> EVDDecomposable<T> for DenseMatrix<T> {}
impl<T: Number + RealNumber> LUDecomposable<T> for DenseMatrix<T> {}
impl<T: Number + RealNumber> SVDDecomposable<T> for DenseMatrix<T> {}

impl<T: Debug + Display + Copy + Sized> Array<T, (usize, usize)> for DenseMatrixView<'_, T> {
    fn get(&self, pos: (usize, usize)) -> &T {
        if self.column_major {
            &self.values[pos.0 + pos.1 * self.stride]
        } else {
            &self.values[pos.0 * self.stride + pos.1]
        }
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn is_empty(&self) -> bool {
        self.nrows == 0 || self.ncols == 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        self.iter(axis)
    }
}

impl<T: Debug + Display + Copy + Sized> Array<T, usize> for DenseMatrixView<'_, T> {
    fn get(&self, i: usize) -> &T {
        if self.nrows == 1 {
            if self.column_major {
                &self.values[i * self.stride]
            } else {
                &self.values[i]
            }
        } else if self.ncols == 1 || (!self.column_major && self.nrows == 1) {
            if self.column_major {
                &self.values[i]
            } else {
                &self.values[i * self.stride]
            }
        } else {
            panic!("This is neither a column nor a row");
        }
    }

    fn shape(&self) -> usize {
        if self.nrows == 1 {
            self.ncols
        } else if self.ncols == 1 {
            self.nrows
        } else {
            panic!("This is neither a column nor a row");
        }
    }

    fn is_empty(&self) -> bool {
        self.nrows == 0 || self.ncols == 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        self.iter(axis)
    }
}

impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for DenseMatrixView<'_, T> {}

impl<T: Debug + Display + Copy + Sized> ArrayView1<T> for DenseMatrixView<'_, T> {}

impl<T: Debug + Display + Copy + Sized> Array<T, (usize, usize)> for DenseMatrixMutView<'_, T> {
    fn get(&self, pos: (usize, usize)) -> &T {
        if self.column_major {
            &self.values[pos.0 + pos.1 * self.stride]
        } else {
            &self.values[pos.0 * self.stride + pos.1]
        }
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn is_empty(&self) -> bool {
        self.nrows == 0 || self.ncols == 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        self.iter(axis)
    }
}

impl<T: Debug + Display + Copy + Sized> MutArray<T, (usize, usize)> for DenseMatrixMutView<'_, T> {
    fn set(&mut self, pos: (usize, usize), x: T) {
        if self.column_major {
            self.values[pos.0 + pos.1 * self.stride] = x;
        } else {
            self.values[pos.0 * self.stride + pos.1] = x;
        }
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        self.iter_mut(axis)
    }
}

impl<T: Debug + Display + Copy + Sized> MutArrayView2<T> for DenseMatrixMutView<'_, T> {}

impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for DenseMatrixMutView<'_, T> {}

impl<T: RealNumber> MatrixStats<T> for DenseMatrix<T> {}

impl<T: RealNumber> MatrixPreprocessing<T> for DenseMatrix<T> {}

#[cfg(test)]
#[warn(clippy::reversed_empty_ranges)]
mod tests {
    use super::*;
    use approx::relative_eq;

    #[test]
    fn test_instantiate_from_2d() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]);
        assert!(x.is_ok());
    }
    #[test]
    fn test_instantiate_from_2d_empty() {
        let input: &[&[f64]] = &[&[]];
        let x = DenseMatrix::from_2d_array(input);
        assert!(x.is_err());
    }
    #[test]
    fn test_instantiate_from_2d_empty2() {
        let input: &[&[f64]] = &[&[], &[]];
        let x = DenseMatrix::from_2d_array(input);
        assert!(x.is_err());
    }

    #[test]
    fn test_from_2d_vec_jagged_returns_err() {
        let jagged = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0, 7.0, 8.0]];
        let result = DenseMatrix::from_2d_vec(&jagged);
        assert!(
            result.is_err(),
            "from_2d_vec should return Err for jagged arrays"
        );
        let msg = format!("{:?}", result.unwrap_err());
        assert!(
            msg.contains("jagged"),
            "error message should mention 'jagged': {msg}"
        );
    }

    #[test]
    fn test_from_2d_vec_uniform_ok() {
        let uniform = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = DenseMatrix::from_2d_vec(&uniform);
        assert!(result.is_ok(), "uniform 2d vec should succeed");
    }

    #[test]
    fn test_instantiate_ok_view1() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();
        let v = DenseMatrixView::new(&x, 0..2, 0..2);
        assert!(v.is_ok());
    }
    #[test]
    fn test_instantiate_ok_view2() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();
        let v = DenseMatrixView::new(&x, 0..3, 0..3);
        assert!(v.is_ok());
    }
    #[test]
    fn test_instantiate_ok_view3() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();
        let v = DenseMatrixView::new(&x, 2..3, 0..3);
        assert!(v.is_ok());
    }
    #[test]
    fn test_instantiate_ok_view4() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();
        let v = DenseMatrixView::new(&x, 3..3, 0..3);
        assert!(v.is_ok());
    }
    #[test]
    fn test_instantiate_err_view1() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();
        let v = DenseMatrixView::new(&x, 3..4, 0..3);
        assert!(v.is_err());
    }
    #[test]
    fn test_instantiate_err_view2() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();
        let v = DenseMatrixView::new(&x, 0..3, 3..4);
        assert!(v.is_err());
    }
    #[test]
    fn test_instantiate_err_view3() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();
        #[allow(clippy::reversed_empty_ranges)]
        let v = DenseMatrixView::new(&x, 0..3, 4..3);
        assert!(v.is_err());
    }

    #[test]
    fn test_is_empty_view_not_empty() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.]]).unwrap();
        let v = DenseMatrixView::new(&x, 0..2, 0..2).unwrap();
        // DenseMatrixView implements Array<T,(usize,usize)> AND Array<T,usize>.
        // Both impls expose is_empty, so we must use fully-qualified syntax to
        // select the 2-D shape variant and avoid E0283.
        assert!(
            !<DenseMatrixView<'_, f64> as Array<f64, (usize, usize)>>::is_empty(&v),
            "2x2 view should not be empty"
        );
    }

    #[test]
    fn test_is_empty_mut_view_not_empty() {
        let mut x = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.]]).unwrap();
        let v = DenseMatrixMutView::new(&mut x, 0..2, 0..2).unwrap();
        assert!(
            !<DenseMatrixMutView<'_, f64> as Array<f64, (usize, usize)>>::is_empty(&v),
            "2x2 mut view should not be empty"
        );
    }

    #[test]
    fn test_display() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();

        println!("{}", &x);
    }

    #[test]
    fn test_get_row_col() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();

        assert_eq!(15.0, x.get_col(1).sum());
        assert_eq!(15.0, x.get_row(1).sum());
        assert_eq!(81.0, x.get_col(1).dot(&(*x.get_row(1))));
    }

    #[test]
    fn test_row_major() {
        let mut x = DenseMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6], false).unwrap();

        assert_eq!(5, *x.get_col(1).get(1));
        assert_eq!(7, x.get_col(1).sum());
        assert_eq!(5, *x.get_row(1).get(1));
        assert_eq!(15, x.get_row(1).sum());
        x.slice_mut(0..2, 1..2)
            .iterator_mut(0)
            .for_each(|v| *v += 2);
        assert_eq!(vec![1, 4, 3, 4, 7, 6], *x.values);
    }

    #[test]
    fn test_get_slice() {
        let x = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9], &[10, 11, 12]])
            .unwrap();

        assert_eq!(
            vec![4, 5, 6],
            DenseMatrix::from_slice(&(*x.slice(1..2, 0..3))).values
        );
        let second_row: Vec<i32> = x.slice(1..2, 0..3).iterator(0).copied().collect();
        assert_eq!(vec![4, 5, 6], second_row);
        let second_col: Vec<i32> = x.slice(0..3, 1..2).iterator(0).copied().collect();
        assert_eq!(vec![2, 5, 8], second_col);
    }

    #[test]
    fn test_iter_mut() {
        let mut x = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]).unwrap();

        assert_eq!(vec![1, 4, 7, 2, 5, 8, 3, 6, 9], x.values);
        // add +2 to some elements
        x.slice_mut(1..2, 0..3)
            .iterator_mut(0)
            .for_each(|v| *v += 2);
        assert_eq!(vec![1, 6, 7, 2, 7, 8, 3, 8, 9], x.values);
        // add +1 to some others
        x.slice_mut(0..3, 1..2)
            .iterator_mut(0)
            .for_each(|v| *v += 1);
        assert_eq!(vec![1, 6, 7, 3, 8, 9, 3, 8, 9], x.values);

        // rewrite matrix as indices of values per axis 1 (row-wise)
        x.iterator_mut(1).enumerate().for_each(|(a, b)| *b = a);
        assert_eq!(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], x.values);
        // rewrite matrix as indices of values per axis 0 (column-wise)
        x.iterator_mut(0).enumerate().for_each(|(a, b)| *b = a);
        assert_eq!(vec![0, 3, 6, 1, 4, 7, 2, 5, 8], x.values);
        // rewrite some by slice
        x.slice_mut(0..3, 0..2)
            .iterator_mut(0)
            .enumerate()
            .for_each(|(a, b)| *b = a);
        assert_eq!(vec![0, 2, 4, 1, 3, 5, 2, 5, 8], x.values);
        x.slice_mut(0..2, 0..3)
            .iterator_mut(1)
            .enumerate()
            .for_each(|(a, b)| *b = a);
        assert_eq!(vec![0, 1, 4, 2, 3, 5, 4, 5, 8], x.values);
    }

    #[test]
    fn test_str_array() {
        let mut x =
            DenseMatrix::from_2d_array(&[&["1", "2", "3"], &["4", "5", "6"], &["7", "8", "9"]])
                .unwrap();

        assert_eq!(vec!["1", "4", "7", "2", "5", "8", "3", "6", "9"], x.values);
        x.iterator_mut(0).for_each(|v| *v = "str");
        assert_eq!(
            vec!["str", "str", "str", "str", "str", "str", "str", "str", "str"],
            x.values
        );
    }

    #[test]
    fn test_transpose() {
        let x = DenseMatrix::<&str>::from_2d_array(&[&["1", "2", "3"], &["4", "5", "6"]]).unwrap();

        assert_eq!(vec!["1", "4", "2", "5", "3", "6"], x.values);
        assert!(x.column_major);

        let x = x.transpose();
        assert_eq!(vec!["1", "4", "2", "5", "3", "6"], x.values);
        assert!(!x.column_major);
    }

    #[test]
    fn test_from_iterator() {
        let data = [1, 2, 3, 4, 5, 6];

        let m = DenseMatrix::from_iterator(data.iter(), 2, 3, 0);

        assert_eq!(
            vec![1, 2, 3, 4, 5, 6],
            m.values.iter().map(|e| **e).collect::<Vec<i32>>()
        );
        assert!(!m.column_major);
    }

    #[test]
    fn test_take() {
        let a = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[1, 2], &[3, 4], &[5, 6]]).unwrap();

        println!("{a}");
        assert_eq!(vec![1, 3, 4, 6], a.take(&[0, 2], 1).values);
        println!("{b}");
        assert_eq!(vec![1, 2, 5, 6], b.take(&[0, 2], 0).values);
    }

    #[test]
    fn test_mut() {
        let a = DenseMatrix::from_2d_array(&[&[1.3, -2.1, 3.4], &[-4., -5.3, 6.1]]).unwrap();

        let a = a.abs();
        assert_eq!(vec![1.3, 4.0, 2.1, 5.3, 3.4, 6.1], a.values);

        let a = a.neg();
        assert_eq!(vec![-1.3, -4.0, -2.1, -5.3, -3.4, -6.1], a.values);
    }

    #[test]
    fn test_reshape() {
        let a = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9], &[10, 11, 12]])
            .unwrap();

        let a = a.reshape(2, 6, 0);
        assert_eq!(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], a.values);
        assert!(a.ncols == 6 && a.nrows == 2 && !a.column_major);

        let a = a.reshape(3, 4, 1);
        assert_eq!(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], a.values);
        assert!(a.ncols == 4 && a.nrows == 3 && a.column_major);
    }

    #[test]
    fn test_eq() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]).unwrap();
        let c = DenseMatrix::from_2d_array(&[
            &[1. + f32::EPSILON, 2., 3.],
            &[4., 5., 6. + f32::EPSILON],
        ])
        .unwrap();
        let d = DenseMatrix::from_2d_array(&[&[1. + 0.5, 2., 3.], &[4., 5., 6. + f32::EPSILON]])
            .unwrap();

        assert!(!relative_eq!(a, b));
        assert!(!relative_eq!(a, d));
        assert!(relative_eq!(a, c));

        let a_int = DenseMatrix::from_2d_array(&[&[1, 2], &[3, 4]]).unwrap();
        let b_int = DenseMatrix::from_2d_array(&[&[1, 2], &[3, 4]]).unwrap();
        let c_int = DenseMatrix::from_2d_array(&[&[5, 6], &[7, 8]]).unwrap();
        assert_eq!(a_int, b_int);
        assert_ne!(a_int, c_int);
    }

    #[test]
    fn test_abs_diff_eq() {
        let a = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0000001]]).unwrap();
        let c = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.1]]).unwrap();

        assert!(a.abs_diff_eq(&b, 1e-6));
        assert!(!a.abs_diff_eq(&c, 1e-6));
    }

    #[test]
    fn test_relative_eq() {
        let a = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0000001]]).unwrap();

        assert!(relative_eq!(a, b, epsilon = 1e-6, max_relative = 1e-6));
    }

    #[test]
    fn test_new_error() {
        let result = DenseMatrix::new(2, 2, vec![1, 2, 3], true);
        assert!(result.is_err());
    }

    #[test]
    fn test_mut_array_iterator_mut_all_cases() {
        // Case B: column-major, axis 1 (col-by-col)
        let mut m1 = DenseMatrix::new(2, 2, vec![1, 2, 3, 4], true).unwrap();
        m1.iterator_mut(1).for_each(|v| *v += 1);
        assert_eq!(m1.values, vec![2, 3, 4, 5]);

        // Case A: column-major, axis 0 (row-by-row)
        let mut m2 = DenseMatrix::new(2, 2, vec![1, 2, 3, 4], true).unwrap();
        let vals: Vec<i32> = m2.iterator_mut(0).map(|v| *v).collect();
        assert_eq!(vals, vec![1, 3, 2, 4]);
        m2.iterator_mut(0).for_each(|v| *v *= 2);
        assert_eq!(m2.values, vec![2, 4, 6, 8]);

        // Case C: row-major, axis 0 (row-by-row)
        let mut m3 = DenseMatrix::new(2, 2, vec![1, 2, 3, 4], false).unwrap();
        m3.iterator_mut(0).for_each(|v| *v += 1);
        assert_eq!(m3.values, vec![2, 3, 4, 5]);

        // Case D: row-major, axis 1 (col-by-col)
        let mut m4 = DenseMatrix::new(2, 2, vec![1, 2, 3, 4], false).unwrap();
        let vals: Vec<i32> = m4.iterator_mut(1).map(|v| *v).collect();
        assert_eq!(vals, vec![1, 3, 2, 4]);
        m4.iterator_mut(1).for_each(|v| *v *= 2);
        assert_eq!(m4.values, vec![2, 4, 6, 8]);
    }

    #[test]
    fn test_iter_empty_matrix() {
        let m00: DenseMatrix<f64> = DenseMatrix::new(0, 0, vec![], true).unwrap();
        assert_eq!(m00.iterator(0).count(), 0);
        assert_eq!(m00.iterator(1).count(), 0);

        let m05: DenseMatrix<f64> = DenseMatrix::new(0, 5, vec![], true).unwrap();
        assert_eq!(m05.iterator(0).count(), 0);
        assert_eq!(m05.iterator(1).count(), 0);

        let m50: DenseMatrix<f64> = DenseMatrix::new(5, 0, vec![], true).unwrap();
        assert_eq!(m50.iterator(0).count(), 0);
        assert_eq!(m50.iterator(1).count(), 0);
    }

    #[test]
    fn test_iterator_mut_empty_matrix() {
        let mut m00: DenseMatrix<f64> = DenseMatrix::new(0, 0, vec![], true).unwrap();
        assert_eq!(m00.iterator_mut(0).count(), 0);
        assert_eq!(m00.iterator_mut(1).count(), 0);

        let mut m05: DenseMatrix<f64> = DenseMatrix::new(0, 5, vec![], true).unwrap();
        assert_eq!(m05.iterator_mut(0).count(), 0);
        assert_eq!(m05.iterator_mut(1).count(), 0);

        let mut m50: DenseMatrix<f64> = DenseMatrix::new(5, 0, vec![], true).unwrap();
        assert_eq!(m50.iterator_mut(0).count(), 0);
        assert_eq!(m50.iterator_mut(1).count(), 0);
    }

    #[test]
    fn test_iter_mut_view_empty_matrix() {
        let mut m: DenseMatrix<f64> = DenseMatrix::fill(5, 5, 0.0);
        // Create an empty view
        let mut v = DenseMatrixMutView::new(&mut m, 0..0, 0..5).unwrap();
        assert_eq!(v.iter_mut(0).count(), 0);
        assert_eq!(v.iter_mut(1).count(), 0);

        let mut v2 = DenseMatrixMutView::new(&mut m, 0..5, 0..0).unwrap();
        assert_eq!(v2.iter_mut(0).count(), 0);
        assert_eq!(v2.iter_mut(1).count(), 0);
    }

    #[test]
    fn test_iter_single_row_column() {
        let m13 = DenseMatrix::from_2d_array(&[&[1.0, 2.0, 3.0]]).unwrap();
        assert_eq!(
            m13.iterator(0).cloned().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            m13.iterator(1).cloned().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );

        let m31 = DenseMatrix::from_2d_array(&[&[1.0], &[2.0], &[3.0]]).unwrap();
        assert_eq!(
            m31.iterator(0).cloned().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            m31.iterator(1).cloned().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    #[should_panic(expected = "iter_mut: chunk size (2) < take (3)")]
    fn test_iter_mut_stride_validation() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let mut view = DenseMatrixMutView {
            values: &mut values,
            stride: 2,
            nrows: 3, // take 3 from chunk of size 2 - should panic
            ncols: 2, // at least 2 columns to trigger chunking
            column_major: true,
        };
        let _ = view.iter_mut(1);
    }

    #[test]
    fn test_dense_matrix_mut_view_iter_mut_all_cases() {
        // Case B: column-major, axis 1 (col-by-col)
        let mut m1 = DenseMatrix::new(3, 3, (1..10).collect(), true).unwrap();
        {
            let mut v = DenseMatrixMutView::new(&mut m1, 0..2, 0..2).unwrap();
            v.iter_mut(1).for_each(|v| *v = 0);
        }
        assert_eq!(m1.values, vec![0, 0, 3, 0, 0, 6, 7, 8, 9]);

        // Case A: column-major, axis 0 (row-by-row)
        let mut m2 = DenseMatrix::new(3, 3, (1..10).collect(), true).unwrap();
        {
            let mut v = DenseMatrixMutView::new(&mut m2, 0..2, 0..2).unwrap();
            let vals: Vec<i32> = v.iter_mut(0).map(|v| *v).collect();
            assert_eq!(vals, vec![1, 4, 2, 5]);
            v.iter_mut(0).for_each(|v| *v = 0);
        }
        assert_eq!(m2.values, vec![0, 0, 3, 0, 0, 6, 7, 8, 9]);

        // Case C: row-major, axis 0 (row-by-row)
        let mut m3 = DenseMatrix::new(3, 3, (1..10).collect(), false).unwrap();
        {
            let mut v = DenseMatrixMutView::new(&mut m3, 0..2, 0..2).unwrap();
            v.iter_mut(0).for_each(|v| *v = 0);
        }
        assert_eq!(m3.values, vec![0, 0, 3, 0, 0, 6, 7, 8, 9]);

        // Case D: row-major, axis 1 (col-by-col)
        let mut m4 = DenseMatrix::new(3, 3, (1..10).collect(), false).unwrap();
        {
            let mut v = DenseMatrixMutView::new(&mut m4, 0..2, 0..2).unwrap();
            let vals: Vec<i32> = v.iter_mut(1).map(|v| *v).collect();
            assert_eq!(vals, vec![1, 4, 2, 5]);
            v.iter_mut(1).for_each(|v| *v = 0);
        }
        assert_eq!(m4.values, vec![0, 0, 3, 0, 0, 6, 7, 8, 9]);
    }

    #[test]
    fn test_is_empty() {
        let m = DenseMatrix::new(2, 2, vec![1, 2, 3, 4], true).unwrap();
        assert!(!m.is_empty());
        let empty: DenseMatrix<i32> = DenseMatrix::new(0, 0, vec![], true).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_stride_range_error() {
        let _m = DenseMatrix::new(2, 2, vec![1, 2, 3, 4], true).unwrap();
    }

    #[test]
    #[should_panic(expected = "Invalid index (2,0) for 2x2 matrix")]
    fn test_get_out_of_bounds() {
        let m = DenseMatrix::from_2d_array(&[&[1, 2], &[3, 4]]).unwrap();
        m.get((2, 0));
    }

    #[test]
    fn test_transpose_row_major() {
        let m = DenseMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6], false).unwrap();
        let mt = m.transpose();
        assert!(mt.column_major);
        assert_eq!(mt.nrows, 3);
        assert_eq!(mt.ncols, 2);
        assert_eq!(mt.values, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[should_panic(expected = "For two dimensional array `axis` should be either 0 or 1")]
    fn test_iterator_invalid_axis() {
        let m = DenseMatrix::from_2d_array(&[&[1, 2], &[3, 4]]).unwrap();
        let _ = m.iterator(2);
    }

    #[test]
    #[should_panic(expected = "For two dimensional array `axis` should be either 0 or 1")]
    fn test_iterator_mut_invalid_axis() {
        let mut m = DenseMatrix::from_2d_array(&[&[1, 2], &[3, 4]]).unwrap();
        let _ = m.iterator_mut(2);
    }

    #[test]
    fn test_view_1d_access() {
        let m = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        let v_row = DenseMatrixView::new(&m, 0..1, 0..3).unwrap();
        assert_eq!(
            <DenseMatrixView<'_, i32> as Array<i32, usize>>::shape(&v_row),
            3
        );
        assert_eq!(
            <DenseMatrixView<'_, i32> as Array<i32, usize>>::get(&v_row, 1),
            &2
        );

        let v_col = DenseMatrixView::new(&m, 0..2, 1..2).unwrap();
        assert_eq!(
            <DenseMatrixView<'_, i32> as Array<i32, usize>>::shape(&v_col),
            2
        );
        assert_eq!(
            <DenseMatrixView<'_, i32> as Array<i32, usize>>::get(&v_col, 1),
            &5
        );
    }

    #[test]
    #[should_panic(expected = "This is neither a column nor a row")]
    fn test_view_1d_access_invalid() {
        let m = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        let v = DenseMatrixView::new(&m, 0..2, 0..2).unwrap();
        let _ = <DenseMatrixView<'_, i32> as Array<i32, usize>>::shape(&v);
    }
}
