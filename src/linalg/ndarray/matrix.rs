use std::fmt::{Debug, Display};
use std::ops::Range;

use crate::linalg::basic::arrays::{
    Array as BaseArray, Array2, ArrayView1, ArrayView2, MutArray, MutArrayView2,
};
use crate::linalg::basic::matrix::DenseMatrix;

use crate::linalg::traits::cholesky::CholeskyDecomposable;
use crate::linalg::traits::evd::EVDDecomposable;
use crate::linalg::traits::lu::LUDecomposable;
use crate::linalg::traits::qr::QRDecomposable;
use crate::linalg::traits::svd::SVDDecomposable;
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;

use ndarray::{s, Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Ix2, Order, OwnedRepr};

// ---------------------------------------------------------------------------
// ArrayBase<OwnedRepr<T>, Ix2>  (owned 2-D array)
// ---------------------------------------------------------------------------

const ROW_MAJOR_AXIS: u8 = 0;

impl<T: Debug + Display + Copy> DenseMatrix<T> {
    /// Copies an owned two-dimensional ndarray into a [`DenseMatrix`].
    ///
    /// The resulting matrix uses row-major (C) storage regardless of the
    /// memory layout of the source array.
    ///
    /// # Notes
    ///
    /// [`ndarray::Array2::iter`] always yields elements in logical row-major
    /// order, independent of whether the source is C- or Fortran-ordered. This
    /// invariant makes transposed-layout conversion correct.
    ///
    /// # Panics
    ///
    /// Panics if `nrows * ncols` overflows `usize`. An empty array (zero
    /// rows or zero columns) does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::Array2;
    /// use smartcore::linalg::basic::arrays::Array;
    /// use smartcore::linalg::basic::matrix::DenseMatrix;
    ///
    /// let array = Array2::from_shape_vec(
    ///     (3, 4),
    ///     (0..12).map(|value| value as f64).collect(),
    /// ).unwrap();
    /// let matrix = DenseMatrix::from_ndarray2(&array);
    /// assert_eq!(matrix.shape(), (3, 4));
    /// assert_eq!(*matrix.get((1, 2)), 6.0);
    /// ```
    pub fn from_ndarray2(a: &ndarray::Array2<T>) -> Self {
        // iter() yields logical row-major order regardless of memory layout.
        Self::from_iterator(a.iter().copied(), a.nrows(), a.ncols(), ROW_MAJOR_AXIS)
    }
}

impl<T: Debug + Display + Copy + Sized> BaseArray<T, (usize, usize)>
    for ArrayBase<OwnedRepr<T>, Ix2>
{
    fn get(&self, pos: (usize, usize)) -> &T {
        &self[[pos.0, pos.1]]
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        match axis {
            0 => Box::new(self.iter()),
            _ => Box::new(
                (0..self.ncols()).flat_map(move |c| (0..self.nrows()).map(move |r| &self[[r, c]])),
            ),
        }
    }
}

impl<T: Debug + Display + Copy + Sized> MutArray<T, (usize, usize)>
    for ArrayBase<OwnedRepr<T>, Ix2>
{
    fn set(&mut self, pos: (usize, usize), x: T) {
        self[[pos.0, pos.1]] = x
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        match axis {
            // axis-0: row-major — ndarray iter_mut() traverses in row-major order.
            0 => Box::new(self.iter_mut()),
            // axis-1: column-major — axis_iter_mut(Axis(1)) yields each column as a
            // non-overlapping ArrayViewMut1<T>; into_iter() gives &mut T.
            // No raw pointers or unsafe blocks required.
            _ => Box::new(self.axis_iter_mut(Axis(1)).flat_map(|col| col.into_iter())),
        }
    }
}

impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: Debug + Display + Copy + Sized> MutArrayView2<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

impl<T: Debug + Display + Copy + Sized> Array2<T> for ArrayBase<OwnedRepr<T>, Ix2> {
    fn get_row<'a>(&'a self, row: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(self.row(row))
    }

    fn get_col<'a>(&'a self, col: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(self.column(col))
    }

    fn slice<'a>(&'a self, rows: Range<usize>, cols: Range<usize>) -> Box<dyn ArrayView2<T> + 'a> {
        Box::new(self.view().slice_move(s![rows, cols]))
    }

    fn slice_mut<'a>(
        &'a mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Box<dyn MutArrayView2<T> + 'a>
    where
        Self: Sized,
    {
        // slice_mut returns ArrayBase<ViewRepr<&mut T>, Ix2> which is ArrayViewMut.
        // We implement MutArrayView2 for ArrayViewMut below, so this cast is valid.
        Box::new(self.view_mut().slice_move(s![rows, cols]))
    }

    fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        Array::from_elem([nrows, ncols], value)
    }

    fn from_iterator<I: Iterator<Item = T>>(iter: I, nrows: usize, ncols: usize, axis: u8) -> Self {
        // `into_shape` was deprecated in ndarray 0.16; use `into_shape_with_order` instead.
        let a = Array::from_iter(iter.take(nrows * ncols))
            .into_shape_with_order(((nrows, ncols), Order::RowMajor))
            .unwrap();
        match axis {
            0 => a,
            _ => a
                .reversed_axes()
                .into_shape_with_order(((nrows, ncols), Order::RowMajor))
                .unwrap(),
        }
    }

    fn transpose(&self) -> Self {
        self.t().to_owned()
    }
}

impl<T: Number + RealNumber> QRDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: Number + RealNumber> CholeskyDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: Number + RealNumber> EVDDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: Number + RealNumber> LUDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: Number + RealNumber> SVDDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

// ---------------------------------------------------------------------------
// ArrayView<'_, T, Ix2>  (immutable 2-D view / slice)
// ---------------------------------------------------------------------------

impl<T: Debug + Display + Copy + Sized> BaseArray<T, (usize, usize)> for ArrayView<'_, T, Ix2> {
    fn get(&self, pos: (usize, usize)) -> &T {
        &self[[pos.0, pos.1]]
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        match axis {
            0 => Box::new(self.iter()),
            _ => Box::new(
                (0..self.ncols()).flat_map(move |c| (0..self.nrows()).map(move |r| &self[[r, c]])),
            ),
        }
    }
}

impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for ArrayView<'_, T, Ix2> {}

// ---------------------------------------------------------------------------
// ArrayViewMut<'_, T, Ix2>  (mutable 2-D view — returned by slice_mut)
// ---------------------------------------------------------------------------

impl<T: Debug + Display + Copy + Sized> BaseArray<T, (usize, usize)> for ArrayViewMut<'_, T, Ix2> {
    fn get(&self, pos: (usize, usize)) -> &T {
        &self[[pos.0, pos.1]]
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        match axis {
            0 => Box::new(self.iter()),
            _ => Box::new(
                (0..self.ncols()).flat_map(move |c| (0..self.nrows()).map(move |r| &self[[r, c]])),
            ),
        }
    }
}

impl<T: Debug + Display + Copy + Sized> MutArray<T, (usize, usize)> for ArrayViewMut<'_, T, Ix2> {
    fn set(&mut self, pos: (usize, usize), x: T) {
        self[[pos.0, pos.1]] = x
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        match axis {
            // axis-0: row-major — safe ndarray iter_mut().
            0 => Box::new(self.iter_mut()),
            // axis-1: column-major — axis_iter_mut(Axis(1)) yields each column as a
            // non-overlapping ArrayViewMut1<T>; into_iter() gives &mut T.
            // No raw pointers or unsafe blocks required.
            _ => Box::new(self.axis_iter_mut(Axis(1)).flat_map(|col| col.into_iter())),
        }
    }
}

// ArrayViewMut satisfies both ArrayView2 (read) and MutArrayView2 (read+write),
// which is exactly what slice_mut's return type Box<dyn MutArrayView2<T>> requires.
impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for ArrayViewMut<'_, T, Ix2> {}
impl<T: Debug + Display + Copy + Sized> MutArrayView2<T> for ArrayViewMut<'_, T, Ix2> {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_dense_matrix_from_ndarray2() {
        let input = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let matrix = DenseMatrix::from_ndarray2(&input);
        let expected = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6]]).unwrap();
        assert_eq!(matrix, expected);

        let transposed = input.reversed_axes();
        let matrix = DenseMatrix::from_ndarray2(&transposed);
        let expected = DenseMatrix::from_2d_array(&[&[1, 4], &[2, 5], &[3, 6]]).unwrap();
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_dense_matrix_from_ndarray2_square() {
        let input = arr2(&[[1, 2], [3, 4]]);
        let matrix = DenseMatrix::from_ndarray2(&input);
        let expected = DenseMatrix::from_2d_array(&[&[1, 2], &[3, 4]]).unwrap();
        assert_eq!(matrix, expected);
    }

    #[test]
    fn test_dense_matrix_from_ndarray2_row_vector() {
        let input = arr2(&[[10, 20, 30, 40]]);
        let matrix = DenseMatrix::from_ndarray2(&input);
        let expected = DenseMatrix::from_2d_array(&[&[10, 20, 30, 40]]).unwrap();
        assert_eq!(matrix, expected);
        assert_eq!(matrix.shape(), (1, 4));
    }

    #[test]
    fn test_dense_matrix_from_ndarray2_col_vector() {
        let input = arr2(&[[10], [20], [30], [40]]);
        let matrix = DenseMatrix::from_ndarray2(&input);
        let expected = DenseMatrix::from_2d_array(&[&[10], &[20], &[30], &[40]]).unwrap();
        assert_eq!(matrix, expected);
        assert_eq!(matrix.shape(), (4, 1));
    }

    #[test]
    fn test_dense_matrix_from_ndarray2_empty() {
        let input = ndarray::Array2::<i32>::zeros((0, 0));
        let matrix = DenseMatrix::from_ndarray2(&input);
        assert!(matrix.is_empty());
        assert_eq!(matrix.shape(), (0, 0));
    }

    #[test]
    fn test_get_row() {
        let m = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let row = Array2::get_row(&m, 1);
        assert_eq!(row.shape(), 3);
        assert_eq!(*row.get(0), 4);
        assert_eq!(*row.get(1), 5);
        assert_eq!(*row.get(2), 6);
    }

    #[test]
    fn test_get_col() {
        let m = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let col = Array2::get_col(&m, 1);
        assert_eq!(col.shape(), 2);
        assert_eq!(*col.get(0), 2);
        assert_eq!(*col.get(1), 5);
    }

    #[test]
    fn test_slice() {
        let m = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let view = Array2::slice(&m, 1..3, 0..2);
        assert_eq!(view.shape(), (2, 2));
        assert_eq!(*view.get((0, 0)), 4);
        assert_eq!(*view.get((0, 1)), 5);
        assert_eq!(*view.get((1, 0)), 7);
        assert_eq!(*view.get((1, 1)), 8);
    }

    #[test]
    fn test_slice_mut() {
        let mut m = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        {
            let mut view = Array2::slice_mut(&mut m, 1..3, 0..2);
            view.set((0, 0), 40);
            view.set((1, 1), 80);
        }
        assert_eq!(m, arr2(&[[1, 2, 3], [40, 5, 6], [7, 80, 9]]));
    }

    #[test]
    fn test_is_empty() {
        let empty = ndarray::Array2::<i32>::from_shape_simple_fn((0, 0), || unreachable!());
        let non_empty = arr2(&[[1, 2], [3, 4]]);
        assert!(BaseArray::is_empty(&empty));
        assert!(!BaseArray::is_empty(&non_empty));
    }
}
