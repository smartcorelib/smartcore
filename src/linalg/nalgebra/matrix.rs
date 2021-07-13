use std::fmt::{Debug, Display};
use std::ops::Range;

use crate::linalg::base::{Array, Array2, ArrayView1, ArrayView2, MutArray, MutArrayView2};
use crate::linalg::cholesky_n::CholeskyDecomposable;
use crate::linalg::evd_n::EVDDecomposable;
use crate::linalg::lu_n::LUDecomposable;
use crate::linalg::qr_n::QRDecomposable;
use crate::linalg::svd_n::SVDDecomposable;
use crate::num::FloatNumber;

use nalgebra::{Dim, Dynamic, Matrix, MatrixSlice, MatrixSliceMut, Scalar, VecStorage};

impl<T: Debug + Display + Copy + Sized + Scalar> Array<T, (usize, usize)>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
    fn get(&self, pos: (usize, usize)) -> &T {
        &self[pos]
    }

    fn shape(&self) -> (usize, usize) {
        self.shape()
    }

    fn is_empty(&self) -> bool {
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        match axis {
            0 => Box::new(
                (0..self.nrows()).flat_map(move |r| (0..self.ncols()).map(move |c| &self[(r, c)])),
            ),
            _ => Box::new(self.iter()),
        }
    }
}

impl<T: Debug + Display + Copy + Sized + Scalar> MutArray<T, (usize, usize)>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
    fn set(&mut self, i: (usize, usize), x: T) {
        self[i] = x
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        let ptr = self.as_mut_ptr();
        let (rstride, cstride) = self.strides();
        match axis {
            0 => Box::new((0..self.nrows()).flat_map(move |r| {
                (0..self.ncols()).map(move |c| unsafe { &mut *ptr.add(r * rstride + c * cstride) })
            })),
            _ => Box::new(self.iter_mut()),
        }
    }
}

impl<T: Debug + Display + Copy + Sized + Scalar> ArrayView2<T>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
}

impl<T: Debug + Display + Copy + Sized + Scalar> MutArrayView2<T>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim>
    Array<T, (usize, usize)> for MatrixSlice<'a, T, Dynamic, Dynamic, RStride, CStride>
{
    fn get(&self, pos: (usize, usize)) -> &T {
        &self[pos]
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn is_empty(&self) -> bool {
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        match axis {
            0 => Box::new(
                (0..self.nrows()).flat_map(move |r| (0..self.ncols()).map(move |c| &self[(r, c)])),
            ),
            _ => Box::new(self.iter()),
        }
    }
}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> ArrayView2<T>
    for MatrixSlice<'a, T, Dynamic, Dynamic, RStride, CStride>
{
}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim>
    Array<T, (usize, usize)> for MatrixSliceMut<'a, T, Dynamic, Dynamic, RStride, CStride>
{
    fn get(&self, pos: (usize, usize)) -> &T {
        &self[pos]
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn is_empty(&self) -> bool {
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        match axis {
            0 => Box::new(self.iter()),
            _ => Box::new(
                (0..self.ncols()).flat_map(move |c| (0..self.nrows()).map(move |r| &self[(r, c)])),
            ),
        }
    }
}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim>
    MutArray<T, (usize, usize)> for MatrixSliceMut<'a, T, Dynamic, Dynamic, RStride, CStride>
{
    fn set(&mut self, pos: (usize, usize), x: T) {
        self[(pos.0, pos.1)] = x
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        let ptr = self.as_mut_ptr();
        let (rstride, cstride) = self.strides();
        match axis {
            0 => Box::new((0..self.nrows()).flat_map(move |r| {
                (0..self.ncols()).map(move |c| unsafe { &mut *ptr.add(r * rstride + c * cstride) })
            })),
            _ => Box::new(self.iter_mut()),
        }
    }
}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> MutArrayView2<T>
    for MatrixSliceMut<'a, T, Dynamic, Dynamic, RStride, CStride>
{
}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> ArrayView2<T>
    for MatrixSliceMut<'a, T, Dynamic, Dynamic, RStride, CStride>
{
}

impl<T: Debug + Display + Copy + Sized + Scalar> Array2<T>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
    fn get_row<'a>(&'a self, row: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(self.row_part(row, self.ncols()))
    }

    fn get_col<'a>(&'a self, col: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(self.column_part(col, self.nrows()))
    }

    fn slice<'a>(&'a self, rows: Range<usize>, cols: Range<usize>) -> Box<dyn ArrayView2<T> + 'a> {
        Box::new(self.slice_range(rows, cols))
    }

    fn slice_mut<'a>(
        &'a mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Box<dyn MutArrayView2<T> + 'a>
    where
        Self: Sized,
    {
        Box::new(self.slice_range_mut(rows, cols))
    }

    fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        Self::from_element(nrows, ncols, value)
    }

    fn from_iterator<I: Iterator<Item = T>>(iter: I, nrows: usize, ncols: usize, axis: u8) -> Self {
        match axis {
            0 => unsafe {
                let mut m =
                    Self::new_uninitialized_generic(Dynamic::new(nrows), Dynamic::new(ncols));
                (0..nrows)
                    .flat_map(move |r| (0..ncols).map(move |c| (r, c)))
                    .zip(iter)
                    .for_each(|((r, c), v)| m[(r, c)] = v);
                m
            },
            _ => Self::from_iterator(nrows, ncols, iter.take(nrows * ncols)),
        }
    }

    fn transpose(&self) -> Self {
        self.transpose()
    }
}

impl<T: FloatNumber + Scalar> QRDecomposable<T>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
}
impl<T: FloatNumber + Scalar> CholeskyDecomposable<T>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
}
impl<T: FloatNumber + Scalar> EVDDecomposable<T>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
}
impl<T: FloatNumber + Scalar> LUDecomposable<T>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
}
impl<T: FloatNumber + Scalar> SVDDecomposable<T>
    for Matrix<T, Dynamic, Dynamic, VecStorage<T, Dynamic, Dynamic>>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, Matrix2x3, RowDVector};

    #[test]
    fn test_get_set() {
        let mut a = DMatrix::from_row_slice(2, 3, &[1, 2, 3, 4, 5, 6]);

        assert_eq!(*Array::get(&a, (1, 1)), 5);
        a.set((1, 1), 9);
        assert_eq!(a, DMatrix::from_row_slice(2, 3, &[1, 2, 3, 4, 9, 6]));
    }

    #[test]
    fn test_my_iterator() {
        let a = DMatrix::from_row_slice(2, 3, &[1, 2, 3, 4, 5, 6]);

        let v: Vec<i32> = a.iterator(0).map(|&v| v).collect();
        assert_eq!(v, vec!(1, 2, 3, 4, 5, 6));
    }

    #[test]
    fn test_mut_iterator() {
        let mut a = DMatrix::from_row_slice(2, 3, &[1, 2, 3, 4, 5, 6]);

        a.iterator_mut(0).enumerate().for_each(|(i, v)| *v = i);
        assert_eq!(a, DMatrix::from_row_slice(2, 3, &[0, 1, 2, 3, 4, 5]));
        a.iterator_mut(1).enumerate().for_each(|(i, v)| *v = i);
        assert_eq!(a, DMatrix::from_row_slice(2, 3, &[0, 2, 4, 1, 3, 5]));
    }

    #[test]
    fn test_slice() {
        let x = DMatrix::from_row_slice(2, 3, &[1, 2, 3, 4, 5, 6]);
        let x_slice = Array2::slice(&x, 0..2, 1..2);
        assert_eq!((2, 1), x_slice.shape());
        assert_eq!(
            x_slice.iterator(0).map(|&v| v).collect::<Vec<i32>>(),
            [2, 5]
        );
    }

    #[test]
    fn test_slice_iter() {
        let x = DMatrix::from_row_slice(2, 3, &[1, 2, 3, 4, 5, 6]);
        let x_slice = Array2::slice(&x, 0..2, 0..3);
        assert_eq!(
            x_slice.iterator(0).map(|&v| v).collect::<Vec<i32>>(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(
            x_slice.iterator(1).map(|&v| v).collect::<Vec<i32>>(),
            vec![1, 4, 2, 5, 3, 6]
        );
    }

    #[test]
    fn test_slice_mut_iter() {
        let mut x = DMatrix::from_row_slice(2, 3, &[1, 2, 3, 4, 5, 6]);
        {
            let mut x_slice = Array2::slice_mut(&mut x, 0..2, 0..3);
            x_slice
                .iterator_mut(0)
                .enumerate()
                .for_each(|(i, v)| *v = i);
        }
        assert_eq!(x, DMatrix::from_row_slice(2, 3, &[0, 1, 2, 3, 4, 5]));
        {
            let mut x_slice = Array2::slice_mut(&mut x, 0..2, 0..3);
            x_slice
                .iterator_mut(1)
                .enumerate()
                .for_each(|(i, v)| *v = i);
        }
        assert_eq!(x, DMatrix::from_row_slice(2, 3, &[0, 2, 4, 1, 3, 5]));
    }

    #[test]
    fn test_c_from_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let a: DMatrix<i32> = Array2::from_iterator(data.clone().into_iter(), 4, 3, 0);
        println!("{}", a);
        let a: DMatrix<i32> = Array2::from_iterator(data.into_iter(), 4, 3, 1);
        println!("{}", a);
    }

    #[test]
    fn test_matmul() {
        let a = DMatrix::from_row_slice(2, 3, &[1, 2, 3, 4, 5, 6]);
        let b = DMatrix::from_row_slice(3, 2, &[1, 2, 3, 4, 5, 6]);
        println!("{}", a.matmul(&b));
    }
}
