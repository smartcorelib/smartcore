use std::fmt;
use std::fmt::{Debug, Display};
use std::ops::Range;

use crate::linalg::base::{
    Array as BaseArray, Array1, Array2, ArrayView1, ArrayView2, MutArray, MutArrayView2,
};

use crate::num::FloatNumber;
use crate::linalg::cholesky_n::CholeskyDecomposableMatrix;
use crate::linalg::evd_n::EVDDecomposableMatrix;
use crate::linalg::lu_n::LUDecomposableMatrix;
use crate::linalg::qr_n::QRDecomposableMatrix;
use crate::linalg::svd_n::SVDDecomposableMatrix;

use ndarray::ScalarOperand;
use ndarray::{
    concatenate, s, Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Ix1, Ix2, OwnedRepr,
};

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
        let ptr = self.as_mut_ptr();
        let stride = self.strides();
        let (rstride, cstride) = (stride[0] as usize, stride[1] as usize);
        match axis {
            0 => Box::new(self.iter_mut()),
            _ => Box::new((0..self.ncols()).flat_map(move |c| {
                (0..self.nrows()).map(move |r| unsafe { &mut *ptr.add(r * rstride + c * cstride) })
            })),
        }
    }
}

impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

impl<T: Debug + Display + Copy + Sized> MutArrayView2<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

impl<'a, T: Debug + Display + Copy + Sized> BaseArray<T, (usize, usize)> for ArrayView<'a, T, Ix2> {
    fn get(&self, pos: (usize, usize)) -> &T {
        &self[[pos.0, pos.1]]
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
                (0..self.ncols()).flat_map(move |c| (0..self.nrows()).map(move |r| &self[[r, c]])),
            ),
        }
    }
}

impl<T: Debug + Display + Copy + Sized> Array2<T> for ArrayBase<OwnedRepr<T>, Ix2> {
    fn get_row<'a>(&'a self, row: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(self.row(row))
    }

    fn get_col<'a>(&'a self, col: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(self.column(col))
    }

    fn slice<'a>(&'a self, rows: Range<usize>, cols: Range<usize>) -> Box<dyn ArrayView2<T> + 'a> {
        Box::new(self.slice(s![rows, cols]))
    }

    fn slice_mut<'a>(
        &'a mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Box<dyn MutArrayView2<T> + 'a>
    where
        Self: Sized,
    {
        Box::new(self.slice_mut(s![rows, cols]))
    }

    fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        Array::from_elem([nrows, ncols], value)
    }

    fn from_iterator<I: Iterator<Item = T>>(iter: I, nrows: usize, ncols: usize, axis: u8) -> Self {
        let a = Array::from_iter(iter.take(nrows * ncols))
            .into_shape((nrows, ncols))
            .unwrap();
        match axis {
            0 => a,
            _ => a.reversed_axes().into_shape((nrows, ncols)).unwrap(),
        }
    }

    fn transpose(&self) -> Self {
        self.t().to_owned()
    }
}

impl<T: FloatNumber> QRDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: FloatNumber> CholeskyDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: FloatNumber> EVDDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: FloatNumber> LUDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: FloatNumber> SVDDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

impl<'a, T: Debug + Display + Copy + Sized> ArrayView2<T> for ArrayView<'a, T, Ix2> {}

impl<'a, T: Debug + Display + Copy + Sized> BaseArray<T, (usize, usize)>
    for ArrayViewMut<'a, T, Ix2>
{
    fn get(&self, pos: (usize, usize)) -> &T {
        &self[[pos.0, pos.1]]
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
                (0..self.ncols()).flat_map(move |c| (0..self.nrows()).map(move |r| &self[[r, c]])),
            ),
        }
    }
}

impl<'a, T: Debug + Display + Copy + Sized> MutArray<T, (usize, usize)>
    for ArrayViewMut<'a, T, Ix2>
{
    fn set(&mut self, pos: (usize, usize), x: T) {
        self[[pos.0, pos.1]] = x
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        let ptr = self.as_mut_ptr();
        let stride = self.strides();
        let (rstride, cstride) = (stride[0] as usize, stride[1] as usize);
        match axis {
            0 => Box::new(self.iter_mut()),
            _ => Box::new((0..self.ncols()).flat_map(move |c| {
                (0..self.nrows()).map(move |r| unsafe { &mut *ptr.add(r * rstride + c * cstride) })
            })),
        }
    }
}

impl<'a, T: Debug + Display + Copy + Sized> MutArrayView2<T> for ArrayViewMut<'a, T, Ix2> {}

impl<'a, T: Debug + Display + Copy + Sized> ArrayView2<T> for ArrayViewMut<'a, T, Ix2> {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2 as NDArray2};

    #[test]
    fn test_get_set() {
        let mut a = arr2(&[[1, 2, 3], [4, 5, 6]]);

        assert_eq!(*BaseArray::get(&a, (1, 1)), 5);
        a.set((1, 1), 9);
        assert_eq!(a, arr2(&[[1, 2, 3], [4, 9, 6]]));
    }

    #[test]
    fn test_iterator() {
        let a = arr2(&[[1, 2, 3], [4, 5, 6]]);

        let v: Vec<i32> = a.iterator(0).map(|&v| v).collect();
        assert_eq!(v, vec!(1, 2, 3, 4, 5, 6));
    }

    #[test]
    fn test_mut_iterator() {
        let mut a = arr2(&[[1, 2, 3], [4, 5, 6]]);

        a.iterator_mut(0).enumerate().for_each(|(i, v)| *v = i);
        assert_eq!(a, arr2(&[[0, 1, 2], [3, 4, 5]]));
        a.iterator_mut(1).enumerate().for_each(|(i, v)| *v = i);
        assert_eq!(a, arr2(&[[0, 2, 4], [1, 3, 5]]));
    }

    #[test]
    fn test_slice() {
        let x = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let x_slice = Array2::slice(&x, 0..2, 1..2);
        assert_eq!((2, 1), x_slice.shape());
        let v: Vec<i32> = x_slice.iterator(0).map(|&v| v).collect();
        assert_eq!(v, [2, 5]);
    }

    #[test]
    fn test_slice_iter() {
        let x = arr2(&[[1, 2, 3], [4, 5, 6]]);
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
        let mut x = arr2(&[[1, 2, 3], [4, 5, 6]]);
        {
            let mut x_slice = Array2::slice_mut(&mut x, 0..2, 0..3);
            x_slice
                .iterator_mut(0)
                .enumerate()
                .for_each(|(i, v)| *v = i);
        }
        assert_eq!(x, arr2(&[[0, 1, 2], [3, 4, 5]]));
        {
            let mut x_slice = Array2::slice_mut(&mut x, 0..2, 0..3);
            x_slice
                .iterator_mut(1)
                .enumerate()
                .for_each(|(i, v)| *v = i);
        }
        assert_eq!(x, arr2(&[[0, 2, 4], [1, 3, 5]]));
    }

    #[test]
    fn test_c_from_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let a: NDArray2<i32> = Array2::from_iterator(data.clone().into_iter(), 4, 3, 0);
        println!("{}", a);
        let a: NDArray2<i32> = Array2::from_iterator(data.into_iter(), 4, 3, 1);
        println!("{}", a);
    }
}
