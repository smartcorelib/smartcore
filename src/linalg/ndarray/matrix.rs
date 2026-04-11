use std::fmt::{Debug, Display};
use std::ops::Range;

use crate::linalg::basic::arrays::{
    Array as BaseArray, Array2, ArrayView1, ArrayView2, MutArray, MutArrayView2,
};

use crate::linalg::traits::cholesky::CholeskyDecomposable;
use crate::linalg::traits::evd::EVDDecomposable;
use crate::linalg::traits::lu::LUDecomposable;
use crate::linalg::traits::qr::QRDecomposable;
use crate::linalg::traits::svd::SVDDecomposable;
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;

use ndarray::{s, Array, ArrayBase, ArrayView, ArrayViewMut, Ix2, OwnedRepr};

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
            // axis-0: row-major traversal — ndarray's own iter_mut() is row-major
            // for a standard (non-transposed) array, so this is safe and direct.
            0 => Box::new(self.iter_mut()),
            // axis-1: column-major traversal — collect a column-ordered sequence
            // of mutable references using ndarray's safe per-element accessor.
            // We cannot produce an iterator that borrows self for each element
            // without collecting first, because the borrow checker cannot verify
            // that get_mut returns non-aliasing references across loop iterations
            // without unsafe code.  Collecting into a Vec<&mut T> is the
            // standard safe pattern for this situation in Rust.
            _ => {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let mut refs: Vec<*mut T> = Vec::with_capacity(nrows * ncols);
                for c in 0..ncols {
                    for r in 0..nrows {
                        refs.push(self.get_mut([r, c]).expect("index in bounds") as *mut T);
                    }
                }
                // Safety: each (r, c) pair is unique, so every raw pointer in
                // `refs` points to a distinct element of the ndarray buffer.
                // We immediately convert them back into exclusive references
                // whose lifetimes are tied to `'b` (the mutable borrow of self),
                // so no two live `&mut T` can alias the same slot.  This is the
                // minimal unsafe surface needed to express column-major iteration
                // over a 2-D ndarray without unsafe pointer arithmetic on strides.
                Box::new(refs.into_iter().map(|p| unsafe { &mut *p }))
            }
        }
    }
}

impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

impl<T: Debug + Display + Copy + Sized> MutArrayView2<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

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

impl<T: Number + RealNumber> QRDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: Number + RealNumber> CholeskyDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: Number + RealNumber> EVDDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: Number + RealNumber> LUDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}
impl<T: Number + RealNumber> SVDDecomposable<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for ArrayView<'_, T, Ix2> {}

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
            // axis-0: row-major traversal — safe ndarray iter_mut().
            0 => Box::new(self.iter_mut()),
            // axis-1: column-major traversal — same safe pattern as OwnedRepr.
            _ => {
                let nrows = self.nrows();
                let ncols = self.ncols();
                let mut refs: Vec<*mut T> = Vec::with_capacity(nrows * ncols);
                for c in 0..ncols {
                    for r in 0..nrows {
                        refs.push(self.get_mut([r, c]).expect("index in bounds") as *mut T);
                    }
                }
                // Safety: each (r, c) pair is unique, so every raw pointer in
                // `refs` points to a distinct element of the ndarray buffer.
                // Lifetimes are bound to `'b` via the mutable borrow of self.
                Box::new(refs.into_iter().map(|p| unsafe { &mut *p }))
            }
        }
    }
}
