use std::fmt;
use std::fmt::{Debug, Display};
use std::ops::Range;
use std::slice::Iter;

use approx::{AbsDiffEq, RelativeEq};

use crate::linalg::basic::arrays::{
    Array, Array2, ArrayView1, ArrayView2, MutArray, MutArrayView2,
};
use crate::linalg::traits::cholesky::CholeskyDecomposable;
use crate::linalg::traits::evd::EVDDecomposable;
use crate::linalg::traits::lu::LUDecomposable;
use crate::linalg::traits::qr::QRDecomposable;
use crate::linalg::traits::svd::SVDDecomposable;
use crate::numbers::floatnum::FloatNumber;

/// Dense matrix
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
    fn new(m: &'a DenseMatrix<T>, rows: Range<usize>, cols: Range<usize>) -> Self {
        let (start, end, stride) = if m.column_major {
            (
                rows.start + cols.start * m.nrows,
                rows.end + (cols.end - 1) * m.nrows,
                m.nrows,
            )
        } else {
            (
                rows.start * m.ncols + cols.start,
                (rows.end - 1) * m.ncols + cols.end,
                m.ncols,
            )
        };
        let view = DenseMatrixView {
            values: &m.values[start..end],
            stride,
            nrows: rows.end - rows.start,
            ncols: cols.end - cols.start,
            column_major: m.column_major,
        };
        view
    }

    fn iter<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
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

impl<'a, T: Debug + Display + Copy + Sized> fmt::Display for DenseMatrixView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display(f)
    }
}

impl<'a, T: Debug + Display + Copy + Sized> DenseMatrixMutView<'a, T> {
    fn new(m: &'a mut DenseMatrix<T>, rows: Range<usize>, cols: Range<usize>) -> Self {
        let (start, end, stride) = if m.column_major {
            (
                rows.start + cols.start * m.nrows,
                rows.end + (cols.end - 1) * m.nrows,
                m.nrows,
            )
        } else {
            (
                rows.start * m.ncols + cols.start,
                (rows.end - 1) * m.ncols + cols.end,
                m.ncols,
            )
        };
        let view = DenseMatrixMutView {
            values: &mut m.values[start..end],
            stride: stride,
            nrows: rows.end - rows.start,
            ncols: cols.end - cols.start,
            column_major: m.column_major,
        };
        view
    }

    fn iter<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
        match axis {
            0 => Box::new(
                (0..self.nrows).flat_map(move |r| (0..self.ncols).map(move |c| self.get((r, c)))),
            ),
            _ => Box::new(
                (0..self.ncols).flat_map(move |c| (0..self.nrows).map(move |r| self.get((r, c)))),
            ),
        }
    }

    fn iter_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &mut T> + 'b> {
        let column_major = self.column_major;
        let stride = self.stride;
        let ptr = self.values.as_mut_ptr();
        match axis {
            0 => Box::new((0..self.nrows).flat_map(move |r| {
                (0..self.ncols).map(move |c| unsafe {
                    &mut *ptr.add(if column_major {
                        r + c * stride
                    } else {
                        r * stride + c
                    })
                })
            })),
            _ => Box::new((0..self.ncols).flat_map(move |c| {
                (0..self.nrows).map(move |r| unsafe {
                    &mut *ptr.add(if column_major {
                        r + c * stride
                    } else {
                        r * stride + c
                    })
                })
            })),
        }
    }
}

impl<'a, T: Debug + Display + Copy + Sized> fmt::Display for DenseMatrixMutView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.display(f)
    }
}

impl<T: Debug + Display + Copy + Sized> DenseMatrix<T> {
    /// Create new instance of `DenseMatrix` without copying data.
    /// `values` should be in column-major order.
    pub fn new(nrows: usize, ncols: usize, values: Vec<T>, column_major: bool) -> Self {
        DenseMatrix {
            ncols,
            nrows,
            values,
            column_major,
        }
    }

    /// New instance of `DenseMatrix` from 2d array.
    pub fn from_2d_array(values: &[&[T]]) -> Self {
        DenseMatrix::from_2d_vec(&values.iter().map(|row| Vec::from(*row)).collect())
    }

    /// New instance of `DenseMatrix` from 2d vector.
    pub fn from_2d_vec(values: &Vec<Vec<T>>) -> Self {
        let nrows = values.len();
        let ncols = values
            .first()
            .unwrap_or_else(|| panic!("Cannot create 2d matrix from an empty vector"))
            .len();
        let mut m_values = Vec::with_capacity(nrows * ncols);

        for c in 0..ncols {
            for r in 0..nrows {
                m_values.push(values[r][c])
            }
        }

        DenseMatrix::new(nrows, ncols, m_values, true)
    }

    /// Iterate over values of matrix
    pub fn iter(&self) -> Iter<T> {
        self.values.iter()
    }
}

impl<T: Debug + Display + Copy + Sized> fmt::Display for DenseMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl<T: FloatNumber + AbsDiffEq> AbsDiffEq for DenseMatrix<T>
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

impl<T: FloatNumber + RelativeEq> RelativeEq for DenseMatrix<T>
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
        self.ncols > 0 && self.ncols > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(
            axis == 1 || axis == 0,
            "For two dimensional array `axis` should be either 0 or 1"
        );
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
        let ptr = self.values.as_mut_ptr();
        let column_major = self.column_major;
        let (nrows, ncols) = self.shape();
        match axis {
            0 => Box::new((0..self.nrows).flat_map(move |r| {
                (0..self.ncols).map(move |c| unsafe {
                    &mut *ptr.add(if column_major {
                        r + c * nrows
                    } else {
                        r * ncols + c
                    })
                })
            })),
            _ => Box::new((0..self.ncols).flat_map(move |c| {
                (0..self.nrows).map(move |r| unsafe {
                    &mut *ptr.add(if column_major {
                        r + c * nrows
                    } else {
                        r * ncols + c
                    })
                })
            })),
        }
    }
}

impl<T: Debug + Display + Copy + Sized> ArrayView2<T> for DenseMatrix<T> {}

impl<T: Debug + Display + Copy + Sized> MutArrayView2<T> for DenseMatrix<T> {}

impl<T: Debug + Display + Copy + Sized> Array2<T> for DenseMatrix<T> {
    fn get_row<'a>(&'a self, row: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(DenseMatrixView::new(self, row..row + 1, 0..self.ncols))
    }

    fn get_col<'a>(&'a self, col: usize) -> Box<dyn ArrayView1<T> + 'a> {
        Box::new(DenseMatrixView::new(self, 0..self.nrows, col..col + 1))
    }

    fn slice<'a>(&'a self, rows: Range<usize>, cols: Range<usize>) -> Box<dyn ArrayView2<T> + 'a> {
        Box::new(DenseMatrixView::new(self, rows, cols))
    }

    fn slice_mut<'a>(
        &'a mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Box<dyn MutArrayView2<T> + 'a>
    where
        Self: Sized,
    {
        Box::new(DenseMatrixMutView::new(self, rows, cols))
    }

    fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        DenseMatrix::new(nrows, ncols, vec![value; nrows * ncols], true)
    }

    fn from_iterator<I: Iterator<Item = T>>(iter: I, nrows: usize, ncols: usize, axis: u8) -> Self {
        DenseMatrix::new(
            nrows,
            ncols,
            iter.collect(),
            if axis == 0 { false } else { true },
        )
    }

    fn transpose(&self) -> Self {
        let mut m = self.clone();
        m.ncols = self.nrows;
        m.nrows = self.ncols;
        m.column_major = !self.column_major;
        m
    }
}

impl<T: FloatNumber> QRDecomposable<T> for DenseMatrix<T> {}
impl<T: FloatNumber> CholeskyDecomposable<T> for DenseMatrix<T> {}
impl<T: FloatNumber> EVDDecomposable<T> for DenseMatrix<T> {}
impl<T: FloatNumber> LUDecomposable<T> for DenseMatrix<T> {}
impl<T: FloatNumber> SVDDecomposable<T> for DenseMatrix<T> {}

impl<'a, T: Debug + Display + Copy + Sized> Array<T, (usize, usize)> for DenseMatrixView<'a, T> {
    fn get(&self, pos: (usize, usize)) -> &T {
        if self.column_major {
            &self.values[(pos.0 + pos.1 * self.stride)]
        } else {
            &self.values[(pos.0 * self.stride + pos.1)]
        }
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn is_empty(&self) -> bool {
        self.nrows * self.ncols > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        self.iter(axis)
    }
}

impl<'a, T: Debug + Display + Copy + Sized> Array<T, usize> for DenseMatrixView<'a, T> {
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
        self.nrows * self.ncols > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        self.iter(axis)
    }
}

impl<'a, T: Debug + Display + Copy + Sized> ArrayView2<T> for DenseMatrixView<'a, T> {}

impl<'a, T: Debug + Display + Copy + Sized> ArrayView1<T> for DenseMatrixView<'a, T> {}

impl<'a, T: Debug + Display + Copy + Sized> Array<T, (usize, usize)> for DenseMatrixMutView<'a, T> {
    fn get(&self, pos: (usize, usize)) -> &T {
        if self.column_major {
            &self.values[(pos.0 + pos.1 * self.stride)]
        } else {
            &self.values[(pos.0 * self.stride + pos.1)]
        }
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn is_empty(&self) -> bool {
        self.nrows * self.ncols > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        self.iter(axis)
    }
}

impl<'a, T: Debug + Display + Copy + Sized> MutArray<T, (usize, usize)>
    for DenseMatrixMutView<'a, T>
{
    fn set(&mut self, pos: (usize, usize), x: T) {
        if self.column_major {
            self.values[(pos.0 + pos.1 * self.stride)] = x;
        } else {
            self.values[(pos.0 * self.stride + pos.1)] = x;
        }
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        self.iter_mut(axis)
    }
}

impl<'a, T: Debug + Display + Copy + Sized> MutArrayView2<T> for DenseMatrixMutView<'a, T> {}

impl<'a, T: Debug + Display + Copy + Sized> ArrayView2<T> for DenseMatrixMutView<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::relative_eq;

    #[test]
    fn test_get_row_col() {
        let x = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]);

        println!("{:?}", x.get_col(1).sum());
        println!("{:?}", x.get_row(1).sum());

        println!("{:?}", x.get_col(1).dot(&(*x.get_row(1))));
    }

    #[test]
    fn test_row_major() {
        let mut x = DenseMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6], false);

        println!("{:?}", x.get_col(1).get(1));
        println!("{:?}", x.get_col(1).sum());
        println!("{:?}", x.get_row(1).get(1));
        println!("{:?}", x.get_row(1).sum());
        x.slice_mut(0..2, 1..2)
            .iterator_mut(0)
            .for_each(|v| *v += 2);
        println!("{}", x);
    }

    #[test]
    fn test_get_slice() {
        let x = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9], &[10, 11, 12]]);

        println!("{}", DenseMatrix::from_slice(&(*x.slice(1..2, 0..3))));
        let second_row: Vec<i32> = x.slice(1..2, 0..3).iterator(0).map(|x| *x).collect();
        println!("{:?}", second_row);
        let second_col: Vec<i32> = x.slice(0..3, 1..2).iterator(0).map(|x| *x).collect();
        println!("{:?}", second_col);
    }

    #[test]
    fn test_iter_mut() {
        let mut x = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9]]);

        println!("{:?}", x);
        x.slice_mut(1..2, 0..3)
            .iterator_mut(0)
            .for_each(|v| *v += 2);
        x.slice_mut(0..3, 1..2)
            .iterator_mut(0)
            .for_each(|v| *v += 1);
        println!("{:?}", x);
        x.iterator_mut(1).enumerate().for_each(|(a, b)| *b = a);
        println!("{}", x);
        x.iterator_mut(0).enumerate().for_each(|(a, b)| *b = a);
        println!("{}", x);
        x.slice_mut(0..3, 0..2)
            .iterator_mut(0)
            .enumerate()
            .for_each(|(a, b)| *b = a);
        println!("{}", x);
        x.slice_mut(0..2, 0..3)
            .iterator_mut(1)
            .enumerate()
            .for_each(|(a, b)| *b = a);
        println!("{}", x);
    }

    #[test]
    fn test_str_array() {
        let mut x =
            DenseMatrix::from_2d_array(&[&["1", "2", "3"], &["4", "5", "6"], &["7", "8", "9"]]);

        println!("{:?}", x);
        x.iterator_mut(0).for_each(|v| *v = "str");
        println!("{:?}", x);
    }

    #[test]
    fn test_transpose() {
        let x = DenseMatrix::from_2d_array(&[&["1", "2", "3"], &["4", "5", "6"]]);

        println!("{:?}", x);
        println!("{:?}", x.transpose());
    }

    #[test]
    fn test_from_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6];

        println!("{}", DenseMatrix::from_iterator(data.iter(), 2, 3, 0));
    }

    #[test]
    fn test_take() {
        let a = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6]]);
        let b = DenseMatrix::from_2d_array(&[&[1, 2], &[3, 4], &[5, 6]]);

        println!("{}", a);
        println!("{}", a.take(&[0, 2], 1));
        println!("{}", b);
        println!("{}", b.take(&[0, 2], 0));
    }

    #[test]
    fn test_mut() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]);
        a.abs();
        a.neg();

        println!("{}", a);
    }

    #[test]
    fn test_reshape() {
        let a = DenseMatrix::from_2d_array(&[&[1, 2, 3], &[4, 5, 6], &[7, 8, 9], &[10, 11, 12]]);

        println!("{}", a.reshape(2, 6, 0));
        println!("{}", a.reshape(3, 4, 1));
    }

    #[test]
    fn test_eq() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]);
        let b = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]);
        let c = DenseMatrix::from_2d_array(&[
            &[1. + f32::EPSILON, 2., 3.],
            &[4., 5., 6. + f32::EPSILON],
        ]);
        let d = DenseMatrix::from_2d_array(&[&[1. + 0.5, 2., 3.], &[4., 5., 6. + f32::EPSILON]]);

        assert!(!relative_eq!(a, b));
        assert!(!relative_eq!(a, d));
        assert!(relative_eq!(a, c));
    }
}
