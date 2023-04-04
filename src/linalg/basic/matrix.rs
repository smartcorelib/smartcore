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

// functional utility functions used across types
fn is_valid_matrix_window(
    mrows: usize,
    mcols: usize,
    vrows: &Range<usize>,
    vcols: &Range<usize>,
) -> bool {
    debug_assert!(
        vrows.end <= mrows && vcols.end <= mcols,
        "The window end is outside of the matrix range"
    );
    debug_assert!(
        vrows.start <= mrows && vcols.start <= mcols,
        "The window start is outside of the matrix range"
    );
    debug_assert!(
        // depends on a properly formed range
        vrows.start <= vrows.end && vcols.start <= vcols.end,
        "Invalid range: start <= end failed"
    );

    !(vrows.end <= mrows && vcols.end <= mcols && vrows.start <= mrows && vcols.start <= mcols)
}
fn start_end_stride(
    mrows: usize,
    mcols: usize,
    vrows: &Range<usize>,
    vcols: &Range<usize>,
    column_major: bool,
) -> (usize, usize, usize) {
    let (start, end, stride) = if column_major {
        (
            vrows.start + vcols.start * mrows,
            vrows.end + (vcols.end - 1) * mrows,
            mrows,
        )
    } else {
        (
            vrows.start * mcols + vcols.start,
            (vrows.end - 1) * mcols + vcols.end,
            mcols,
        )
    };
    (start, end, stride)
}

impl<'a, T: Debug + Display + Copy + Sized> DenseMatrixView<'a, T> {
    fn new(
        m: &'a DenseMatrix<T>,
        vrows: Range<usize>,
        vcols: Range<usize>,
    ) -> Result<Self, Failed> {
        let (mrows, mcols) = m.shape();

        if is_valid_matrix_window(mrows, mcols, &vrows, &vcols) {
            Err(Failed::input(&format!(
                "The specified window is outside of the matrix range"
            )))
        } else {
            let (start, end, stride) =
                start_end_stride(mrows, mcols, &vrows, &vcols, m.column_major);

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
        let (mrows, mcols) = m.shape();
        if is_valid_matrix_window(mrows, mcols, &vrows, &vcols) {
            Err(Failed::input(&format!(
                "The specified window is outside of the matrix range"
            )))
        } else {
            let (start, end, stride) =
                start_end_stride(mrows, mcols, &vrows, &vcols, m.column_major);

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
        debug_assert!(
            nrows * ncols == values.len(),
            "Instantiatint DenseMatrix requires nrows * ncols == values.len()"
        );
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
        DenseMatrix::from_2d_vec(&values.iter().map(|row| Vec::from(*row)).collect())
    }

    /// New instance of `DenseMatrix` from 2d vector.
    pub fn from_2d_vec(values: &Vec<Vec<T>>) -> Result<Self, Failed> {
        debug_assert!(
            !(values.is_empty() || values[0].is_empty()),
            "Instantiating DenseMatrix requires a non-empty 2d_vec"
        );

        if values.is_empty() || values[0].is_empty() {
            Err(Failed::input(&format!(
                "The 2d vec provided is empty; cannot instantiate the matrix"
            )))
        } else {
            let nrows = values.len();
            let ncols = values
                .first()
                .unwrap_or_else(|| {
                    panic!("Invalid state: Cannot create 2d matrix from an empty vector")
                })
                .len();
            let mut m_values = Vec::with_capacity(nrows * ncols);

            for c in 0..ncols {
                for r in values.iter().take(nrows) {
                    m_values.push(r[c])
                }
            }

            DenseMatrix::new(nrows, ncols, m_values, true)
        }
    }

    /// Iterate over values of matrix
    pub fn iter(&self) -> Iter<'_, T> {
        self.values.iter()
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

    // equality in differences in absolute values, according to an epsilon
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
        let idx_target = col * self.nrows + row;

        println!("------ 🦀 ----- 📋 target: {}", self.values.len());
        println!("------ 🦀 ----- 📋 nrows: {}", &self.nrows);
        println!("row: {} col: {}", &row, &col);
        println!("DenseMatrix get target: {}", idx_target);

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
        self.ncols > 0 && self.nrows > 0
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

    // private function so for now assume infalible
    fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        DenseMatrix::new(nrows, ncols, vec![value; nrows * ncols], true).unwrap()
    }

    // private function so for now assume infalible
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

impl<T: RealNumber> MatrixStats<T> for DenseMatrix<T> {}

impl<T: RealNumber> MatrixPreprocessing<T> for DenseMatrix<T> {}

#[cfg(test)]
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
        let v = DenseMatrixView::new(&x, 0..3, 4..3);
        assert!(v.is_err());
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

        // transpose
        let x = x.transpose();
        assert_eq!(vec!["1", "4", "2", "5", "3", "6"], x.values);
        assert!(!x.column_major); // should change column_major
    }

    #[test]
    fn test_from_iterator() {
        let data = vec![1, 2, 3, 4, 5, 6];

        let m = DenseMatrix::from_iterator(data.iter(), 2, 3, 0);

        // make a vector into a 2x3 matrix.
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
        // take column 0 and 2
        assert_eq!(vec![1, 3, 4, 6], a.take(&[0, 2], 1).values);
        println!("{b}");
        // take rows 0 and 2
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
    }
}
