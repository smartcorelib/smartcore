//! In this module you will find composite of matrix operations that are used elsewhere
//! for improved efficiency.

use crate::linalg::basic::arrays::Array2;
use crate::numbers::basenum::Number;

/// High order matrix operations.
pub trait HighOrderOperations<T: Number>: Array2<T> {
    /// Y = AB
    /// ```
    /// use smartcore::linalg::basic::matrix::*;
    /// use smartcore::linalg::traits::high_order::HighOrderOperations;
    /// use smartcore::linalg::basic::arrays::Array2;
    ///
    /// let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.]]);
    /// let b = DenseMatrix::from_2d_array(&[&[5., 6.], &[7., 8.], &[9., 10.]]);
    /// let expected = DenseMatrix::from_2d_array(&[&[71., 80.], &[92., 104.]]);
    ///
    /// assert_eq!(a.ab(true, &b, false), expected);
    /// ```
    fn ab(&self, a_transpose: bool, b: &Self, b_transpose: bool) -> Self {
        match (a_transpose, b_transpose) {
            (true, true) => b.matmul(self).transpose(),
            (false, true) => self.matmul(&b.transpose()),
            (true, false) => self.transpose().matmul(b),
            (false, false) => self.matmul(b),
        }
    }
}

mod tests {
    /* TODO: Add tests  */
}
