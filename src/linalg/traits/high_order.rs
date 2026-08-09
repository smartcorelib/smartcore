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
    /// let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.]]).unwrap();
    /// let b = DenseMatrix::from_2d_array(&[&[5., 6.], &[7., 8.], &[9., 10.]]).unwrap();
    /// let expected = DenseMatrix::from_2d_array(&[&[71., 80.], &[92., 104.]]).unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn ab_false_false() {
        // a * b
        let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[5., 6.], &[7., 8.]]).unwrap();
        let expected = DenseMatrix::from_2d_array(&[&[19., 22.], &[43., 50.]]).unwrap();
        assert_eq!(a.ab(false, &b, false), expected);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn ab_false_true() {
        // a * b^T
        let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[5., 6.], &[7., 8.]]).unwrap();
        let expected = DenseMatrix::from_2d_array(&[&[17., 23.], &[39., 53.]]).unwrap();
        assert_eq!(a.ab(false, &b, true), expected);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn ab_true_false() {
        // a^T * b
        let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[5., 6.], &[7., 8.]]).unwrap();
        let expected = DenseMatrix::from_2d_array(&[&[26., 30.], &[38., 44.]]).unwrap();
        assert_eq!(a.ab(true, &b, false), expected);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn ab_true_true() {
        // ab(true, true) = A^T · B^T = (B·A)^T (by the identity (AB)^T = B^T A^T)
        let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[5., 6.], &[7., 8.]]).unwrap();
        let expected = DenseMatrix::from_2d_array(&[&[23., 31.], &[34., 46.]]).unwrap();
        assert_eq!(a.ab(true, &b, true), expected);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn ab_nonsquare_true_false() {
        // a^T * b with a, b both 3x2 -> result 2x2 (matches the doc-test example)
        let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[5., 6.], &[7., 8.], &[9., 10.]]).unwrap();
        let expected = DenseMatrix::from_2d_array(&[&[71., 80.], &[92., 104.]]).unwrap();
        assert_eq!(a.ab(true, &b, false), expected);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn ab_nonsquare_false_true() {
        // a * b^T with a, b both 3x2 -> result 3x3
        let a = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[5., 6.], &[7., 8.], &[9., 10.]]).unwrap();
        let expected =
            DenseMatrix::from_2d_array(&[&[17., 23., 29.], &[39., 53., 67.], &[61., 83., 105.]])
                .unwrap();
        assert_eq!(a.ab(false, &b, true), expected);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn ab_matches_direct_matmul_and_transpose() {
        // ab(false,false) must equal a.matmul(b); ab(false,true) must equal a.matmul(&b.transpose())
        let a = DenseMatrix::from_2d_array(&[&[2., 0.], &[1., 3.]]).unwrap();
        let b = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.]]).unwrap();

        assert_eq!(a.ab(false, &b, false), a.matmul(&b));
        assert_eq!(a.ab(false, &b, true), a.matmul(&b.transpose()));
        assert_eq!(a.ab(true, &b, false), a.transpose().matmul(&b));
        assert_eq!(a.ab(true, &b, true), b.matmul(&a).transpose());
    }
}
