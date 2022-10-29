//! # Various Statistical Methods
//!
//! This module provides reference implementations for  various statistical functions.
//! Concrete implementations of the `BaseMatrix` trait are free to override these methods for better performance.

//! This module is deprecated. There are some ideas that can be ported to `linalg::arrays`

use crate::linalg::basic::arrays::{ArrayView2, Array2};
use crate::numbers::realnum::RealNumber;

/// Defines baseline implementations for various statistical functions
pub trait MatrixStats<T: RealNumber>: ArrayView2<T> + Array2<T> {
    /// Computes the arithmetic mean along the specified axis.
    fn mean(&self, axis: u8) -> Vec<T> {
        let (n, m) = match axis {
            0 => {
                let (n, m) = self.shape();
                (m, n)
            }
            _ => self.shape(),
        };

        let mut x: Vec<T> = vec![T::zero(); n];

        for (i, x_i) in x.iter_mut().enumerate().take(n) {
            let vec = match axis {
                0 => self.get_col(i).iterator(0).copied().collect::<Vec<T>>(),
                _ => self.get_row(i).iterator(0).copied().collect::<Vec<T>>(),
            };
            *x_i = Self::_mean_of_vector(&vec[..]);
        }
        x
    }

    /// Computes variance along the specified axis.
    fn var(&self, axis: u8) -> Vec<T> {
        let (n, m) = match axis {
            0 => {
                let (n, m) = self.shape();
                (m, n)
            }
            _ => self.shape(),
        };

        let mut x: Vec<T> = vec![T::zero(); n];

        for (i, x_i) in x.iter_mut().enumerate().take(n) {
            let vec = match axis {
                0 => self.get_col(i).iterator(0).copied().collect::<Vec<T>>(),
                _ => self.get_row(i).iterator(0).copied().collect::<Vec<T>>(),
            };
            *x_i = Self::_var_of_vec(&vec[..], Option::None);
        }

        x
    }

    /// Computes the standard deviation along the specified axis.
    fn std(&self, axis: u8) -> Vec<T> {
        let mut x = MatrixStats::var(self, axis);

        let n = match axis {
            0 => self.shape().1,
            _ => self.shape().0,
        };

        for x_i in x.iter_mut().take(n) {
            *x_i = x_i.sqrt();
        }

        x
    }

    /// (reference)[http://en.wikipedia.org/wiki/Arithmetic_mean]
    /// Taken from statistical
    /// The MIT License (MIT)
    /// Copyright (c) 2015 Jeff Belgum
    fn _mean_of_vector(v: &[T]) -> T
    {
        let len = num::cast(v.len()).unwrap();
        v.iter().fold(T::zero(), |acc: T, elem| acc + *elem) / len
    }

    /// Taken from statistical
    /// The MIT License (MIT)
    /// Copyright (c) 2015 Jeff Belgum
    fn sum_square_deviations_vec(v: &[T], c: Option<T>) -> T
    {
        let c = match c {
            Some(c) => c,
            None => Self::_mean_of_vector(v),
        };

        let sum = v.iter().map( |x| (*x - c) * (*x - c) ).fold(T::zero(), |acc, elem| acc + elem);
        assert!(sum >= T::zero(), "negative sum of square root deviations");
        sum
    }

    /// (Sample variance)[http://en.wikipedia.org/wiki/Variance#Sample_variance]
    /// Taken from statistical
    /// The MIT License (MIT)
    /// Copyright (c) 2015 Jeff Belgum
    fn _var_of_vec(v: &[T], xbar: Option<T>) -> T
    {
        assert!(v.len() > 1, "variance requires at least two data points");
        let len: T = num::cast(v.len()).unwrap();
        let sum = Self::sum_square_deviations_vec(v, xbar);
        sum / (len - T::one())
    }

    // TODO: this is processing. Should have its own "processing.rs" module
    // /// standardize values by removing the mean and scaling to unit variance
    // fn scale_mut(&mut self, mean: &[T], std: &[T], axis: u8) {
    //     let (n, m) = match axis {
    //         0 => {
    //             let (n, m) = self.shape();
    //             (m, n)
    //         }
    //         _ => self.shape(),
    //     };

    //     for i in 0..n {
    //         for j in 0..m {
    //             match axis {
    //                 0 => self.set((j, i), ((*self.get((j, i)) - mean[i]) / std[i])),
    //                 _ => self.set((i, j), ((*self.get((i, j)) - mean[i]) / std[i])),
    //             }
    //         }
    //     }
    // }
}

// TODO: this is processing. Should have its own "processing.rs" module
// /// Defines baseline implementations for various matrix processing functions
// pub trait MatrixPreprocessing<T: RealNumber >: MutArrayView2<T> {
//     /// Each element of the matrix greater than the threshold becomes 1, while values less than or equal to the threshold become 0
//     /// ```
//     /// use smartcore::linalg::basic::matrix::*;
//     /// use smartcore::linalg::traits::stats::MatrixPreprocessing;
//     /// let mut a = DenseMatrix::from_array(2, 3, &[0., 2., 3., -5., -6., -7.]);
//     /// let expected = DenseMatrix::from_array(2, 3, &[0., 1., 1., 0., 0., 0.]);
//     /// a.binarize_mut(0.);
//     ///
//     /// assert_eq!(a, expected);
//     /// ```

//     fn binarize_mut(&mut self, threshold: T) {
//         let (nrows, ncols) = self.shape();
//         for row in 0..nrows {
//             for col in 0..ncols {
//                 if *self.get((row, col)) > threshold {
//                     self.set((row, col), T::one());
//                 } else {
//                     self.set((row, col), T::zero());
//                 }
//             }
//         }
//     }
//     /// Returns new matrix where elements are binarized according to a given threshold.
//     /// ```
//     /// use smartcore::linalg::basic::matrix::*;
//     /// use smartcore::linalg::traits::stats::MatrixPreprocessing;
//     /// let a = DenseMatrix::from_array(2, 3, &[0., 2., 3., -5., -6., -7.]);
//     /// let expected = DenseMatrix::from_array(2, 3, &[0., 1., 1., 0., 0., 0.]);
//     ///
//     /// assert_eq!(a.binarize(0.), expected);
//     /// ```
//     fn binarize(&self, threshold: T) -> Self {
//         let mut m = self.clone();
//         m.binarize_mut(threshold);
//         m
//     }
// }

#[cfg(test)]
mod tests {
    use crate::linalg::basic::arrays::{ArrayView2, Array1};
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn mean() {
        let m = DenseMatrix::from_2d_array(&[
            &[1., 2., 3., 1., 2.],
            &[4., 5., 6., 3., 4.],
            &[7., 8., 9., 5., 6.],
        ]);
        let expected_0 = vec![4., 5., 6., 3., 4.];
        let expected_1 = vec![1.8, 4.4, 7.];

        assert_eq!(m.mean(0), expected_0);
        assert_eq!(m.mean(1), expected_1);
    }

    #[test]
    fn std() {
        let m = DenseMatrix::from_2d_array(&[
            &[1., 2., 3., 1., 2.],
            &[4., 5., 6., 3., 4.],
            &[7., 8., 9., 5., 6.],
        ]);
        let expected_0 = vec![2.44, 2.44, 2.44, 1.63, 1.63];
        let expected_1 = vec![0.74, 1.01, 1.41];

        // assert!(m.std(0).approximate_eq(&expected_0, 1e-2));
        // assert!(m.std(1).approximate_eq(&expected_1, 1e-2));
        assert_eq!(m.mean(0), expected_0);
        assert_eq!(m.mean(1), expected_1);
    }

    #[test]
    fn var() {
        let m = DenseMatrix::from_2d_array(
            &[
                &[1., 2., 3., 4.],
                &[5., 6., 7., 8.]
            ]);
        let expected_0 = vec![4., 4., 4., 4.];
        let expected_1 = vec![1.25, 1.25];

        assert!(m.var(0).approximate_eq(&expected_0, std::f64::EPSILON));
        assert!(m.var(1).approximate_eq(&expected_1, std::f64::EPSILON));
        assert_eq!(m.mean(0), expected_0);
        assert_eq!(m.mean(1), expected_1);
    }

    // TODO: this is processing operation
    // #[test]
    // fn scale() {
    //     let mut m = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]);
    //     let expected_0 = DenseMatrix::from_2d_array(&[&[-1., -1., -1.], &[1., 1., 1.]]);
    //     let expected_1 = DenseMatrix::from_2d_array(&[&[-1.22, 0.0, 1.22], &[-1.22, 0.0, 1.22]]);

    //     {
    //         let mut m = m.clone();
    //         m.scale_mut(&m.mean(0), &m.std(0), 0);
    //         assert!(m.approximate_eq(&expected_0, std::f32::EPSILON));
    //     }

    //     m.scale_mut(&m.mean(1), &m.std(1), 1);
    //     assert!(m.approximate_eq(&expected_1, 1e-2));
    // }
}
