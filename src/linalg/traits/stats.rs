//! # Various Statistical Methods
//!
//! This module provides reference implementations for  various statistical functions.
//! Concrete implementations of the `BaseMatrix` trait are free to override these methods for better performance.

use crate::linalg::basic::arrays::{ArrayView2, MutArrayView2};
use crate::numbers::realnum::RealNumber;


/// Defines baseline implementations for various statistical functions
pub trait MatrixStats<T: RealNumber >: ArrayView2<T> {
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

        let div = T::from_usize(m).unwrap();

        for (i, x_i) in x.iter_mut().enumerate().take(n) {
            for j in 0..m {
                *x_i += match axis {
                    0 => *self.get((j, i)),
                    _ => *self.get((i, j)),
                };
            }
            *x_i /= div;
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

        let div = T::from_usize(m).unwrap();

        for (i, x_i) in x.iter_mut().enumerate().take(n) {
            let mut mu = T::zero();
            let mut sum = T::zero();
            for j in 0..m {
                let a = match axis {
                    0 => self.get((j, i)),
                    _ => self.get((i, j)),
                };
                mu += *a;
                sum += *a * *a;
            }
            mu /= div;
            *x_i = sum / div - mu.powi(2);
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
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::linalg::basic::arrays::ArrayView2;
    
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
    
    // #[test]
    // fn std() {
    //     let m = DenseMatrix::from_2d_array(&[
    //         &[1., 2., 3., 1., 2.],
    //         &[4., 5., 6., 3., 4.],
    //         &[7., 8., 9., 5., 6.],
    //     ]);
    //     let expected_0 = vec![2.44, 2.44, 2.44, 1.63, 1.63];
    //     let expected_1 = vec![0.74, 1.01, 1.41];

    //     // assert!(m.std(0).approximate_eq(&expected_0, 1e-2));
    //     // assert!(m.std(1).approximate_eq(&expected_1, 1e-2));
    //     assert_eq!(m.mean(0), expected_0);
    //     assert_eq!(m.mean(1), expected_1);
    // }
    
    // #[test]
    // fn var() {
    //     let m = DenseMatrix::from_2d_array(&[&[1., 2., 3., 4.], &[5., 6., 7., 8.]]);
    //     let expected_0 = vec![4., 4., 4., 4.];
    //     let expected_1 = vec![1.25, 1.25];

    //     // assert!(m.var(0).approximate_eq(&expected_0, std::f64::EPSILON));
    //     // assert!(m.var(1).approximate_eq(&expected_1, std::f64::EPSILON));
    //     assert_eq!(m.mean(0), expected_0);
    //     assert_eq!(m.mean(1), expected_1);
    // }

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
