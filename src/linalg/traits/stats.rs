//! # Various Statistical Methods
//!
//! This module provides reference implementations for  various statistical functions.
//! Concrete implementations of the `BaseMatrix` trait are free to override these methods for better performance.

//! This methods shall be used when dealing with `DenseMatrix`. Use the ones in `linalg::arrays` for `Array` types.

use crate::linalg::basic::arrays::{Array2, ArrayView2, MutArrayView2};
use crate::numbers::realnum::RealNumber;

/// Defines baseline implementations for various statistical functions
pub trait MatrixStats<T: RealNumber>: ArrayView2<T> + Array2<T> {
    /// Computes the arithmetic mean along the specified axis.
    fn mean(&self, axis: u8) -> Vec<T> {
        let (n, _m) = match axis {
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
        let (n, _m) = match axis {
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
        let mut x = Self::var(self, axis);

        let n = match axis {
            0 => self.shape().1,
            _ => self.shape().0,
        };

        for x_i in x.iter_mut().take(n) {
            *x_i = x_i.sqrt();
        }

        x
    }

    /// <http://en.wikipedia.org/wiki/Arithmetic_mean>
    /// Taken from `statistical`
    /// The MIT License (MIT)
    /// Copyright (c) 2015 Jeff Belgum
    fn _mean_of_vector(v: &[T]) -> T {
        let len = num::cast(v.len()).unwrap();
        v.iter().fold(T::zero(), |acc: T, elem| acc + *elem) / len
    }

    /// Taken from statistical
    /// The MIT License (MIT)
    /// Copyright (c) 2015 Jeff Belgum
    fn _sum_square_deviations_vec(v: &[T], c: Option<T>) -> T {
        let c = match c {
            Some(c) => c,
            None => Self::_mean_of_vector(v),
        };

        let sum = v
            .iter()
            .map(|x| (*x - c) * (*x - c))
            .fold(T::zero(), |acc, elem| acc + elem);
        assert!(sum >= T::zero(), "negative sum of square root deviations");
        sum
    }

    /// <http://en.wikipedia.org/wiki/Variance#Sample_variance>
    /// Taken from statistical
    /// The MIT License (MIT)
    /// Copyright (c) 2015 Jeff Belgum
    fn _var_of_vec(v: &[T], xbar: Option<T>) -> T {
        assert!(v.len() > 1, "variance requires at least two data points");
        let len: T = num::cast(v.len()).unwrap();
        let sum = Self::_sum_square_deviations_vec(v, xbar);
        sum / len
    }

    /// standardize values by removing the mean and scaling to unit variance
    fn standard_scale_mut(&mut self, mean: &[T], std: &[T], axis: u8) {
        let (n, m) = match axis {
            0 => {
                let (n, m) = self.shape();
                (m, n)
            }
            _ => self.shape(),
        };

        for i in 0..n {
            for j in 0..m {
                match axis {
                    0 => self.set((j, i), (*self.get((j, i)) - mean[i]) / std[i]),
                    _ => self.set((i, j), (*self.get((i, j)) - mean[i]) / std[i]),
                }
            }
        }
    }
}

//TODO: this is processing. Should have its own "processing.rs" module
/// Defines baseline implementations for various matrix processing functions
pub trait MatrixPreprocessing<T: RealNumber>: MutArrayView2<T> + Clone {
    /// Each element of the matrix greater than the threshold becomes 1, while values less than or equal to the threshold become 0
    /// ```rust
    /// use smartcore::linalg::basic::matrix::DenseMatrix;
    /// use smartcore::linalg::traits::stats::MatrixPreprocessing;
    /// let mut a = DenseMatrix::from_2d_array(&[&[0., 2., 3.], &[-5., -6., -7.]]).unwrap();
    /// let expected = DenseMatrix::from_2d_array(&[&[0., 1., 1.],&[0., 0., 0.]]).unwrap();
    /// a.binarize_mut(0.);
    ///
    /// assert_eq!(a, expected);
    /// ```

    fn binarize_mut(&mut self, threshold: T) {
        let (nrows, ncols) = self.shape();
        for row in 0..nrows {
            for col in 0..ncols {
                if *self.get((row, col)) > threshold {
                    self.set((row, col), T::one());
                } else {
                    self.set((row, col), T::zero());
                }
            }
        }
    }
    /// Returns new matrix where elements are binarized according to a given threshold.
    /// ```rust
    /// use smartcore::linalg::basic::matrix::DenseMatrix;
    /// use smartcore::linalg::traits::stats::MatrixPreprocessing;
    /// let a = DenseMatrix::from_2d_array(&[&[0., 2., 3.], &[-5., -6., -7.]]).unwrap();
    /// let expected = DenseMatrix::from_2d_array(&[&[0., 1., 1.],&[0., 0., 0.]]).unwrap();
    ///
    /// assert_eq!(a.binarize(0.), expected);
    /// ```
    fn binarize(self, threshold: T) -> Self
    where
        Self: Sized,
    {
        let mut m = self;
        m.binarize_mut(threshold);
        m
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::basic::arrays::Array1;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::linalg::traits::stats::MatrixStats;

    #[test]
    fn test_mean() {
        let m = DenseMatrix::from_2d_array(&[
            &[1., 2., 3., 1., 2.],
            &[4., 5., 6., 3., 4.],
            &[7., 8., 9., 5., 6.],
        ])
        .unwrap();
        let expected_0 = vec![4., 5., 6., 3., 4.];
        let expected_1 = vec![1.8, 4.4, 7.];

        assert_eq!(m.mean(0), expected_0);
        assert_eq!(m.mean(1), expected_1);
    }

    #[test]
    fn test_var() {
        let m = DenseMatrix::from_2d_array(&[&[1., 2., 3., 4.], &[5., 6., 7., 8.]]).unwrap();
        let expected_0 = vec![4., 4., 4., 4.];
        let expected_1 = vec![1.25, 1.25];

        assert!(m.var(0).approximate_eq(&expected_0, 1e-6));
        assert!(m.var(1).approximate_eq(&expected_1, 1e-6));
        assert_eq!(m.mean(0), vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.mean(1), vec![2.5, 6.5]);
    }

    #[test]
    fn test_var_other() {
        let m = DenseMatrix::from_2d_array(&[
            &[0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25],
            &[0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25],
        ])
        .unwrap();
        let expected_0 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let expected_1 = vec![1.25, 1.25];

        assert!(m.var(0).approximate_eq(&expected_0, std::f64::EPSILON));
        assert!(m.var(1).approximate_eq(&expected_1, std::f64::EPSILON));
        assert_eq!(
            m.mean(0),
            vec![0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25]
        );
        assert_eq!(m.mean(1), vec![1.375, 1.375]);
    }

    #[test]
    fn test_std() {
        let m = DenseMatrix::from_2d_array(&[
            &[1., 2., 3., 1., 2.],
            &[4., 5., 6., 3., 4.],
            &[7., 8., 9., 5., 6.],
        ])
        .unwrap();
        let expected_0 = vec![
            2.449489742783178,
            2.449489742783178,
            2.449489742783178,
            1.632993161855452,
            1.632993161855452,
        ];
        let expected_1 = vec![0.7483314773547883, 1.019803902718557, 1.4142135623730951];

        println!("{:?}", m.var(0));

        assert!(m.std(0).approximate_eq(&expected_0, f64::EPSILON));
        assert!(m.std(1).approximate_eq(&expected_1, f64::EPSILON));
        assert_eq!(m.mean(0), vec![4.0, 5.0, 6.0, 3.0, 4.0]);
        assert_eq!(m.mean(1), vec![1.8, 4.4, 7.0]);
    }

    #[test]
    fn test_scale() {
        let m: DenseMatrix<f64> =
            DenseMatrix::from_2d_array(&[&[1., 2., 3., 4.], &[5., 6., 7., 8.]]).unwrap();

        let expected_0: DenseMatrix<f64> =
            DenseMatrix::from_2d_array(&[&[-1., -1., -1., -1.], &[1., 1., 1., 1.]]).unwrap();
        let expected_1: DenseMatrix<f64> = DenseMatrix::from_2d_array(&[
            &[
                -1.3416407864998738,
                -0.4472135954999579,
                0.4472135954999579,
                1.3416407864998738,
            ],
            &[
                -1.3416407864998738,
                -0.4472135954999579,
                0.4472135954999579,
                1.3416407864998738,
            ],
        ])
        .unwrap();

        assert_eq!(m.mean(0), vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.mean(1), vec![2.5, 6.5]);

        assert_eq!(m.var(0), vec![4., 4., 4., 4.]);
        assert_eq!(m.var(1), vec![1.25, 1.25]);

        assert_eq!(m.std(0), vec![2., 2., 2., 2.]);
        assert_eq!(m.std(1), vec![1.118033988749895, 1.118033988749895]);

        {
            let mut m = m.clone();
            m.standard_scale_mut(&m.mean(0), &m.std(0), 0);
            assert_eq!(&m, &expected_0);
        }

        {
            let mut m = m;
            m.standard_scale_mut(&m.mean(1), &m.std(1), 1);
            assert_eq!(&m, &expected_1);
        }
    }
}
