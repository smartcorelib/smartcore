//! # Various Statistical Methods
//!
//! This module provides reference implementations for  various statistical functions.
//! Concrete implementations of the `BaseMatrix` trait are free to override these methods for better performance.

use crate::linalg::BaseMatrix;
use crate::math::num::RealNumber;

/// Defines baseline implementations for various statistical functions
pub trait MatrixStats<T: RealNumber>: BaseMatrix<T> {
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

        for i in 0..n {
            for j in 0..m {
                x[i] += match axis {
                    0 => self.get(j, i),
                    _ => self.get(i, j),
                };
            }
            x[i] /= div;
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

        for i in 0..n {
            let mut mu = T::zero();
            let mut sum = T::zero();
            for j in 0..m {
                let a = match axis {
                    0 => self.get(j, i),
                    _ => self.get(i, j),
                };
                mu += a;
                sum += a * a;
            }
            mu /= div;
            x[i] = sum / div - mu * mu;
        }

        x
    }

    /// Computes the standard deviation along the specified axis.
    fn std(&self, axis: u8) -> Vec<T> {

        let mut x = self.var(axis);

        let n = match axis {
            0 => self.shape().1,
            _ => self.shape().0,
        };

        for i in 0..n {            
            x[i] = x[i].sqrt();
        }

        x
    }

    /// standardize values by removing the mean and scaling to unit variance
    fn scale_mut(&mut self, mean: &Vec<T>, std: &Vec<T>, axis: u8) {
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
                    0 => self.set(j, i, (self.get(j, i) - mean[i]) / std[i]),
                    _ => self.set(i, j, (self.get(i, j) - mean[i]) / std[i]),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;
    use crate::linalg::BaseVector;

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

        assert!(m.std(0).approximate_eq(&expected_0, 1e-2));
        assert!(m.std(1).approximate_eq(&expected_1, 1e-2));
    }

    #[test]
    fn var() {
        let m = DenseMatrix::from_2d_array(&[
            &[1., 2., 3., 4.],
            &[5., 6., 7., 8.]
        ]);
        let expected_0 = vec![4., 4., 4., 4.];
        let expected_1 = vec![1.25, 1.25];

        assert!(m.var(0).approximate_eq(&expected_0, std::f64::EPSILON));
        assert!(m.var(1).approximate_eq(&expected_1, std::f64::EPSILON));
    }   

    #[test]
    fn scale() {
        let mut m = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]);
        let expected_0 = DenseMatrix::from_2d_array(&[&[-1., -1., -1.], &[1., 1., 1.]]);
        let expected_1 = DenseMatrix::from_2d_array(&[&[-1.22, 0.0, 1.22], &[-1.22, 0.0, 1.22]]);

        {
            let mut m = m.clone();
            m.scale_mut(&m.mean(0), &m.std(0), 0);
            assert!(m.approximate_eq(&expected_0, std::f32::EPSILON));
        }

        m.scale_mut(&m.mean(1), &m.std(1), 1);
        assert!(m.approximate_eq(&expected_1, 1e-2));
    }
}
