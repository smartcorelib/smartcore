//! # Connector for ndarray
//!
//! If you want to use [ndarray](https://docs.rs/ndarray) matrices and vectors with SmartCore:
//!
//! ```
//! use ndarray::{arr1, arr2};
//! use smartcore::linear::logistic_regression::*;
//! // Enable ndarray connector
//! use smartcore::linalg::ndarray_bindings::*;
//!
//! // Iris dataset
//! let x = arr2(&[
//!            [5.1, 3.5, 1.4, 0.2],
//!            [4.9, 3.0, 1.4, 0.2],
//!            [4.7, 3.2, 1.3, 0.2],
//!            [4.6, 3.1, 1.5, 0.2],
//!            [5.0, 3.6, 1.4, 0.2],
//!            [5.4, 3.9, 1.7, 0.4],
//!            [4.6, 3.4, 1.4, 0.3],
//!            [5.0, 3.4, 1.5, 0.2],
//!            [4.4, 2.9, 1.4, 0.2],
//!            [4.9, 3.1, 1.5, 0.1],
//!            [7.0, 3.2, 4.7, 1.4],
//!            [6.4, 3.2, 4.5, 1.5],
//!            [6.9, 3.1, 4.9, 1.5],
//!            [5.5, 2.3, 4.0, 1.3],
//!            [6.5, 2.8, 4.6, 1.5],
//!            [5.7, 2.8, 4.5, 1.3],
//!            [6.3, 3.3, 4.7, 1.6],
//!            [4.9, 2.4, 3.3, 1.0],
//!            [6.6, 2.9, 4.6, 1.3],
//!            [5.2, 2.7, 3.9, 1.4],
//!         ]);
//! let y = arr1(&[
//!             0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
//!             1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
//!         ]);
//!
//! let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
//! let y_hat = lr.predict(&x).unwrap();
//! ```
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::DivAssign;
use std::ops::MulAssign;
use std::ops::Range;
use std::ops::SubAssign;

use ndarray::ScalarOperand;
use ndarray::{concatenate, s, Array, ArrayBase, Axis, Ix1, Ix2, OwnedRepr};

use crate::linalg::cholesky::CholeskyDecomposableMatrix;
use crate::linalg::evd::EVDDecomposableMatrix;
use crate::linalg::high_order::HighOrderOperations;
use crate::linalg::lu::LUDecomposableMatrix;
use crate::linalg::qr::QRDecomposableMatrix;
use crate::linalg::stats::{MatrixPreprocessing, MatrixStats};
use crate::linalg::svd::SVDDecomposableMatrix;
use crate::linalg::Matrix;
use crate::linalg::{BaseMatrix, BaseVector};
use crate::math::num::RealNumber;

impl<T: RealNumber + ScalarOperand> BaseVector<T> for ArrayBase<OwnedRepr<T>, Ix1> {
    fn get(&self, i: usize) -> T {
        self[i]
    }
    fn set(&mut self, i: usize, x: T) {
        self[i] = x;
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_vec(&self) -> Vec<T> {
        self.to_owned().to_vec()
    }

    fn zeros(len: usize) -> Self {
        Array::zeros(len)
    }

    fn ones(len: usize) -> Self {
        Array::ones(len)
    }

    fn fill(len: usize, value: T) -> Self {
        Array::from_elem(len, value)
    }

    fn dot(&self, other: &Self) -> T {
        self.dot(other)
    }

    fn norm2(&self) -> T {
        self.iter().map(|x| *x * *x).sum::<T>().sqrt()
    }

    fn norm(&self, p: T) -> T {
        if p.is_infinite() && p.is_sign_positive() {
            self.iter().fold(T::neg_infinity(), |f, &val| {
                let v = val.abs();
                if f > v {
                    f
                } else {
                    v
                }
            })
        } else if p.is_infinite() && p.is_sign_negative() {
            self.iter().fold(T::infinity(), |f, &val| {
                let v = val.abs();
                if f < v {
                    f
                } else {
                    v
                }
            })
        } else {
            let mut norm = T::zero();

            for xi in self.iter() {
                norm += xi.abs().powf(p);
            }

            norm.powf(T::one() / p)
        }
    }

    fn div_element_mut(&mut self, pos: usize, x: T) {
        self[pos] /= x;
    }

    fn mul_element_mut(&mut self, pos: usize, x: T) {
        self[pos] *= x;
    }

    fn add_element_mut(&mut self, pos: usize, x: T) {
        self[pos] += x;
    }

    fn sub_element_mut(&mut self, pos: usize, x: T) {
        self[pos] -= x;
    }

    fn approximate_eq(&self, other: &Self, error: T) -> bool {
        (self - other).iter().all(|v| v.abs() <= error)
    }

    fn add_mut(&mut self, other: &Self) -> &Self {
        *self += other;
        self
    }

    fn sub_mut(&mut self, other: &Self) -> &Self {
        *self -= other;
        self
    }

    fn mul_mut(&mut self, other: &Self) -> &Self {
        *self *= other;
        self
    }

    fn div_mut(&mut self, other: &Self) -> &Self {
        *self /= other;
        self
    }

    fn sum(&self) -> T {
        self.sum()
    }

    fn unique(&self) -> Vec<T> {
        let mut result = self.clone().into_raw_vec();
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result.dedup();
        result
    }

    fn copy_from(&mut self, other: &Self) {
        self.assign(&other);
    }
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum>
    BaseMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
    type RowVector = ArrayBase<OwnedRepr<T>, Ix1>;

    fn from_row_vector(vec: Self::RowVector) -> Self {
        let vec_size = vec.len();
        vec.into_shape((1, vec_size)).unwrap()
    }

    fn to_row_vector(self) -> Self::RowVector {
        let vec_size = self.nrows() * self.ncols();
        self.into_shape(vec_size).unwrap()
    }

    fn get(&self, row: usize, col: usize) -> T {
        self[[row, col]]
    }

    fn get_row_as_vec(&self, row: usize) -> Vec<T> {
        self.row(row).to_vec()
    }

    fn get_row(&self, row: usize) -> Self::RowVector {
        self.row(row).to_owned()
    }

    fn copy_row_as_vec(&self, row: usize, result: &mut Vec<T>) {
        for (r, e) in self.row(row).iter().enumerate() {
            result[r] = *e;
        }
    }

    fn get_col_as_vec(&self, col: usize) -> Vec<T> {
        self.column(col).to_vec()
    }

    fn copy_col_as_vec(&self, col: usize, result: &mut Vec<T>) {
        for (c, e) in self.column(col).iter().enumerate() {
            result[c] = *e;
        }
    }

    fn set(&mut self, row: usize, col: usize, x: T) {
        self[[row, col]] = x;
    }

    fn eye(size: usize) -> Self {
        Array::eye(size)
    }

    fn zeros(nrows: usize, ncols: usize) -> Self {
        Array::zeros((nrows, ncols))
    }

    fn ones(nrows: usize, ncols: usize) -> Self {
        Array::ones((nrows, ncols))
    }

    fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        Array::from_elem((nrows, ncols), value)
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    fn h_stack(&self, other: &Self) -> Self {
        concatenate(Axis(1), &[self.view(), other.view()]).unwrap()
    }

    fn v_stack(&self, other: &Self) -> Self {
        concatenate(Axis(0), &[self.view(), other.view()]).unwrap()
    }

    fn matmul(&self, other: &Self) -> Self {
        self.dot(other)
    }

    fn dot(&self, other: &Self) -> T {
        self.dot(&other.view().reversed_axes())[[0, 0]]
    }

    fn slice(&self, rows: Range<usize>, cols: Range<usize>) -> Self {
        self.slice(s![rows, cols]).to_owned()
    }

    fn approximate_eq(&self, other: &Self, error: T) -> bool {
        (self - other).iter().all(|v| v.abs() <= error)
    }

    fn add_mut(&mut self, other: &Self) -> &Self {
        *self += other;
        self
    }

    fn sub_mut(&mut self, other: &Self) -> &Self {
        *self -= other;
        self
    }

    fn mul_mut(&mut self, other: &Self) -> &Self {
        *self *= other;
        self
    }

    fn div_mut(&mut self, other: &Self) -> &Self {
        *self /= other;
        self
    }

    fn add_scalar_mut(&mut self, scalar: T) -> &Self {
        *self += scalar;
        self
    }

    fn sub_scalar_mut(&mut self, scalar: T) -> &Self {
        *self -= scalar;
        self
    }

    fn mul_scalar_mut(&mut self, scalar: T) -> &Self {
        *self *= scalar;
        self
    }

    fn div_scalar_mut(&mut self, scalar: T) -> &Self {
        *self /= scalar;
        self
    }

    fn transpose(&self) -> Self {
        self.clone().reversed_axes()
    }

    fn rand(nrows: usize, ncols: usize) -> Self {
        let values: Vec<T> = (0..nrows * ncols).map(|_| T::rand()).collect();
        Array::from_shape_vec((nrows, ncols), values).unwrap()
    }

    fn norm2(&self) -> T {
        self.iter().map(|x| *x * *x).sum::<T>().sqrt()
    }

    fn norm(&self, p: T) -> T {
        if p.is_infinite() && p.is_sign_positive() {
            self.iter().fold(T::neg_infinity(), |f, &val| {
                let v = val.abs();
                if f > v {
                    f
                } else {
                    v
                }
            })
        } else if p.is_infinite() && p.is_sign_negative() {
            self.iter().fold(T::infinity(), |f, &val| {
                let v = val.abs();
                if f < v {
                    f
                } else {
                    v
                }
            })
        } else {
            let mut norm = T::zero();

            for xi in self.iter() {
                norm += xi.abs().powf(p);
            }

            norm.powf(T::one() / p)
        }
    }

    fn column_mean(&self) -> Vec<T> {
        self.mean_axis(Axis(0)).unwrap().to_vec()
    }

    fn div_element_mut(&mut self, row: usize, col: usize, x: T) {
        self[[row, col]] /= x;
    }

    fn mul_element_mut(&mut self, row: usize, col: usize, x: T) {
        self[[row, col]] *= x;
    }

    fn add_element_mut(&mut self, row: usize, col: usize, x: T) {
        self[[row, col]] += x;
    }

    fn sub_element_mut(&mut self, row: usize, col: usize, x: T) {
        self[[row, col]] -= x;
    }

    fn negative_mut(&mut self) {
        *self *= -T::one();
    }

    fn reshape(&self, nrows: usize, ncols: usize) -> Self {
        self.clone().into_shape((nrows, ncols)).unwrap()
    }

    fn copy_from(&mut self, other: &Self) {
        self.assign(&other);
    }

    fn abs_mut(&mut self) -> &Self {
        for v in self.iter_mut() {
            *v = v.abs()
        }
        self
    }

    fn sum(&self) -> T {
        self.sum()
    }

    fn max(&self) -> T {
        self.iter().fold(T::neg_infinity(), |a, b| a.max(*b))
    }

    fn min(&self) -> T {
        self.iter().fold(T::infinity(), |a, b| a.min(*b))
    }

    fn max_diff(&self, other: &Self) -> T {
        let mut max_diff = T::zero();
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                max_diff = max_diff.max((self[(r, c)] - other[(r, c)]).abs());
            }
        }
        max_diff
    }

    fn softmax_mut(&mut self) {
        let max = self
            .iter()
            .map(|x| x.abs())
            .fold(T::neg_infinity(), |a, b| a.max(b));
        let mut z = T::zero();
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                let p = (self[(r, c)] - max).exp();
                self.set(r, c, p);
                z += p;
            }
        }
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                self.set(r, c, self[(r, c)] / z);
            }
        }
    }

    fn pow_mut(&mut self, p: T) -> &Self {
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                self.set(r, c, self[(r, c)].powf(p));
            }
        }
        self
    }

    fn argmax(&self) -> Vec<usize> {
        let mut res = vec![0usize; self.nrows()];

        for r in 0..self.nrows() {
            let mut max = T::neg_infinity();
            let mut max_pos = 0usize;
            for c in 0..self.ncols() {
                let v = self[(r, c)];
                if max < v {
                    max = v;
                    max_pos = c;
                }
            }
            res[r] = max_pos;
        }

        res
    }

    fn unique(&self) -> Vec<T> {
        let mut result = self.clone().into_raw_vec();
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result.dedup();
        result
    }

    fn cov(&self) -> Self {
        panic!("Not implemented");
    }
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum>
    SVDDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum>
    EVDDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum>
    QRDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum>
    LUDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum>
    CholeskyDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum>
    MatrixStats<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum>
    MatrixPreprocessing<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum>
    HighOrderOperations<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
}

impl<T: RealNumber + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum> Matrix<T>
    for ArrayBase<OwnedRepr<T>, Ix2>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::random_forest_regressor::*;
    use crate::linear::logistic_regression::*;
    use crate::metrics::mean_absolute_error;
    use ndarray::{arr1, arr2, Array1, Array2};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vec_get_set() {
        let mut result = arr1(&[1., 2., 3.]);
        let expected = arr1(&[1., 5., 3.]);

        result.set(1, 5.);

        assert_eq!(result, expected);
        assert_eq!(5., BaseVector::get(&result, 1));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vec_copy_from() {
        let mut v1 = arr1(&[1., 2., 3.]);
        let mut v2 = arr1(&[4., 5., 6.]);
        v1.copy_from(&v2);
        assert_eq!(v1, v2);
        v2[0] = 10.0;
        assert_ne!(v1, v2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vec_len() {
        let v = arr1(&[1., 2., 3.]);
        assert_eq!(3, v.len());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vec_to_vec() {
        let v = arr1(&[1., 2., 3.]);
        assert_eq!(vec![1., 2., 3.], v.to_vec());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vec_dot() {
        let v1 = arr1(&[1., 2., 3.]);
        let v2 = arr1(&[4., 5., 6.]);
        assert_eq!(32.0, BaseVector::dot(&v1, &v2));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vec_approximate_eq() {
        let a = arr1(&[1., 2., 3.]);
        let noise = arr1(&[1e-5, 2e-5, 3e-5]);
        assert!(a.approximate_eq(&(&noise + &a), 1e-4));
        assert!(!a.approximate_eq(&(&noise + &a), 1e-5));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn from_to_row_vec() {
        let vec = arr1(&[1., 2., 3.]);
        assert_eq!(Array2::from_row_vector(vec.clone()), arr2(&[[1., 2., 3.]]));
        assert_eq!(
            Array2::from_row_vector(vec.clone()).to_row_vector(),
            arr1(&[1., 2., 3.])
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn col_matrix_to_row_vector() {
        let m: Array2<f64> = BaseMatrix::zeros(10, 1);
        assert_eq!(m.to_row_vector().len(), 10)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_mut() {
        let mut a1 = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let a2 = a1.clone();
        let a3 = a1.clone() + a2.clone();
        a1.add_mut(&a2);

        assert_eq!(a1, a3);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sub_mut() {
        let mut a1 = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let a2 = a1.clone();
        let a3 = a1.clone() - a2.clone();
        a1.sub_mut(&a2);

        assert_eq!(a1, a3);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mul_mut() {
        let mut a1 = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let a2 = a1.clone();
        let a3 = a1.clone() * a2.clone();
        a1.mul_mut(&a2);

        assert_eq!(a1, a3);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn div_mut() {
        let mut a1 = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let a2 = a1.clone();
        let a3 = a1.clone() / a2.clone();
        a1.div_mut(&a2);

        assert_eq!(a1, a3);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn div_element_mut() {
        let mut a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        a.div_element_mut(1, 1, 5.);

        assert_eq!(BaseMatrix::get(&a, 1, 1), 1.);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mul_element_mut() {
        let mut a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        a.mul_element_mut(1, 1, 5.);

        assert_eq!(BaseMatrix::get(&a, 1, 1), 25.);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_element_mut() {
        let mut a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        a.add_element_mut(1, 1, 5.);

        assert_eq!(BaseMatrix::get(&a, 1, 1), 10.);
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sub_element_mut() {
        let mut a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        a.sub_element_mut(1, 1, 5.);

        assert_eq!(BaseMatrix::get(&a, 1, 1), 0.);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vstack_hstack() {
        let a1 = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let a2 = arr2(&[[7.], [8.]]);

        let a3 = arr2(&[[9., 10., 11., 12.]]);

        let expected = arr2(&[[1., 2., 3., 7.], [4., 5., 6., 8.], [9., 10., 11., 12.]]);

        let result = a1.h_stack(&a2).v_stack(&a3);

        assert_eq!(result, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn get_set() {
        let mut result = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let expected = arr2(&[[1., 2., 3.], [4., 10., 6.]]);

        result.set(1, 1, 10.);

        assert_eq!(result, expected);
        assert_eq!(10., BaseMatrix::get(&result, 1, 1));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matmul() {
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let b = arr2(&[[1., 2.], [3., 4.], [5., 6.]]);
        let expected = arr2(&[[22., 28.], [49., 64.]]);
        let result = BaseMatrix::matmul(&a, &b);
        assert_eq!(result, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dot() {
        let a = arr2(&[[1., 2., 3.]]);
        let b = arr2(&[[1., 2., 3.]]);
        assert_eq!(14., BaseMatrix::dot(&a, &b));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn slice() {
        let a = arr2(&[
            [1., 2., 3., 1., 2.],
            [4., 5., 6., 3., 4.],
            [7., 8., 9., 5., 6.],
        ]);
        let expected = arr2(&[[2., 3.], [5., 6.]]);
        let result = BaseMatrix::slice(&a, 0..2, 1..3);
        assert_eq!(result, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn scalar_ops() {
        let a = arr2(&[[1., 2., 3.]]);
        assert_eq!(&arr2(&[[2., 3., 4.]]), a.clone().add_scalar_mut(1.));
        assert_eq!(&arr2(&[[0., 1., 2.]]), a.clone().sub_scalar_mut(1.));
        assert_eq!(&arr2(&[[2., 4., 6.]]), a.clone().mul_scalar_mut(2.));
        assert_eq!(&arr2(&[[0.5, 1., 1.5]]), a.clone().div_scalar_mut(2.));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn transpose() {
        let m = arr2(&[[1.0, 3.0], [2.0, 4.0]]);
        let expected = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let m_transposed = m.transpose();
        assert_eq!(m_transposed, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn norm() {
        let v = arr2(&[[3., -2., 6.]]);
        assert_eq!(v.norm(1.), 11.);
        assert_eq!(v.norm(2.), 7.);
        assert_eq!(v.norm(std::f64::INFINITY), 6.);
        assert_eq!(v.norm(std::f64::NEG_INFINITY), 2.);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn negative_mut() {
        let mut v = arr2(&[[3., -2., 6.]]);
        v.negative_mut();
        assert_eq!(v, arr2(&[[-3., 2., -6.]]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape() {
        let m_orig = arr2(&[[1., 2., 3., 4., 5., 6.]]);
        let m_2_by_3 = BaseMatrix::reshape(&m_orig, 2, 3);
        let m_result = BaseMatrix::reshape(&m_2_by_3, 1, 6);
        assert_eq!(BaseMatrix::shape(&m_2_by_3), (2, 3));
        assert_eq!(BaseMatrix::get(&m_2_by_3, 1, 1), 5.);
        assert_eq!(BaseMatrix::get(&m_result, 0, 1), 2.);
        assert_eq!(BaseMatrix::get(&m_result, 0, 3), 4.);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn copy_from() {
        let mut src = arr2(&[[1., 2., 3.]]);
        let dst = Array2::<f64>::zeros((1, 3));
        src.copy_from(&dst);
        assert_eq!(src, dst);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn min_max_sum() {
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(21., a.sum());
        assert_eq!(1., a.min());
        assert_eq!(6., a.max());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn max_diff() {
        let a1 = arr2(&[[1., 2., 3.], [4., -5., 6.]]);
        let a2 = arr2(&[[2., 3., 4.], [1., 0., -12.]]);
        assert_eq!(a1.max_diff(&a2), 18.);
        assert_eq!(a2.max_diff(&a2), 0.);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn softmax_mut() {
        let mut prob: Array2<f64> = arr2(&[[1., 2., 3.]]);
        prob.softmax_mut();
        assert!((BaseMatrix::get(&prob, 0, 0) - 0.09).abs() < 0.01);
        assert!((BaseMatrix::get(&prob, 0, 1) - 0.24).abs() < 0.01);
        assert!((BaseMatrix::get(&prob, 0, 2) - 0.66).abs() < 0.01);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pow_mut() {
        let mut a = arr2(&[[1., 2., 3.]]);
        a.pow_mut(3.);
        assert_eq!(a, arr2(&[[1., 8., 27.]]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn argmax() {
        let a = arr2(&[[1., 2., 3.], [-5., -6., -7.], [0.1, 0.2, 0.1]]);
        let res = a.argmax();
        assert_eq!(res, vec![2, 0, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique() {
        let a = arr2(&[[1., 2., 2.], [-2., -6., -7.], [2., 3., 4.]]);
        let res = a.unique();
        assert_eq!(res.len(), 7);
        assert_eq!(res, vec![-7., -6., -2., 1., 2., 3., 4.]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn get_row_as_vector() {
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let res = a.get_row_as_vec(1);
        assert_eq!(res, vec![4., 5., 6.]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn get_row() {
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(arr1(&[4., 5., 6.]), a.get_row(1));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn get_col_as_vector() {
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let res = a.get_col_as_vec(1);
        assert_eq!(res, vec![2., 5., 8.]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn copy_row_col_as_vec() {
        let m = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let mut v = vec![0f32; 3];

        m.copy_row_as_vec(1, &mut v);
        assert_eq!(v, vec!(4., 5., 6.));
        m.copy_col_as_vec(1, &mut v);
        assert_eq!(v, vec!(2., 5., 8.));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn col_mean() {
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let res = a.column_mean();
        assert_eq!(res, vec![4., 5., 6.]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eye() {
        let a = arr2(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        let res: Array2<f64> = BaseMatrix::eye(3);
        assert_eq!(res, a);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand() {
        let m: Array2<f64> = BaseMatrix::rand(3, 3);
        for c in 0..3 {
            for r in 0..3 {
                assert!(m[[r, c]] != 0f64);
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn approximate_eq() {
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let noise = arr2(&[[1e-5, 2e-5, 3e-5], [4e-5, 5e-5, 6e-5], [7e-5, 8e-5, 9e-5]]);
        assert!(a.approximate_eq(&(&noise + &a), 1e-4));
        assert!(!a.approximate_eq(&(&noise + &a), 1e-5));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn abs_mut() {
        let mut a = arr2(&[[1., -2.], [3., -4.]]);
        let expected = arr2(&[[1., 2.], [3., 4.]]);
        a.abs_mut();
        assert_eq!(a, expected);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lr_fit_predict_iris() {
        let x = arr2(&[
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5.0, 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [7.0, 3.2, 4.7, 1.4],
            [6.4, 3.2, 4.5, 1.5],
            [6.9, 3.1, 4.9, 1.5],
            [5.5, 2.3, 4.0, 1.3],
            [6.5, 2.8, 4.6, 1.5],
            [5.7, 2.8, 4.5, 1.3],
            [6.3, 3.3, 4.7, 1.6],
            [4.9, 2.4, 3.3, 1.0],
            [6.6, 2.9, 4.6, 1.3],
            [5.2, 2.7, 3.9, 1.4],
        ]);
        let y: Array1<f64> = arr1(&[
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ]);

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

        let y_hat = lr.predict(&x).unwrap();

        let error: f64 = y
            .into_iter()
            .zip(y_hat.into_iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(error <= 1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn my_fit_longley_ndarray() {
        let x = arr2(&[
            [234.289, 235.6, 159., 107.608, 1947., 60.323],
            [259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            [258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            [284.599, 335.1, 165., 110.929, 1950., 61.187],
            [328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            [346.999, 193.2, 359.4, 113.27, 1952., 63.639],
            [365.385, 187., 354.7, 115.094, 1953., 64.989],
            [363.112, 357.8, 335., 116.219, 1954., 63.761],
            [397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            [419.18, 282.2, 285.7, 118.734, 1956., 67.857],
            [442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            [444.546, 468.1, 263.7, 121.95, 1958., 66.513],
            [482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            [502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            [518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            [554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);
        let y = arr1(&[
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ]);

        let y_hat = RandomForestRegressor::fit(
            &x,
            &y,
            RandomForestRegressorParameters {
                max_depth: None,
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 1000,
                m: Option::None,
                keep_samples: false,
            },
        )
        .unwrap()
        .predict(&x)
        .unwrap();

        assert!(mean_absolute_error(&y, &y_hat) < 1.0);
    }
}
