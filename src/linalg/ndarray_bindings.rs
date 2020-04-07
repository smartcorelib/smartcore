use std::ops::Range;
use std::iter::Sum;
use std::ops::AddAssign;
use std::ops::SubAssign;
use std::ops::MulAssign;
use std::ops::DivAssign;

use ndarray::{Array, ArrayBase, OwnedRepr, Ix2, Ix1, Axis, stack, s};
use ndarray::ScalarOperand;

use crate::math::num::FloatExt;
use crate::linalg::BaseMatrix;
use crate::linalg::Matrix;
use crate::linalg::svd::SVDDecomposableMatrix;
use crate::linalg::evd::EVDDecomposableMatrix;
use crate::linalg::qr::QRDecomposableMatrix;


impl<T: FloatExt + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum> BaseMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2>
{
    type RowVector = ArrayBase<OwnedRepr<T>, Ix1>;

    fn from_row_vector(vec: Self::RowVector) -> Self{
        let vec_size = vec.len();
        vec.into_shape((1, vec_size)).unwrap()
    }

    fn to_row_vector(self) -> Self::RowVector{
        let vec_size = self.nrows() * self.ncols();
        self.into_shape(vec_size).unwrap()
    }

    fn get(&self, row: usize, col: usize) -> T {
        self[[row, col]]
    }

    fn get_row_as_vec(&self, row: usize) -> Vec<T> {
        self.row(row).to_vec()
    }

    fn get_col_as_vec(&self, col: usize) -> Vec<T> {
        self.column(col).to_vec()
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

    fn to_raw_vector(&self) -> Vec<T> {
        self.to_owned().iter().map(|v| *v).collect()
    }

    fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        Array::from_elem((nrows, ncols), value)
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }    

    fn v_stack(&self, other: &Self) -> Self {
        stack(Axis(1), &[self.view(), other.view()]).unwrap()
    }

    fn h_stack(&self, other: &Self) -> Self {
        stack(Axis(0), &[self.view(), other.view()]).unwrap()
    }

    fn dot(&self, other: &Self) -> Self {
       self.dot(other)
    }    

    fn vector_dot(&self, other: &Self) -> T {
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

    fn div_mut(&mut self, other: &Self) -> &Self{
        *self /= other;
        self
    }

    fn add_scalar_mut(&mut self, scalar: T) -> &Self{
        *self += scalar;
        self
    }

    fn sub_scalar_mut(&mut self, scalar: T) -> &Self{
        *self -= scalar;
        self
    }

    fn mul_scalar_mut(&mut self, scalar: T) -> &Self{
        *self *= scalar;
        self
    }

    fn div_scalar_mut(&mut self, scalar: T) -> &Self{
        *self /= scalar;
        self
    }

    fn transpose(&self) -> Self{
        self.clone().reversed_axes()
    }

    fn rand(nrows: usize, ncols: usize) -> Self{        
        let values: Vec<T> = (0..nrows*ncols).map(|_| {
            T::rand()
        }).collect();
        Array::from_shape_vec((nrows, ncols), values).unwrap()
    }

    fn norm2(&self) -> T{
        self.iter().map(|x| *x * *x).sum::<T>().sqrt()
    }

    fn norm(&self, p:T) -> T {
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
                norm = norm + xi.abs().powf(p);
            }

            norm.powf(T::one()/p)
        }
    }

    fn column_mean(&self) -> Vec<T> {
        self.mean_axis(Axis(0)).unwrap().to_vec()
    }

    fn div_element_mut(&mut self, row: usize, col: usize, x: T){
        self[[row, col]] = self[[row, col]] / x;
    }

    fn mul_element_mut(&mut self, row: usize, col: usize, x: T){
        self[[row, col]] = self[[row, col]] * x;
    }

    fn add_element_mut(&mut self, row: usize, col: usize, x: T){
        self[[row, col]] = self[[row, col]] + x;
    }

    fn sub_element_mut(&mut self, row: usize, col: usize, x: T){
        self[[row, col]] = self[[row, col]] - x;
    }

    fn negative_mut(&mut self){
        *self *= -T::one();
    }

    fn reshape(&self, nrows: usize, ncols: usize) -> Self{
        self.clone().into_shape((nrows, ncols)).unwrap()
    }

    fn copy_from(&mut self, other: &Self){
        self.assign(&other);
    }

    fn abs_mut(&mut self) -> &Self{
        for v in self.iter_mut(){
            *v = v.abs()
        }        
        self
    }

    fn sum(&self) -> T{
        self.sum()
    }

    fn max_diff(&self, other: &Self) -> T{
        let mut max_diff = T::zero();
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                max_diff = max_diff.max((self[(r, c)] - other[(r, c)]).abs());
            }
        }
        max_diff
    }
    
    fn softmax_mut(&mut self){
        let max = self.iter().map(|x| x.abs()).fold(T::neg_infinity(), |a, b| a.max(b));
        let mut z = T::zero();
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                let p = (self[(r, c)] - max).exp();
                self.set(r, c, p);
                z = z + p;
            }
        }
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                self.set(r, c, self[(r, c)] / z);
            }
        }
    }

    fn pow_mut(&mut self, p: T) -> &Self{
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                self.set(r, c, self[(r, c)].powf(p));
            }
        }
        self
    }

    fn argmax(&self) -> Vec<usize>{
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

}

impl<T: FloatExt + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum> SVDDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

impl<T: FloatExt + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum> EVDDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

impl<T: FloatExt + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum> QRDecomposableMatrix<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

impl<T: FloatExt + ScalarOperand + AddAssign + SubAssign + MulAssign + DivAssign + Sum> Matrix<T> for ArrayBase<OwnedRepr<T>, Ix2> {}

#[cfg(test)]
mod tests {    
    use super::*; 
    use ndarray::{arr1, arr2, Array2};

    #[test]
    fn from_to_row_vec() { 

        let vec = arr1(&[ 1.,  2.,  3.]);
        assert_eq!(Array2::from_row_vector(vec.clone()), arr2(&[[1., 2., 3.]]));
        assert_eq!(Array2::from_row_vector(vec.clone()).to_row_vector(), arr1(&[1., 2., 3.]));

    }

    #[test]
    fn add_mut() { 

        let mut a1 = arr2(&[[ 1.,  2.,  3.],
                            [4., 5., 6.]]);
        let a2 = a1.clone();
        let a3 = a1.clone() + a2.clone();
        a1.add_mut(&a2);

        assert_eq!(a1, a3);        

    }

    #[test]
    fn sub_mut() { 

        let mut a1 = arr2(&[[ 1.,  2.,  3.],
                            [4., 5., 6.]]);
        let a2 = a1.clone();
        let a3 = a1.clone() - a2.clone();
        a1.sub_mut(&a2);

        assert_eq!(a1, a3);        

    }

    #[test]
    fn mul_mut() { 

        let mut a1 = arr2(&[[ 1.,  2.,  3.],
                            [4., 5., 6.]]);
        let a2 = a1.clone();
        let a3 = a1.clone() * a2.clone();
        a1.mul_mut(&a2);        

        assert_eq!(a1, a3);        

    }

    #[test]
    fn div_mut() { 

        let mut a1 = arr2(&[[ 1.,  2.,  3.],
                            [4., 5., 6.]]);
        let a2 = a1.clone();
        let a3 = a1.clone() / a2.clone();
        a1.div_mut(&a2);        

        assert_eq!(a1, a3);        

    }

    #[test]
    fn div_element_mut() { 

        let mut a = arr2(&[[ 1.,  2.,  3.],
                            [4., 5., 6.]]);
        a.div_element_mut(1, 1, 5.);

        assert_eq!(BaseMatrix::get(&a, 1, 1), 1.);

    }

    #[test]
    fn mul_element_mut() { 

        let mut a = arr2(&[[ 1.,  2.,  3.],
                            [4., 5., 6.]]);
        a.mul_element_mut(1, 1, 5.);

        assert_eq!(BaseMatrix::get(&a, 1, 1), 25.);

    }

    #[test]
    fn add_element_mut() { 

        let mut a = arr2(&[[ 1.,  2.,  3.],
                            [4., 5., 6.]]);
        a.add_element_mut(1, 1, 5.);

        assert_eq!(BaseMatrix::get(&a, 1, 1), 10.);

    }

    #[test]
    fn sub_element_mut() { 

        let mut a = arr2(&[[ 1.,  2.,  3.],
                            [4., 5., 6.]]);
        a.sub_element_mut(1, 1, 5.);

        assert_eq!(BaseMatrix::get(&a, 1, 1), 0.);

    }

    #[test]
    fn vstack_hstack() { 

        let a1 = arr2(&[[1.,  2.,  3.],
                        [4., 5., 6.]]);
        let a2 = arr2(&[[ 7.], [8.]]);

        let a3 = arr2(&[[9., 10., 11., 12.]]);

        let expected = arr2(&[[1., 2., 3., 7.],
                              [4., 5., 6., 8.],
                              [9., 10., 11., 12.]]);

        let result = a1.v_stack(&a2).h_stack(&a3);        
        
        assert_eq!(result, expected);      

    }

    #[test]
    fn to_raw_vector() {
        let result = arr2(&[[1.,  2.,  3.], [4., 5., 6.]]).to_raw_vector();
        let expected = vec![1., 2., 3., 4., 5., 6.];

        assert_eq!(result, expected);     
    }

    #[test]
    fn get_set() {
        let mut result = arr2(&[[1.,  2.,  3.], [4., 5., 6.]]);
        let expected = arr2(&[[1.,  2.,  3.], [4., 10., 6.]]);

        result.set(1, 1, 10.);

        assert_eq!(result, expected);
        assert_eq!(10., BaseMatrix::get(&result, 1, 1));     
    }

    #[test]
    fn dot() { 

            let a = arr2(&[
                [1., 2., 3.],
                [4., 5., 6.]]);
            let b = arr2(&[
                 [1., 2.],
                 [3., 4.],
                 [5., 6.]]);
            let expected = arr2(&[
                    [22., 28.], 
                    [49., 64.]]);            
            let result = BaseMatrix::dot(&a, &b);
            assert_eq!(result, expected);
    }    

    #[test]
    fn vector_dot() { 
            let a = arr2(&[[1., 2., 3.]]);
            let b = arr2(&[[1., 2., 3.]]);            
            assert_eq!(14., a.vector_dot(&b));
    }

    #[test]
    fn slice() { 

            let a = arr2(
                &[
                    [1., 2., 3., 1., 2.], 
                    [4., 5., 6., 3., 4.], 
                    [7., 8., 9., 5., 6.]]);
            let expected = arr2(
                &[
                    [2., 3.], 
                    [5., 6.]]);
            let result = BaseMatrix::slice(&a, 0..2, 1..3);
            assert_eq!(result, expected);
    }

    #[test]
    fn scalar_ops() { 
            let a = arr2(&[[1., 2., 3.]]);            
            assert_eq!(&arr2(&[[2., 3., 4.]]), a.clone().add_scalar_mut(1.));
            assert_eq!(&arr2(&[[0., 1., 2.]]), a.clone().sub_scalar_mut(1.));
            assert_eq!(&arr2(&[[2., 4., 6.]]), a.clone().mul_scalar_mut(2.));
            assert_eq!(&arr2(&[[0.5, 1., 1.5]]), a.clone().div_scalar_mut(2.));
    }

    #[test]
    fn transpose() {
        let m = arr2(&[[1.0, 3.0], [2.0, 4.0]]);
        let expected = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let m_transposed = m.transpose();
        assert_eq!(m_transposed, expected);       
    }

    #[test]
    fn norm() { 
        let v = arr2(&[[3., -2., 6.]]);            
        assert_eq!(v.norm(1.), 11.);
        assert_eq!(v.norm(2.), 7.);
        assert_eq!(v.norm(std::f64::INFINITY), 6.);
        assert_eq!(v.norm(std::f64::NEG_INFINITY), 2.);
    }

    #[test]
    fn negative_mut() { 
        let mut v = arr2(&[[3., -2., 6.]]);       
        v.negative_mut();     
        assert_eq!(v, arr2(&[[-3., 2., -6.]]));        
    }

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

    #[test]
    fn copy_from() {
        let mut src = arr2(&[[1., 2., 3.]]);
        let dst = Array2::<f64>::zeros((1, 3)); 
        src.copy_from(&dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn sum() {
        let src = arr2(&[[1., 2., 3.]]);        
        assert_eq!(src.sum(), 6.);
    }

    #[test]
    fn max_diff() {
        let a1 = arr2(&[[1., 2., 3.], [4., -5., 6.]]);    
        let a2 = arr2(&[[2., 3., 4.], [1., 0., -12.]]);        
        assert_eq!(a1.max_diff(&a2), 18.);
        assert_eq!(a2.max_diff(&a2), 0.);
    }

    #[test]
    fn softmax_mut(){
        let mut prob: Array2<f64> = arr2(&[[1., 2., 3.]]);  
        prob.softmax_mut();            
        assert!((BaseMatrix::get(&prob, 0, 0) - 0.09).abs() < 0.01);     
        assert!((BaseMatrix::get(&prob, 0, 1) - 0.24).abs() < 0.01);     
        assert!((BaseMatrix::get(&prob, 0, 2) - 0.66).abs() < 0.01); 
    }

    #[test]
    fn pow_mut(){
        let mut a = arr2(&[[1., 2., 3.]]);  
        a.pow_mut(3.);
        assert_eq!(a, arr2(&[[1., 8., 27.]]));
    }

    #[test]
    fn argmax(){
        let a = arr2(&[[1., 2., 3.], [-5., -6., -7.], [0.1, 0.2, 0.1]]);  
        let res = a.argmax();
        assert_eq!(res, vec![2, 0, 1]);
    }

    #[test]
    fn unique(){
        let a = arr2(&[[1., 2., 2.], [-2., -6., -7.], [2., 3., 4.]]);  
        let res = a.unique();
        assert_eq!(res.len(), 7);
        assert_eq!(res, vec![-7., -6., -2., 1., 2., 3., 4.]);
    }

    #[test]
    fn get_row_as_vector(){
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);  
        let res = a.get_row_as_vec(1);
        assert_eq!(res, vec![4., 5., 6.]);        
    }

    #[test]
    fn get_col_as_vector(){
        let a = arr2(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);  
        let res = a.get_col_as_vec(1);
        assert_eq!(res, vec![2., 5., 8.]);        
    }

    #[test]
    fn col_mean(){
        let a = arr2(&[[1., 2., 3.],
                       [4., 5., 6.], 
                       [7., 8., 9.]]);  
        let res = a.column_mean();
        assert_eq!(res, vec![4., 5., 6.]);        
    }

    #[test]
    fn eye(){
        let a = arr2(&[[1., 0., 0.],
                       [0., 1., 0.], 
                       [0., 0., 1.]]);  
        let res: Array2<f64> = BaseMatrix::eye(3);
        assert_eq!(res, a);
    }

    #[test]
    fn rand() {
        let m: Array2<f64> = BaseMatrix::rand(3, 3);
        for c in 0..3 {
            for r in 0..3 {
                assert!(m[[r, c]] != 0f64);
            }
        }
    }

    #[test]
    fn approximate_eq() {
        let a = arr2(&[[1., 2., 3.],
            [4., 5., 6.], 
            [7., 8., 9.]]);
        let noise = arr2(&[[1e-5, 2e-5, 3e-5],
            [4e-5, 5e-5, 6e-5], 
            [7e-5, 8e-5, 9e-5]]);        
        assert!(a.approximate_eq(&(&noise + &a), 1e-4));
        assert!(!a.approximate_eq(&(&noise + &a), 1e-5));
    }

    #[test]
    fn abs_mut() {
        let mut a = arr2(&[[1., -2.], [3., -4.]]);
        let expected = arr2(&[[1., 2.], [3., 4.]]);
        a.abs_mut();
        assert_eq!(a, expected);
    }
}