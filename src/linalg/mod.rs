pub mod naive;
pub mod qr;
pub mod svd;
pub mod evd;
pub mod ndarray_bindings;

use std::ops::Range;
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::math::num::FloatExt;
use svd::SVDDecomposableMatrix;
use evd::EVDDecomposableMatrix;
use qr::QRDecomposableMatrix;

pub trait BaseMatrix<T: FloatExt + Debug>: Clone + Debug {  

    type RowVector: Clone + Debug;    

    fn from_row_vector(vec: Self::RowVector) -> Self;

    fn to_row_vector(self) -> Self::RowVector;

    fn get(&self, row: usize, col: usize) -> T; 

    fn get_row_as_vec(&self, row: usize) -> Vec<T>;

    fn get_col_as_vec(&self, col: usize) -> Vec<T>;    

    fn set(&mut self, row: usize, col: usize, x: T);         

    fn eye(size: usize) -> Self;

    fn zeros(nrows: usize, ncols: usize) -> Self;

    fn ones(nrows: usize, ncols: usize) -> Self;

    fn to_raw_vector(&self) -> Vec<T>;

    fn fill(nrows: usize, ncols: usize, value: T) -> Self;

    fn shape(&self) -> (usize, usize);    

    fn v_stack(&self, other: &Self) -> Self;

    fn h_stack(&self, other: &Self) -> Self;

    fn dot(&self, other: &Self) -> Self;    

    fn vector_dot(&self, other: &Self) -> T;     

    fn slice(&self, rows: Range<usize>, cols: Range<usize>) -> Self;    

    fn approximate_eq(&self, other: &Self, error: T) -> bool;

    fn add_mut(&mut self, other: &Self) -> &Self;

    fn sub_mut(&mut self, other: &Self) -> &Self;

    fn mul_mut(&mut self, other: &Self) -> &Self;

    fn div_mut(&mut self, other: &Self) -> &Self;

    fn div_element_mut(&mut self, row: usize, col: usize, x: T);

    fn mul_element_mut(&mut self, row: usize, col: usize, x: T);

    fn add_element_mut(&mut self, row: usize, col: usize, x: T);

    fn sub_element_mut(&mut self, row: usize, col: usize, x: T);

    fn add(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.add_mut(other);
        r
    }

    fn sub(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.sub_mut(other);
        r
    }

    fn mul(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.mul_mut(other);
        r
    }

    fn div(&self, other: &Self) -> Self {
        let mut r = self.clone();
        r.div_mut(other);
        r
    }

    fn add_scalar_mut(&mut self, scalar: T) -> &Self;

    fn sub_scalar_mut(&mut self, scalar: T) -> &Self;

    fn mul_scalar_mut(&mut self, scalar: T) -> &Self;

    fn div_scalar_mut(&mut self, scalar: T) -> &Self;

    fn add_scalar(&self, scalar: T) -> Self{
        let mut r = self.clone();
        r.add_scalar_mut(scalar);
        r
    }

    fn sub_scalar(&self, scalar: T) -> Self{
        let mut r = self.clone();
        r.sub_scalar_mut(scalar);
        r
    }

    fn mul_scalar(&self, scalar: T) -> Self{
        let mut r = self.clone();
        r.mul_scalar_mut(scalar);
        r
    }

    fn div_scalar(&self, scalar: T) -> Self{
        let mut r = self.clone();
        r.div_scalar_mut(scalar);
        r
    }

    fn transpose(&self) -> Self;    

    fn rand(nrows: usize, ncols: usize) -> Self;

    fn norm2(&self) -> T;

    fn norm(&self, p:T) -> T;

    fn column_mean(&self) -> Vec<T>;

    fn negative_mut(&mut self);

    fn negative(&self) -> Self {
        let mut result = self.clone();
        result.negative_mut();
        result
    }    

    fn reshape(&self, nrows: usize, ncols: usize) -> Self;

    fn copy_from(&mut self, other: &Self);

    fn abs_mut(&mut self) -> &Self;

    fn abs(&self) -> Self {
        let mut result = self.clone();
        result.abs_mut();
        result
    }

    fn sum(&self) -> T;

    fn max_diff(&self, other: &Self) -> T;   
    
    fn softmax_mut(&mut self); 

    fn pow_mut(&mut self, p: T) -> &Self;

    fn pow(&mut self, p: T) -> Self {
        let mut result = self.clone();
        result.pow_mut(p);
        result
    }

    fn argmax(&self) -> Vec<usize>; 
    
    fn unique(&self) -> Vec<T>;    

}

pub trait Matrix<T: FloatExt + Debug>: BaseMatrix<T> + SVDDecomposableMatrix<T> + EVDDecomposableMatrix<T> + QRDecomposableMatrix<T> {}

pub fn row_iter<F: FloatExt + Debug, M: Matrix<F>>(m: &M) -> RowIter<F, M> {
    RowIter{
        m: m,
        pos: 0,
        max_pos: m.shape().0,
        phantom: PhantomData
    }
}

pub struct RowIter<'a, T: FloatExt + Debug, M: Matrix<T>> {
    m: &'a M,
    pos: usize,
    max_pos: usize,
    phantom: PhantomData<&'a T>
}

impl<'a, T: FloatExt + Debug, M: Matrix<T>> Iterator for RowIter<'a, T, M> {

    type Item = Vec<T>;

    fn next(&mut self) -> Option<Vec<T>> {
        let res;
        if self.pos < self.max_pos {
            res = Some(self.m.get_row_as_vec(self.pos))
        } else {
            res = None
        }
        self.pos += 1;
        res
    }

}