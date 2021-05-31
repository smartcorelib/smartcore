use std::fmt::{Debug, Display};
use std::ops::Range;

use crate::linalg::base::{Array, MutArray, ArrayView1, MutArrayView1, Array1};

use nalgebra::{Matrix, U1, Dynamic, VecStorage, MatrixSlice, MatrixSliceMut, Scalar, Dim};

impl<T: Debug + Display + Copy + Sized + Scalar> Array<T, usize> for Matrix<T, U1, Dynamic, VecStorage<T, U1, Dynamic>> {

    fn get(&self, i: usize) -> &T {
        &self[i]
    }

    fn shape(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool{
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {        
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter())
    } 

}

impl<T: Debug + Display + Copy + Sized + Scalar> Array<T, usize> for Matrix<T, Dynamic, U1, VecStorage<T, Dynamic, U1>> {

    fn get(&self, i: usize) -> &T {
        &self[i]
    }

    fn shape(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool{
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {        
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter())
    } 

}

impl<T: Debug + Display + Copy + Sized + Scalar> MutArray<T, usize> for Matrix<T, U1, Dynamic, VecStorage<T, U1, Dynamic>> {    

    fn set(&mut self, i: usize, x: T) {        
        self[i] = x
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter_mut())
    }    

}

impl<T: Debug + Display + Copy + Sized + Scalar> MutArray<T, usize> for Matrix<T, Dynamic, U1, VecStorage<T, Dynamic, U1>> {    

    fn set(&mut self, i: usize, x: T) {        
        self[i] = x
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter_mut())
    }    

}

impl<T: Debug + Display + Copy + Sized + Scalar> ArrayView1<T> for Matrix<T, U1, Dynamic, VecStorage<T, U1, Dynamic>> {}

impl<T: Debug + Display + Copy + Sized + Scalar> ArrayView1<T> for Matrix<T, Dynamic, U1, VecStorage<T, Dynamic, U1>> {}

impl<T: Debug + Display + Copy + Sized + Scalar> MutArrayView1<T> for Matrix<T, U1, Dynamic, VecStorage<T, U1, Dynamic>> {}

impl<T: Debug + Display + Copy + Sized + Scalar> MutArrayView1<T> for Matrix<T, Dynamic, U1, VecStorage<T, Dynamic, U1>> {}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> Array<T, usize> for MatrixSlice<'a, T, U1, Dynamic, RStride, CStride> {

    fn get(&self, i: usize) -> &T {
        &self[i]
    }

    fn shape(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool{
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {        
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter())
    } 

}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> Array<T, usize> for MatrixSlice<'a, T, Dynamic, U1, RStride, CStride> {

    fn get(&self, i: usize) -> &T {
        &self[i]
    }

    fn shape(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool{
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {        
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter())
    } 

}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> ArrayView1<T> for MatrixSlice<'a, T, U1, Dynamic, RStride, CStride> {}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> ArrayView1<T> for MatrixSlice<'a, T, Dynamic, U1, RStride, CStride> {}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> Array<T, usize> for MatrixSliceMut<'a, T, U1, Dynamic, RStride, CStride> {

    fn get(&self, i: usize) -> &T {
        &self[i]
    }

    fn shape(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool{
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {        
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter())
    } 

}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> Array<T, usize> for MatrixSliceMut<'a, T, Dynamic, U1, RStride, CStride> {

    fn get(&self, i: usize) -> &T {
        &self[i]
    }

    fn shape(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool{
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {        
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter())
    } 

}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> MutArray<T, usize> for MatrixSliceMut<'a, T, U1, Dynamic, RStride, CStride> {    

    fn set(&mut self, i: usize, x: T) {
        self[i] = x;
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter_mut())
    }
    
}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> MutArray<T, usize> for MatrixSliceMut<'a, T, Dynamic, U1, RStride, CStride> {    

    fn set(&mut self, i: usize, x: T) {
        self[i] = x;
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter_mut())
    }
    
}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> ArrayView1<T> for MatrixSliceMut<'a, T, U1, Dynamic, RStride, CStride> {}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> ArrayView1<T> for MatrixSliceMut<'a, T, Dynamic, U1, RStride, CStride> {}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> MutArrayView1<T> for MatrixSliceMut<'a, T, U1, Dynamic, RStride, CStride> {}

impl<'a, T: Debug + Display + Copy + Sized + Scalar, RStride: Dim, CStride: Dim> MutArrayView1<T> for MatrixSliceMut<'a, T, Dynamic, U1, RStride, CStride> {}

impl<T: Debug + Display + Copy + Sized + Scalar> Array1<T> for Matrix<T, U1, Dynamic, VecStorage<T, U1, Dynamic>> {  

    fn slice<'a>(&'a self, range: Range<usize>) -> Box<dyn ArrayView1<T> +'a> {
        assert!(range.end <= self.len(), "`range` should be <= {}", self.len());
        Box::new(self.columns_range(range))        
    } 

    fn slice_mut<'b>(&'b mut self, range: Range<usize>) -> Box<dyn MutArrayView1<T> + 'b>{
        assert!(range.end <= self.len(), "`range` should be <= {}", self.len());
        Box::new(self.columns_range_mut(range))        
    }

    fn fill(len: usize, value: T) -> Self {
        Self::from_element(len, value)
    }

    fn from_iterator<'a, I: Iterator<Item = &'a T>>(iter: I, len: usize) -> Self where Self: Sized, T: 'a {                
        Self::from_iterator(len, iter.map(|&v| v))
    }

    fn from_vec_slice(slice: &[T]) -> Self {        
        Self::from_row_slice(slice)
    }

    fn from_slice(slice: &dyn ArrayView1<T>) -> Self {
        Self::from_iterator(slice.shape(), slice.iterator(0).map(|&v| v))
    }

}

impl<T: Debug + Display + Copy + Sized + Scalar> Array1<T> for Matrix<T, Dynamic, U1, VecStorage<T, Dynamic, U1>> {  

    fn slice<'a>(&'a self, range: Range<usize>) -> Box<dyn ArrayView1<T> +'a> {
        assert!(range.end <= self.len(), "`range` should be <= {}", self.len());
        Box::new(self.rows_range(range))        
    } 

    fn slice_mut<'b>(&'b mut self, range: Range<usize>) -> Box<dyn MutArrayView1<T> + 'b>{
        assert!(range.end <= self.len(), "`range` should be <= {}", self.len());
        Box::new(self.rows_range_mut(range))        
    }

    fn fill(len: usize, value: T) -> Self {
        Self::from_element(len, value)
    }

    fn from_iterator<'a, I: Iterator<Item = &'a T>>(iter: I, len: usize) -> Self where Self: Sized, T: 'a {                
        Self::from_iterator(len, iter.map(|&v| v))
    }

    fn from_vec_slice(slice: &[T]) -> Self {        
        Self::from_row_slice(slice)
    }

    fn from_slice(slice: &dyn ArrayView1<T>) -> Self {
        Self::from_iterator(slice.shape(), slice.iterator(0).map(|&v| v))
    }

}

#[cfg(test)]
mod tests {
    use super::*;    
    use nalgebra::{RowDVector};

    #[test]
    fn test_get_set() {
        let mut a = RowDVector::from_vec(vec![1, 2, 3]);        
        
        assert_eq!(*Array::get(&a, 1), 2);
        a.set(1, 9);
        assert_eq!(a, RowDVector::from_vec(vec![1, 9, 3]));        
    }

    #[test]
    fn test_iterator() {
        let a = RowDVector::from_vec(vec![1, 2, 3]);        
        
        let v: Vec<i32> =  a.iterator(0).map(|&v| v).collect();
        assert_eq!(v, vec!(1, 2, 3));        
    }

    #[test]
    fn test_mut_iterator() {
        let mut a = RowDVector::from_vec(vec![1, 2, 3]);
        
        a.iterator_mut(0).for_each(|v| *v = 1);
        assert_eq!(a, RowDVector::from_vec(vec![1, 1, 1]));        
    }

    #[test]
    fn test_slice() {   
        let x = RowDVector::from_vec(vec![1, 2, 3, 4, 5]);        
        let x_slice = Array1::slice(&x, 2..3);    
        assert_eq!(1, x_slice.shape());
        assert_eq!(3, *x_slice.get(0));            
    }

    #[test]
    fn test_mut_slice() {   
        let mut x = RowDVector::from_vec(vec![1, 2, 3, 4, 5]);
        let mut x_slice = Array1::slice_mut(&mut x, 2..4);   
        x_slice.set(0, 9);
        assert_eq!(2, x_slice.shape());
        assert_eq!(9, *x_slice.get(0));
        assert_eq!(4, *x_slice.get(1));            
    }

    #[test]
    fn test_fill() {   
        let x: RowDVector<i32> = Array1::fill(3, 1);
        assert_eq!(x, RowDVector::from_vec(vec![1, 1, 1]));            
    }

    #[test]
    fn test_from_iterator() {   
        let x: RowDVector<i32> = Array1::from_iterator(vec![1, 2, 3].iter(), 3);
        assert_eq!(x, RowDVector::from_vec(vec![1, 2, 3]));            
    }

    #[test]
    fn test_from_vec_slice() {   
        let x: RowDVector<i32> = Array1::from_vec_slice(&vec![1, 2, 3]);
        assert_eq!(x, RowDVector::from_vec(vec![1, 2, 3]));            
    }

    #[test]
    fn test_from_slice() {   
        let x = RowDVector::from_vec(vec![1, 2, 3, 4, 5]);
        let y: RowDVector<i32> = Array1::from_slice(Array1::slice(&x, 1..4).as_ref());
        assert_eq!(y, RowDVector::from_vec(vec![2, 3, 4]));            
    }

}

