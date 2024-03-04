use std::fmt::{Debug, Display};
use std::ops::Range;

use crate::linalg::basic::arrays::{Array, Array1, ArrayView1, MutArray, MutArrayView1};

/// Provide mutable window on array
#[derive(Debug)]
pub struct VecMutView<'a, T: Debug + Display + Copy + Sized> {
    ptr: &'a mut [T],
}

/// Provide window on array
#[derive(Debug, Clone)]
pub struct VecView<'a, T: Debug + Display + Copy + Sized> {
    ptr: &'a [T],
}

impl<T: Debug + Display + Copy + Sized> Array<T, usize> for &[T] {
    fn get(&self, i: usize) -> &T {
        &self[i]
    }

    fn shape(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter())
    }
}

impl<T: Debug + Display + Copy + Sized> Array<T, usize> for Vec<T> {
    fn get(&self, i: usize) -> &T {
        &self[i]
    }

    fn shape(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter())
    }
}

impl<T: Debug + Display + Copy + Sized> MutArray<T, usize> for Vec<T> {
    fn set(&mut self, i: usize, x: T) {
        // NOTE: this panics in case of out of bounds index
        self[i] = x
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter_mut())
    }
}

impl<T: Debug + Display + Copy + Sized> ArrayView1<T> for Vec<T> {}
impl<T: Debug + Display + Copy + Sized> ArrayView1<T> for &[T] {}

impl<T: Debug + Display + Copy + Sized> MutArrayView1<T> for Vec<T> {}

impl<T: Debug + Display + Copy + Sized> Array1<T> for Vec<T> {
    fn slice<'a>(&'a self, range: Range<usize>) -> Box<dyn ArrayView1<T> + 'a> {
        assert!(
            range.end <= self.len(),
            "`range` should be <= {}",
            self.len()
        );
        let view = VecView { ptr: &self[range] };
        Box::new(view)
    }

    fn slice_mut<'b>(&'b mut self, range: Range<usize>) -> Box<dyn MutArrayView1<T> + 'b> {
        assert!(
            range.end <= self.len(),
            "`range` should be <= {}",
            self.len()
        );
        let view = VecMutView {
            ptr: &mut self[range],
        };
        Box::new(view)
    }

    fn fill(len: usize, value: T) -> Self {
        vec![value; len]
    }

    fn from_iterator<I: Iterator<Item = T>>(iter: I, len: usize) -> Self
    where
        Self: Sized,
    {
        let mut v: Vec<T> = Vec::with_capacity(len);
        iter.take(len).for_each(|i| v.push(i));
        v
    }

    fn from_vec_slice(slice: &[T]) -> Self {
        let mut v: Vec<T> = Vec::with_capacity(slice.len());
        slice.iter().for_each(|i| v.push(*i));
        v
    }

    fn from_slice(slice: &dyn ArrayView1<T>) -> Self {
        let mut v: Vec<T> = Vec::with_capacity(slice.shape());
        slice.iterator(0).for_each(|i| v.push(*i));
        v
    }
}

impl<'a, T: Debug + Display + Copy + Sized> Array<T, usize> for VecMutView<'a, T> {
    fn get(&self, i: usize) -> &T {
        &self.ptr[i]
    }

    fn shape(&self) -> usize {
        self.ptr.len()
    }

    fn is_empty(&self) -> bool {
        self.ptr.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.ptr.iter())
    }
}

impl<'a, T: Debug + Display + Copy + Sized> MutArray<T, usize> for VecMutView<'a, T> {
    fn set(&mut self, i: usize, x: T) {
        self.ptr[i] = x;
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.ptr.iter_mut())
    }
}

impl<'a, T: Debug + Display + Copy + Sized> ArrayView1<T> for VecMutView<'a, T> {}
impl<'a, T: Debug + Display + Copy + Sized> MutArrayView1<T> for VecMutView<'a, T> {}

impl<'a, T: Debug + Display + Copy + Sized> Array<T, usize> for VecView<'a, T> {
    fn get(&self, i: usize) -> &T {
        &self.ptr[i]
    }

    fn shape(&self) -> usize {
        self.ptr.len()
    }

    fn is_empty(&self) -> bool {
        self.ptr.len() > 0
    }

    fn iterator<'b>(&'b self, axis: u8) -> Box<dyn Iterator<Item = &'b T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.ptr.iter())
    }
}

impl<'a, T: Debug + Display + Copy + Sized> ArrayView1<T> for VecView<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numbers::basenum::Number;

    fn dot_product<T: Number, V: Array1<T>>(v: &V) -> T {
        let vv = V::zeros(10);
        let v_s = vv.slice(0..3);

        v_s.dot(v)
    }

    fn vector_ops<T: Number + PartialOrd, V: Array1<T>>(_: &V) -> T {
        let v = V::zeros(10);
        v.max()
    }

    #[test]
    fn test_get_set() {
        let mut x = vec![1, 2, 3];
        assert_eq!(3, *x.get(2));
        x.set(1, 1);
        assert_eq!(1, *x.get(1));
    }

    #[test]
    #[should_panic]
    fn test_failed_set() {
        vec![1, 2, 3].set(3, 1);
    }

    #[test]
    #[should_panic]
    fn test_failed_get() {
        vec![1, 2, 3].get(3);
    }

    #[test]
    fn test_len() {
        let x = [1, 2, 3];
        assert_eq!(3, x.len());
    }

    #[test]
    fn test_is_empty() {
        assert!(vec![1; 0].is_empty());
        assert!(!vec![1, 2, 3].is_empty());
    }

    #[test]
    fn test_iterator() {
        let v: Vec<i32> = vec![1, 2, 3].iterator(0).map(|&v| v * 2).collect();
        assert_eq!(vec![2, 4, 6], v);
    }

    #[test]
    #[should_panic]
    fn test_failed_iterator() {
        let _ = vec![1, 2, 3].iterator(1);
    }

    #[test]
    fn test_mut_iterator() {
        let mut x = vec![1, 2, 3];
        x.iterator_mut(0).for_each(|v| *v *= 2);
        assert_eq!(vec![2, 4, 6], x);
    }

    #[test]
    #[should_panic]
    fn test_failed_mut_iterator() {
        let _ = vec![1, 2, 3].iterator_mut(1);
    }

    #[test]
    fn test_slice() {
        let x = vec![1, 2, 3, 4, 5];
        let x_slice = x.slice(2..3);
        assert_eq!(1, x_slice.shape());
        assert_eq!(3, *x_slice.get(0));
    }

    #[test]
    #[should_panic]
    fn test_failed_slice() {
        vec![1, 2, 3].slice(0..4);
    }

    #[test]
    fn test_mut_slice() {
        let mut x = vec![1, 2, 3, 4, 5];
        let mut x_slice = x.slice_mut(2..4);
        x_slice.set(0, 9);
        assert_eq!(2, x_slice.shape());
        assert_eq!(9, *x_slice.get(0));
        assert_eq!(4, *x_slice.get(1));
    }

    #[test]
    #[should_panic]
    fn test_failed_mut_slice() {
        vec![1, 2, 3].slice_mut(0..4);
    }

    #[test]
    fn test_init() {
        assert_eq!(Vec::fill(3, 0), vec![0, 0, 0]);
        assert_eq!(
            Vec::from_iterator([0, 1, 2, 3].iter().cloned(), 3),
            vec![0, 1, 2]
        );
        assert_eq!(Vec::from_vec_slice(&[0, 1, 2]), vec![0, 1, 2]);
        assert_eq!(Vec::from_vec_slice(&[0, 1, 2, 3, 4][2..]), vec![2, 3, 4]);
        assert_eq!(Vec::from_slice(&vec![1, 2, 3, 4, 5]), vec![1, 2, 3, 4, 5]);
        assert_eq!(
            Vec::from_slice(vec![1, 2, 3, 4, 5].slice(0..3).as_ref()),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn test_mul_scalar() {
        let mut x = vec![1., 2., 3.];

        let mut y = Vec::<f32>::zeros(10);

        y.slice_mut(0..2).add_scalar_mut(1.0);
        y.sub_scalar(1.0);
        x.slice_mut(0..2).sub_scalar_mut(2.);

        assert_eq!(vec![-1.0, 0.0, 3.0], x);
    }

    #[test]
    fn test_dot() {
        let y_i = vec![1, 2, 3];
        let y = vec![1.0, 2.0, 3.0];

        println!("Regular dot1: {:?}", dot_product(&y));

        let x = vec![4.0, 5.0, 6.0];
        assert_eq!(32.0, y.slice(0..3).dot(&(*x.slice(0..3))));
        assert_eq!(32.0, y.slice(0..3).dot(&x));
        assert_eq!(32.0, y.dot(&x));
        assert_eq!(14, y_i.dot(&y_i));
    }

    #[test]
    fn test_operators() {
        let mut x: Vec<f32> = Vec::zeros(10);

        x.add_scalar(15.0);
        {
            let mut x_s = x.slice_mut(0..5);
            x_s.add_scalar_mut(1.0);
            assert_eq!(
                vec![1.0, 1.0, 1.0, 1.0, 1.0],
                x_s.iterator(0).copied().collect::<Vec<f32>>()
            );
        }

        assert_eq!(1.0, x.slice(2..3).min());

        assert_eq!(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], x);
    }

    #[test]
    fn test_vector_ops() {
        let x = vec![1., 2., 3.];

        vector_ops(&x);
    }
}
