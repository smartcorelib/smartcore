use std::fmt::{Debug, Display};
use std::ops::Range;

use crate::linalg::basic::arrays::{
    Array as BaseArray, Array1, ArrayView1, MutArray, MutArrayView1,
};

use ndarray::{s, Array, ArrayBase, ArrayView, ArrayViewMut, Ix1, OwnedRepr};

impl<T: Debug + Display + Copy + Sized> BaseArray<T, usize> for ArrayBase<OwnedRepr<T>, Ix1> {
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

impl<T: Debug + Display + Copy + Sized> MutArray<T, usize> for ArrayBase<OwnedRepr<T>, Ix1> {
    fn set(&mut self, i: usize, x: T) {
        self[i] = x
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter_mut())
    }
}

impl<T: Debug + Display + Copy + Sized> ArrayView1<T> for ArrayBase<OwnedRepr<T>, Ix1> {}

impl<T: Debug + Display + Copy + Sized> MutArrayView1<T> for ArrayBase<OwnedRepr<T>, Ix1> {}

impl<'a, T: Debug + Display + Copy + Sized> BaseArray<T, usize> for ArrayView<'a, T, Ix1> {
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

impl<'a, T: Debug + Display + Copy + Sized> ArrayView1<T> for ArrayView<'a, T, Ix1> {}

impl<'a, T: Debug + Display + Copy + Sized> BaseArray<T, usize> for ArrayViewMut<'a, T, Ix1> {
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

impl<'a, T: Debug + Display + Copy + Sized> MutArray<T, usize> for ArrayViewMut<'a, T, Ix1> {
    fn set(&mut self, i: usize, x: T) {
        self[i] = x;
    }

    fn iterator_mut<'b>(&'b mut self, axis: u8) -> Box<dyn Iterator<Item = &'b mut T> + 'b> {
        assert!(axis == 0, "For one dimensional array `axis` should == 0");
        Box::new(self.iter_mut())
    }
}

impl<'a, T: Debug + Display + Copy + Sized> ArrayView1<T> for ArrayViewMut<'a, T, Ix1> {}
impl<'a, T: Debug + Display + Copy + Sized> MutArrayView1<T> for ArrayViewMut<'a, T, Ix1> {}

impl<T: Debug + Display + Copy + Sized> Array1<T> for ArrayBase<OwnedRepr<T>, Ix1> {
    fn slice<'a>(&'a self, range: Range<usize>) -> Box<dyn ArrayView1<T> + 'a> {
        assert!(
            range.end <= self.len(),
            "`range` should be <= {}",
            self.len()
        );
        Box::new(self.slice(s![range]))
    }

    fn slice_mut<'b>(&'b mut self, range: Range<usize>) -> Box<dyn MutArrayView1<T> + 'b> {
        assert!(
            range.end <= self.len(),
            "`range` should be <= {}",
            self.len()
        );
        Box::new(self.slice_mut(s![range]))
    }

    fn fill(len: usize, value: T) -> Self {
        Array::from_elem(len, value)
    }

    fn from_iterator<I: Iterator<Item = T>>(iter: I, len: usize) -> Self
    where
        Self: Sized,
    {
        Array::from_iter(iter.take(len))
    }

    fn from_vec_slice(slice: &[T]) -> Self {
        Array::from_iter(slice.iter().copied())
    }

    fn from_slice(slice: &dyn ArrayView1<T>) -> Self {
        Array::from_iter(slice.iterator(0).copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_get_set() {
        let mut a = arr1(&[1, 2, 3]);

        assert_eq!(*BaseArray::get(&a, 1), 2);
        a.set(1, 9);
        assert_eq!(a, arr1(&[1, 9, 3]));
    }

    #[test]
    fn test_iterator() {
        let a = arr1(&[1, 2, 3]);

        let v: Vec<i32> = a.iterator(0).map(|&v| v).collect();
        assert_eq!(v, vec!(1, 2, 3));
    }

    #[test]
    fn test_mut_iterator() {
        let mut a = arr1(&[1, 2, 3]);

        a.iterator_mut(0).for_each(|v| *v = 1);
        assert_eq!(a, arr1(&[1, 1, 1]));
    }

    #[test]
    fn test_slice() {
        let x = arr1(&[1, 2, 3, 4, 5]);
        let x_slice = Array1::slice(&x, 2..3);
        assert_eq!(1, x_slice.shape());
        assert_eq!(3, *x_slice.get(0));
    }

    #[test]
    fn test_mut_slice() {
        let mut x = arr1(&[1, 2, 3, 4, 5]);
        let mut x_slice = Array1::slice_mut(&mut x, 2..4);
        x_slice.set(0, 9);
        assert_eq!(2, x_slice.shape());
        assert_eq!(9, *x_slice.get(0));
        assert_eq!(4, *x_slice.get(1));
    }
}
