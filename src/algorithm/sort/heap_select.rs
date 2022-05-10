//! # Heap Selection Algorithm
//!
//! The goal is to find the k smallest elements in a list or array.
use std::cmp::Ordering;
use std::fmt::Debug;

#[derive(Debug)]
pub struct HeapSelection<T: PartialOrd + Debug> {
    k: usize,
    n: usize,
    sorted: bool,
    heap: Vec<T>,
}

impl<'a, T: PartialOrd + Debug> HeapSelection<T> {
    pub fn with_capacity(k: usize) -> HeapSelection<T> {
        HeapSelection {
            k,
            n: 0,
            sorted: false,
            heap: Vec::new(),
        }
    }

    pub fn add(&mut self, element: T) {
        self.sorted = false;
        if self.n < self.k {
            self.heap.push(element);
            self.n += 1;
            if self.n == self.k {
                self.sort();
            }
        } else {
            self.n += 1;
            if element.partial_cmp(&self.heap[0]) == Some(Ordering::Less) {
                self.heap[0] = element;
                self.sift_down(0, self.k - 1);
            }
        }
    }

    pub fn heapify(&mut self) {
        let n = self.heap.len();
        if n <= 1 {
            return;
        }
        for i in (0..=(n / 2 - 1)).rev() {
            self.sift_down(i, n - 1);
        }
    }

    pub fn peek(&self) -> &T {
        if self.sorted {
            &self.heap[0]
        } else {
            self.heap
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }
    }

    pub fn peek_mut(&mut self) -> &mut T {
        &mut self.heap[0]
    }

    pub fn get(self) -> Vec<T> {
        self.heap
    }

    fn sift_down(&mut self, k: usize, n: usize) {
        let mut kk = k;
        while 2 * kk <= n {
            let mut j = 2 * kk;
            if j < n && self.heap[j].partial_cmp(&self.heap[j + 1]) == Some(Ordering::Less) {
                j += 1;
            }
            if self.heap[kk].partial_cmp(&self.heap[j]) == Some(Ordering::Equal)
                || self.heap[kk].partial_cmp(&self.heap[j]) == Some(Ordering::Greater)
            {
                break;
            }
            self.heap.swap(kk, j);
            kk = j;
        }
    }

    fn sort(&mut self) {
        self.sorted = true;
        self.heap.sort_by(|a, b| b.partial_cmp(a).unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn with_capacity() {
        let heap = HeapSelection::<i32>::with_capacity(3);
        assert_eq!(3, heap.k);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_add() {
        let mut heap = HeapSelection::with_capacity(3);
        heap.add(-5);
        assert_eq!(-5, *heap.peek());
        heap.add(333);
        assert_eq!(333, *heap.peek());
        heap.add(13);
        heap.add(10);
        heap.add(2);
        heap.add(0);
        heap.add(40);
        heap.add(30);
        assert_eq!(8, heap.n);
        assert_eq!(vec![2, 0, -5], heap.get());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_add1() {
        let mut heap = HeapSelection::with_capacity(3);
        heap.add(std::f64::INFINITY);
        heap.add(-5f64);
        heap.add(4f64);
        heap.add(-1f64);
        heap.add(2f64);
        heap.add(1f64);
        heap.add(0f64);
        assert_eq!(7, heap.n);
        assert_eq!(vec![0f64, -1f64, -5f64], heap.get());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_add2() {
        let mut heap = HeapSelection::with_capacity(3);
        heap.add(std::f64::INFINITY);
        heap.add(0.0);
        heap.add(8.4852);
        heap.add(5.6568);
        heap.add(2.8284);
        assert_eq!(5, heap.n);
        assert_eq!(vec![5.6568, 2.8284, 0.0], heap.get());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_add_ordered() {
        let mut heap = HeapSelection::with_capacity(3);
        heap.add(1.);
        heap.add(2.);
        heap.add(3.);
        heap.add(4.);
        heap.add(5.);
        heap.add(6.);
        assert_eq!(vec![3., 2., 1.], heap.get());
    }
}
