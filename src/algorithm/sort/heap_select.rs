use std::cmp::Ordering;

#[derive(Debug)]
pub struct HeapSelect<T: PartialOrd> {
    k: usize,
    n: usize,
    sorted: bool,
    heap: Vec<T>,
}

impl<'a, T: PartialOrd> HeapSelect<T> {
    pub fn with_capacity(k: usize) -> HeapSelect<T> {
        HeapSelect {
            k: k,
            n: 0,
            sorted: false,
            heap: Vec::<T>::new(),
        }
    }

    pub fn add(&mut self, element: T) {
        self.sorted = false;
        if self.n < self.k {
            self.heap.push(element);
            self.n += 1;
            if self.n == self.k {
                self.heapify();
            }
        } else {
            self.n += 1;
            if element.partial_cmp(&self.heap[0]) == Some(Ordering::Less) {
                self.heap[0] = element;
            }
        }
    }

    pub fn heapify(&mut self) {
        let n = self.heap.len();
        for i in (0..=(n / 2 - 1)).rev() {
            self.sift_down(i, n - 1);
        }
    }

    #[allow(dead_code)]
    pub fn peek(&self) -> &T {
        return &self.heap[0];
    }

    pub fn peek_mut(&mut self) -> &mut T {
        return &mut self.heap[0];
    }

    pub fn sift_down(&mut self, from: usize, n: usize) {
        let mut k = from;
        while 2 * k <= n {
            let mut j = 2 * k;
            if j < n && self.heap[j] < self.heap[j + 1] {
                j += 1;
            }
            if self.heap[k] >= self.heap[j] {
                break;
            }
            self.heap.swap(k, j);
            k = j;
        }
    }

    pub fn get(self) -> Vec<T> {
        return self.heap;
    }

    pub fn sort(&mut self) {
        HeapSelect::shuffle_sort(&mut self.heap, std::cmp::min(self.k, self.n));
    }

    pub fn shuffle_sort(vec: &mut Vec<T>, n: usize) {
        let mut inc = 1;
        while inc <= n {
            inc *= 3;
            inc += 1
        }

        let len = n;
        while inc >= 1 {
            let mut i = inc;
            while i < len {
                let mut j = i;
                while j >= inc && vec[j - inc] > vec[j] {
                    vec.swap(j - inc, j);
                    j -= inc;
                }
                i += 1;
            }
            inc /= 3
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_capacity() {
        let heap = HeapSelect::<i32>::with_capacity(3);
        assert_eq!(3, heap.k);
    }

    #[test]
    fn test_add() {
        let mut heap = HeapSelect::with_capacity(3);
        heap.add(333);
        heap.add(2);
        heap.add(13);
        heap.add(10);
        heap.add(40);
        heap.add(30);
        assert_eq!(6, heap.n);
        assert_eq!(&10, heap.peek());
        assert_eq!(&10, heap.peek_mut());
    }

    #[test]
    fn test_add_ordered() {
        let mut heap = HeapSelect::with_capacity(3);
        heap.add(1.);
        heap.add(2.);
        heap.add(3.);
        heap.add(4.);
        heap.add(5.);
        heap.add(6.);
        let result = heap.get();
        assert_eq!(vec![2., 3., 1.], result);
    }

    #[test]
    fn test_shuffle_sort() {
        let mut v1 = vec![10, 33, 22, 105, 12];
        let n = v1.len();
        HeapSelect::shuffle_sort(&mut v1, n);
        assert_eq!(vec![10, 12, 22, 33, 105], v1);

        let mut v2 = vec![10, 33, 22, 105, 12];
        HeapSelect::shuffle_sort(&mut v2, 3);
        assert_eq!(vec![10, 22, 33, 105, 12], v2);

        let mut v3 = vec![4, 5, 3, 2, 1];
        HeapSelect::shuffle_sort(&mut v3, 3);
        assert_eq!(vec![3, 4, 5, 2, 1], v3);
    }
}
