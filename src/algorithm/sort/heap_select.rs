use std::cmp::Ordering;

pub struct HeapSelect<T: std::cmp::Ord> {

    k: usize,
    n: usize,
    sorted: bool,
    heap: Vec<T>


}

impl<T: std::cmp::Ord> HeapSelect<T> {

    pub fn from_vec(vec: Vec<T>) -> HeapSelect<T> {
        HeapSelect{
            k: vec.len(),
            n: 0,
            sorted: false,
            heap: vec
        }
    }

    pub fn add(&mut self, element: T) {
        self.sorted = false;
        if self.n < self.k {
            self.heap[self.n] = element;
            self.n += 1;
            if self.n == self.k {
                self.heapify();
            }
        } else {
            self.n += 1;
            if element.cmp(&self.heap[0]) == Ordering::Less {
                self.heap[0] = element;                
            }
        }
    }

    pub fn heapify(&mut self){

    }

}

#[cfg(test)]
mod tests {    
    use super::*;        

    #[test]
    fn test_from_vec() {        
        let heap = HeapSelect::from_vec(vec!(1, 2, 3));  
        assert_eq!(3, heap.k);         
    }

    #[test]
    fn test_add() {        
        let mut heap = HeapSelect::from_vec(Vec::<i32>::new()); 
        heap.add(1);
        heap.add(2);
        heap.add(3);
        assert_eq!(3, heap.n);         
    }
}