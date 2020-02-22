use crate::algorithm::neighbour::KNNAlgorithm;
use crate::algorithm::sort::heap_select::HeapSelect;
use std::cmp::{Ordering, PartialOrd};
use num_traits::Float;

pub struct LinearKNNSearch<'a, T> {
    distance: Box<dyn Fn(&T, &T) -> f64 + 'a>,
    data: Vec<T>
}

impl<'a, T> KNNAlgorithm<T> for LinearKNNSearch<'a, T>
{
    fn find(&self, from: &T, k: usize) -> Vec<usize> {
        if k < 1 || k > self.data.len() {
            panic!("k should be >= 1 and <= length(data)");
        }        
        
        let mut heap = HeapSelect::<KNNPoint>::with_capacity(k); 

        for _ in 0..k {
            heap.add(KNNPoint{
                distance: Float::infinity(),
                index: None
            });
        }

        for i in 0..self.data.len() {

            let d = (self.distance)(&from, &self.data[i]);
            let datum = heap.peek_mut();            
            if d < datum.distance {
                datum.distance = d;
                datum.index = Some(i);
                heap.heapify();
            }
        }   

        heap.sort(); 

        heap.get().into_iter().flat_map(|x| x.index).collect()
    }
}

impl<'a, T> LinearKNNSearch<'a, T> {
    pub fn new(data: Vec<T>, distance: &'a dyn Fn(&T, &T) -> f64) -> LinearKNNSearch<T>{
        LinearKNNSearch{
            data: data,
            distance: Box::new(distance)
        }
    }
}

#[derive(Debug)]
struct KNNPoint {
    distance: f64,
    index: Option<usize>
}

impl PartialOrd for KNNPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for KNNPoint {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for KNNPoint {}

#[cfg(test)]
mod tests {    
    use super::*;      
    use crate::math::distance::euclidian;  

    struct SimpleDistance{}

    impl SimpleDistance {
        fn distance(a: &i32, b: &i32) -> f64 {
            (a - b).abs() as f64
        }
    }    

    #[test]
    fn knn_find() {        
        let data1 = vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        let algorithm1 = LinearKNNSearch::new(data1, &SimpleDistance::distance);

        assert_eq!(vec!(1, 2, 0), algorithm1.find(&2, 3));

        let data2 = vec!(vec![1., 1.], vec![2., 2.], vec![3., 3.], vec![4., 4.], vec![5., 5.]);

        let algorithm2 = LinearKNNSearch::new(data2, &euclidian::distance);

        assert_eq!(vec!(2, 3, 1), algorithm2.find(&vec![3., 3.], 3));
    }

    #[test]
    fn knn_point_eq() {
        let point1 = KNNPoint{
            distance: 10.,
            index: Some(0)
        };

        let point2 = KNNPoint{
            distance: 100.,
            index: Some(1)
        };

        let point3 = KNNPoint{
            distance: 10.,
            index: Some(2)
        };

        let point_inf = KNNPoint{
            distance: Float::infinity(),
            index: Some(3)
        };

        assert!(point2 > point1);
        assert_eq!(point3, point1);
        assert_ne!(point3, point2);
        assert!(point_inf > point3 && point_inf > point2 && point_inf > point1);        
    }
}