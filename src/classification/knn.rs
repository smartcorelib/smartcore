use super::Classifier;
use crate::math::distance::Distance;
use crate::math::distance::euclidian::EuclidianDistance;
use crate::algorithm::sort::heap_select::HeapSelect;
use ndarray::prelude::*;
use num_traits::Signed;
use num_traits::{Float, Num};
use std::marker::PhantomData;
use std::cmp::{Ordering, PartialOrd};
use std::fmt::Debug;

pub struct KNNClassifier<E> {
    y: Option<Array1<E>>
}

pub trait KNNAlgorithm<T: Clone + Debug>{
    fn find(&self, from: &T, k: usize) -> Vec<&T>;
}

pub struct SimpleKNNAlgorithm<T, D: Distance<T>>
{
    data: Vec<T>,
    distance: D  
}

impl<T: Clone + Debug, D: Distance<T>> KNNAlgorithm<T> for SimpleKNNAlgorithm<T, D>
{
    fn find(&self, from: &T, k: usize) -> Vec<&T> {
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

            let d = D::distance(&from, &self.data[i]);
            let datum = heap.peek_mut();            
            if d < datum.distance {
                datum.distance = d;
                datum.index = Some(i);
                heap.heapify();
            }
        }   

        heap.sort(); 

        heap.get().into_iter().flat_map(|x| x.index).map(|i| &self.data[i]).collect()
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

impl<A1, A2>  Classifier<A1, A2> for KNNClassifier<A2> 
where    
    A2: Signed + Clone,    
 {
    fn fit(&mut self, x: &Array2<A1>, y: &Array1<A2>){ 
          self.y = Some(Array1::<A2>::zeros(ArrayBase::len(y)));           
    }

    fn predict(&self, x: &Array2<A1>) -> Array1<A2>{             
        let array = Array1::<A2>::zeros(ArrayBase::len(self.y.as_ref().unwrap()));
        array
    }

}

#[cfg(test)]
mod tests {    
    use super::*;   

    struct SimpleDistance{}

    impl Distance<i32> for SimpleDistance {
        fn distance(a: &i32, b: &i32) -> f64 {
            (a - b).abs() as f64
        }
    }     

    #[test]
    fn knn_fit_predict() {        
        let mut knn = KNNClassifier{y: None};
        let x = arr2(&[[1, 2, 3],[4, 5, 6]]);        
        let y = arr1(&[1, 2]);
        knn.fit(&x, &y);
        let r = knn.predict(&x);
        assert_eq!(2, ArrayBase::len(&r));
    }

    #[test]
    fn knn_find() {        
        let sKnn = SimpleKNNAlgorithm{
            data: vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            distance: SimpleDistance{}    
        };        

        assert_eq!(vec!(&2, &3, &1), sKnn.find(&2, 3));
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