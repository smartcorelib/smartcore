use super::Classifier;
use crate::algorithm::sort::heap_select::HeapSelect;
use crate::common::AnyNumber;
use ndarray::prelude::*;
use ndarray::{ArrayBase, Data, Ix1, Ix2};
use num_traits::{Float};
use std::cmp::{Ordering, PartialOrd};
use ndarray::arr1;

pub struct KNNClassifier<X, Y, F> 
where
    X: AnyNumber,
    Y: AnyNumber,    
    F: Fn(&Array1<X>, &Array1<X>) -> f64
{     
    y: Vec<Y>,  
    distance: F,
    k: usize,
    knn_algorithm: Box<KNNAlgorithm<Array1<X>, F>>    
}

impl<X, Y, F> KNNClassifier<X, Y, F> 
where
    X: AnyNumber,
    Y: AnyNumber,    
    F: Fn(&Array1<X>, &Array1<X>) -> f64
{

    pub fn fit<SX: Data<Elem = X>, SY: Data<Elem = Y>>(x: &ArrayBase<SX, Ix2>, y: &ArrayBase<SY, Ix1>, k: usize, distance: F) -> KNNClassifier<X, Y, F> {

        assert!(ArrayBase::shape(x)[0] == ArrayBase::shape(y)[0], format!("Size of x should equal size of y; |x|=[{}], |y|=[{}]", ArrayBase::shape(x)[0], ArrayBase::shape(y)[0]));

        assert!(k > 1, format!("k should be > 1, k=[{}]", k));        
                  
        let v: Vec<Array1<X>> = x.outer_iter().map(|x| x.to_owned()).collect(); 

        let knn = Box::new(SimpleKNNAlgorithm{
            data: v
        });       

        KNNClassifier{y: y.to_owned().to_vec(), k: k, distance: distance, knn_algorithm: knn}
    }
}

impl<X, Y, SX, F> Classifier<X, Y, SX> for KNNClassifier<X, Y, F> 
where    
    X: AnyNumber,
    Y: AnyNumber,
    SX: Data<Elem = X>,    
    F: Fn(&Array1<X>, &Array1<X>) -> f64
 {    

    fn predict(&self, x: &ArrayBase<SX, Ix2>) -> Array1<Y> {             
        let mut result = Vec::new();
        for x in x.outer_iter() {
            let idxs = self.knn_algorithm.find(&x.to_owned(), self.k, &self.distance);
            let mut sum: Y = Y::zero();
            let mut count = 0;
            for i in idxs {
                sum = sum + self.y[i].to_owned();
                count += 1;
            }
            result.push(sum / Y::from_u64(count).unwrap());
        }          
        arr1(&result)
    }

}

pub trait KNNAlgorithm<T: Clone, F: Fn(&T, &T) -> f64>{
    fn find(&self, from: &T, k: usize, d: &F) -> Vec<usize>;
}

pub struct SimpleKNNAlgorithm<T>
{
    data: Vec<T> 
}

impl<T: Clone, F: Fn(&T, &T) -> f64> KNNAlgorithm<T, F> for SimpleKNNAlgorithm<T>
{
    fn find(&self, from: &T, k: usize, d: &F) -> Vec<usize> {
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

            let d = d(&from, &self.data[i]);
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
    use crate::math::distance::Distance;
    use crate::math::distance::euclidian::EuclidianDistance; 

    struct SimpleDistance{}

    impl Distance<i32> for SimpleDistance {
        fn distance(a: &i32, b: &i32) -> f64 {
            (a - b).abs() as f64
        }
    }     

    #[test]
    fn knn_fit_predict() {                
        let x = arr2(&[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]);        
        let y = arr1(&[1, 2, 3, 4, 5]);
        let knn = KNNClassifier::fit(&x, &y, 3, EuclidianDistance::distance);
        let r = knn.predict(&x);
        assert_eq!(5, ArrayBase::len(&r));
        assert_eq!(arr1(&[2, 2, 3, 4, 4]), r);
    }

    #[test]
    fn knn_find() {        
        let simple_knn = SimpleKNNAlgorithm{
            data: vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        };            

        assert_eq!(vec!(1, 2, 0), simple_knn.find(&2, 3, &SimpleDistance::distance));

        let knn2 = SimpleKNNAlgorithm{
            data: vec!(arr1(&[1, 1]), arr1(&[2, 2]), arr1(&[3, 3]), arr1(&[4, 4]), arr1(&[5, 5]))
        }; 

        assert_eq!(vec!(2, 3, 1), knn2.find(&arr1(&[3, 3]), 3, &EuclidianDistance::distance));
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