use super::Classifier;
use std::collections::HashSet;
use crate::algorithm::sort::heap_select::HeapSelect;
use crate::common::Nominal;
use ndarray::{ArrayBase, Data, Ix1, Ix2};
use num_traits::{Float};
use std::cmp::{Ordering, PartialOrd};


type F<X> = Fn(&X, &X) -> f64;

pub struct KNNClassifier<X, Y> 
where
    Y: Nominal
{     
    classes: Vec<Y>,
    y: Vec<usize>,  
    data: Vec<X>,      
    distance: Box<F<X>>,
    k: usize,        
}

impl<X, Y> KNNClassifier<X, Y> 
where
    Y: Nominal
{

    pub fn fit(x: Vec<X>, y: Vec<Y>, k: usize, distance: &'static F<X>) -> KNNClassifier<X, Y> {

        assert!(Vec::len(&x) == Vec::len(&y), format!("Size of x should equal size of y; |x|=[{}], |y|=[{}]", Vec::len(&x), Vec::len(&y)));

        assert!(k > 1, format!("k should be > 1, k=[{}]", k));      
        
        let c_hash: HashSet<Y> = y.clone().into_iter().collect(); 
        let classes: Vec<Y> = c_hash.into_iter().collect();
        let y_i:Vec<usize> = y.into_iter().map(|y| classes.iter().position(|yy| yy == &y).unwrap()).collect();   

        KNNClassifier{classes:classes, y: y_i, data: x, k: k, distance: Box::new(distance)}
    }
    
}

impl<X, Y> Classifier<X, Y> for KNNClassifier<X, Y> 
where    
    Y: Nominal
 {    

    fn predict(&self, x: &X) -> Y {                       
        let idxs = self.data.find(x, self.k, &self.distance);        
        let mut c = vec![0; self.classes.len()];
        let mut max_c = 0;
        let mut max_i = 0;
        for i in idxs {
            c[self.y[i]] += 1;  
            if c[self.y[i]] > max_c {
                max_c = c[self.y[i]];
                max_i = self.y[i];
            }          
        }                    

        self.classes[max_i].clone()
    }

}

pub struct NDArrayUtils {

}

impl NDArrayUtils {

    pub fn array2_to_vec<E, S>(x: &ArrayBase<S, Ix2>) -> Vec<ArrayBase<S, Ix1>> 
    where
        E: Nominal,
        S: Data<Elem = E>,
        std::vec::Vec<ArrayBase<S, Ix1>>: std::iter::FromIterator<ndarray::ArrayBase<ndarray::OwnedRepr<E>, Ix1>>{            
            let x_vec: Vec<ArrayBase<S, Ix1>> = x.outer_iter().map(|x| x.to_owned()).collect();            
            x_vec
    }
}

pub trait KNNAlgorithm<T>{
    fn find(&self, from: &T, k: usize, d: &Fn(&T, &T) -> f64) -> Vec<usize>;
}

impl<T> KNNAlgorithm<T> for Vec<T>
{
    fn find(&self, from: &T, k: usize, d: &Fn(&T, &T) -> f64) -> Vec<usize> {
        if k < 1 || k > self.len() {
            panic!("k should be >= 1 and <= length(data)");
        }        
        
        let mut heap = HeapSelect::<KNNPoint>::with_capacity(k); 

        for _ in 0..k {
            heap.add(KNNPoint{
                distance: Float::infinity(),
                index: None
            });
        }

        for i in 0..self.len() {

            let d = d(&from, &self[i]);
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
    use ndarray::{arr1, arr2, Array1};

    struct SimpleDistance{}

    impl SimpleDistance {
        fn distance(a: &i32, b: &i32) -> f64 {
            (a - b).abs() as f64
        }
    }   

    #[test]
    fn knn_fit_predict() {                
        let x = arr2(&[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]); 
        let y = arr1(&[2, 2, 2, 3, 3]);        
        let knn = KNNClassifier::fit(NDArrayUtils::array2_to_vec(&x), y.to_vec(), 3, &Array1::distance);
        let r = knn.predict_vec(&NDArrayUtils::array2_to_vec(&x));
        assert_eq!(5, Vec::len(&r));
        assert_eq!(y.to_vec(), r);
    }

    #[test]
    fn knn_find() {        
        let data1 = vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);            

        assert_eq!(vec!(1, 2, 0), data1.find(&2, 3, &SimpleDistance::distance));

        let data2 = vec!(arr1(&[1, 1]), arr1(&[2, 2]), arr1(&[3, 3]), arr1(&[4, 4]), arr1(&[5, 5]));

        assert_eq!(vec!(2, 3, 1), data2.find(&arr1(&[3, 3]), 3, &Array1::distance));
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