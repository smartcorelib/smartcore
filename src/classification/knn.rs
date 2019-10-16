use super::Classifier;
use std::collections::HashSet;
use crate::algorithm::neighbour::{KNNAlgorithm, KNNAlgorithmName};
use crate::algorithm::neighbour::linear_search::LinearKNNSearch;
use crate::algorithm::neighbour::cover_tree::CoverTree;
use crate::common::Nominal;
use ndarray::{ArrayBase, Data, Ix1, Ix2};
use std::fmt::Debug;


type F<X> = dyn Fn(&X, &X) -> f64;

pub struct KNNClassifier<'a, X, Y> 
where
    Y: Nominal,
    X: Debug
{     
    classes: Vec<Y>,
    y: Vec<usize>,    
    knn_algorithm: Box<dyn KNNAlgorithm<X> + 'a>,
    k: usize,        
}

impl<'a, X, Y> KNNClassifier<'a, X, Y> 
where
    Y: Nominal,
    X: Debug
{

    pub fn fit(x: Vec<X>, y: Vec<Y>, k: usize, distance: &'a F<X>, algorithm: KNNAlgorithmName) -> KNNClassifier<X, Y> {

        assert!(Vec::len(&x) == Vec::len(&y), format!("Size of x should equal size of y; |x|=[{}], |y|=[{}]", Vec::len(&x), Vec::len(&y)));

        assert!(k > 1, format!("k should be > 1, k=[{}]", k));      
        
        let c_hash: HashSet<Y> = y.clone().into_iter().collect(); 
        let classes: Vec<Y> = c_hash.into_iter().collect();
        let y_i:Vec<usize> = y.into_iter().map(|y| classes.iter().position(|yy| yy == &y).unwrap()).collect();    

        let knn_algorithm: Box<dyn KNNAlgorithm<X> + 'a> = match algorithm {
            KNNAlgorithmName::CoverTree => Box::new(CoverTree::<X>::new(x, distance)),
            KNNAlgorithmName::LinearSearch => Box::new(LinearKNNSearch::<X>::new(x, distance))
        };

        KNNClassifier{classes:classes, y: y_i, k: k, knn_algorithm: knn_algorithm}

    }
    
}

impl<'a, X, Y> Classifier<X, Y> for KNNClassifier<'a, X, Y> 
where    
    Y: Nominal,
    X: Debug
 {    

    fn predict(&self, x: &X) -> Y {                       
        let idxs = self.knn_algorithm.find(x, self.k);        
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

#[cfg(test)]
mod tests {    
    use super::*;          
    use crate::math::distance::Distance;
    use ndarray::{arr1, arr2, Array1}; 

    #[test]
    fn knn_fit_predict() {                
        let x = arr2(&[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]); 
        let y = arr1(&[2, 2, 2, 3, 3]);        
        let knn = KNNClassifier::fit(NDArrayUtils::array2_to_vec(&x), y.to_vec(), 3, &Array1::distance, KNNAlgorithmName::LinearSearch);        
        let r = knn.predict_vec(&NDArrayUtils::array2_to_vec(&x));
        assert_eq!(5, Vec::len(&r));
        assert_eq!(y.to_vec(), r);
    }
}