use super::Classifier;
use super::super::math::distance::Distance;
use super::super::math::distance::euclidian::EuclidianDistance;
use ndarray::prelude::*;
use num_traits::Signed;
use num_traits::Float;
use std::marker::PhantomData;

pub struct KNNClassifier<E> {
    y: Option<Array1<E>>
}

pub trait KNNAlgorithm<T>{
    fn find(&self, from: &T, k: i32) -> &Vec<T>;
}

pub struct SimpleKNNAlgorithm<T, A, D> 
where     
    A: Float,
    D: Distance<T, A>
{
    data: Vec<T>,
    distance: D,
    __phantom: PhantomData<A>  
}

impl<T, A, D> KNNAlgorithm<T> for SimpleKNNAlgorithm<T, A, D> 
where     
    A: Float,
    D: Distance<T, A>
{
    fn find(&self, from: &T, k: i32) -> &Vec<T> {
        &self.data
    }
}

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
            data: vec!(arr1(&[1., 2.]), arr1(&[1., 2.]), arr1(&[1., 2.])),
            distance: EuclidianDistance{},
            __phantom: PhantomData
        };

        assert_eq!(&vec!(arr1(&[1., 2.]), arr1(&[1., 2.]), arr1(&[1., 2.])), sKnn.find(&arr1(&[1., 2.]), 3));
    }
}