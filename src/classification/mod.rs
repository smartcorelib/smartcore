use crate::common::AnyNumber;
use ndarray::{Array1, ArrayBase, Data, Ix2};

pub mod knn;

pub trait Classifier<X, Y, SX>
where 
    X: AnyNumber,
    Y: AnyNumber,
    SX: Data<Elem = X>    
{    

    fn predict(&self, x: &ArrayBase<SX, Ix2>) -> Array1<Y>;

}