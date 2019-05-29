use ndarray::prelude::*;
use ndarray::{arr1, arr2};
use ndarray::FixedInitializer;

pub mod knn;

pub trait Classifier<E1, E2>
{

    fn fit(&mut self, x: &Array2<E1>, y: &Array1<E2>);

    fn predict(&self, x: &Array2<E1>) -> Array1<E2>;

}