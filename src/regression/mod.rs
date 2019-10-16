pub mod linear_regression;

use crate::linalg::Matrix;

pub trait Regression<M: Matrix> {    

    fn predict(&self, x: &M) -> M;

}