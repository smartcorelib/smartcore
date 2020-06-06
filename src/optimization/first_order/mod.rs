pub mod gradient_descent;
pub mod lbfgs;

use std::clone::Clone;
use std::fmt::Debug;

use crate::linalg::Matrix;
use crate::math::num::FloatExt;
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{DF, F};

pub trait FirstOrderOptimizer<T: FloatExt> {
    fn optimize<'a, X: Matrix<T>, LS: LineSearchMethod<T>>(
        &self,
        f: &F<T, X>,
        df: &'a DF<X>,
        x0: &X,
        ls: &'a LS,
    ) -> OptimizerResult<T, X>;
}

#[derive(Debug, Clone)]
pub struct OptimizerResult<T: FloatExt, X: Matrix<T>> {
    pub x: X,
    pub f_x: T,
    pub iterations: usize,
}
