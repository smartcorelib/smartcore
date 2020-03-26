pub mod lbfgs;
pub mod gradient_descent;

use std::clone::Clone;
use std::fmt::Debug;

use crate::math::num::FloatExt;
use crate::linalg::Matrix;
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{F, DF};

pub trait FirstOrderOptimizer<T: FloatExt + Debug> {
    fn optimize<'a, X: Matrix<T>, LS: LineSearchMethod<T>>(&self, f: &F<T, X>, df: &'a DF<X>, x0: &X, ls: &'a LS) -> OptimizerResult<T, X>;    
}

#[derive(Debug, Clone)]
pub struct OptimizerResult<T: FloatExt + Debug, X: Matrix<T>> 
{
    pub x: X,
    pub f_x: T,
    pub iterations: usize
}