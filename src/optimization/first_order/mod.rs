pub mod lbfgs;
pub mod gradient_descent;
use crate::linalg::Matrix;
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{F, DF};

pub trait FirstOrderOptimizer {
    fn optimize<'a, X: Matrix, LS: LineSearchMethod>(&self, f: &F<X>, df: &'a DF<X>, x0: &X, ls: &'a LS) -> OptimizerResult<X>;    
}

#[derive(Debug, Clone)]
pub struct OptimizerResult<X> 
where X: Matrix
{
    pub x: X,
    pub f_x: f64,
    pub iterations: usize
}