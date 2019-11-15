pub mod lbfgs;
pub mod gradient_descent;
use crate::linalg::Vector;
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{F, DF};

pub trait FirstOrderOptimizer {
    fn optimize<'a, X: Vector, LS: LineSearchMethod>(&self, f: &'a F<X>, df: &'a DF<X>, x0: &X, ls: &'a LS) -> OptimizerResult<X>;    
}

#[derive(Debug, Clone)]
pub struct OptimizerResult<X> 
where X: Vector
{
    pub x: X,
    pub f_x: f64,
    pub iterations: usize
}