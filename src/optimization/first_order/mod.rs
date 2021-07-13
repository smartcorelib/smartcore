pub mod gradient_descent;
pub mod lbfgs;

use std::clone::Clone;
use std::fmt::Debug;

use crate::linalg::base::Array1;
use crate::num::FloatNumber;
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{DF, F};

pub trait FirstOrderOptimizer<T: FloatNumber> {
    fn optimize<'a, X: Array1<T>, LS: LineSearchMethod<T>>(
        &self,
        f: &F<'_, T, X>,
        df: &'a DF<'_, X>,
        x0: &X,
        ls: &'a LS,
    ) -> OptimizerResult<T, X>;
}

#[derive(Debug, Clone)]
pub struct OptimizerResult<T: FloatNumber, X: Array1<T>> {
    pub x: X,
    pub f_x: T,
    pub iterations: usize,
}
