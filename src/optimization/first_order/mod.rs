/// Gradient descent optimization algorithm
pub mod gradient_descent;
/// Limited-memory BFGS optimization algorithm
pub mod lbfgs;

use std::clone::Clone;
use std::fmt::Debug;

use crate::linalg::basic::arrays::Array1;
use crate::numbers::floatnum::FloatNumber;
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{DF, F};

/// First-order optimization is a class of algorithms that use the first derivative of a function to find optimal solutions.
pub trait FirstOrderOptimizer<T: FloatNumber> {
    /// run first order optimization
    fn optimize<'a, X: Array1<T>, LS: LineSearchMethod<T>>(
        &self,
        f: &F<'_, T, X>,
        df: &'a DF<'_, X>,
        x0: &X,
        ls: &'a LS,
    ) -> OptimizerResult<T, X>;
}

/// Result of optimization
#[derive(Debug, Clone)]
pub struct OptimizerResult<T: FloatNumber, X: Array1<T>> {
    /// Solution
    pub x: X,
    /// f(x) value
    pub f_x: T,
    /// number of iterations
    pub iterations: usize,
}
