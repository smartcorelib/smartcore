use std::default::Default;
use crate::math::EPSILON;
use crate::linalg::Vector;
use crate::optimization::{F, DF};
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::first_order::{FirstOrderOptimizer, OptimizerResult};

pub struct GradientDescent {
    pub max_iter: usize,
    pub g_rtol: f64,
    pub g_atol: f64
}

impl Default for GradientDescent {
    fn default() -> Self {
        GradientDescent {
            max_iter: 10000,
            g_rtol: EPSILON.sqrt(),
            g_atol: EPSILON
        }
     }
}

impl FirstOrderOptimizer for GradientDescent
{

    fn optimize<'a, X: Vector, LS: LineSearchMethod>(&self, f: &'a F<X>, df: &'a DF<X>, x0: &X, ls: &'a LS) -> OptimizerResult<X> {        

        let mut x = x0.clone();     
        let mut fx = f(&x);

        let mut gvec = x0.clone();   
        let mut gnorm = gvec.norm2();        

        let gtol = (gvec.norm2() * self.g_rtol).max(self.g_atol);        

        let mut iter = 0;
        let mut alpha = 1.0;        
        df(&mut gvec, &x);         

        while iter < self.max_iter && gnorm > gtol {
            iter += 1;
                        
            let mut step = gvec.negative();

            let f_alpha = |alpha: f64| -> f64 {
                let mut dx = step.clone();
                dx.mul_scalar_mut(alpha);
                f(&dx.add_mut(&x)) // f(x) = f(x .+ gvec .* alpha)
            };

            let df_alpha = |alpha: f64| -> f64 {                
                let mut dx = step.clone();
                let mut dg = gvec.clone();
                dx.mul_scalar_mut(alpha);
                df(&mut dg, &dx.add_mut(&x)); //df(x) = df(x .+ gvec .* alpha)
                gvec.dot(&dg)
            };

            let df0 = step.dot(&gvec);            

            let ls_r = ls.search(&f_alpha, &df_alpha, alpha, fx, df0);
            alpha = ls_r.alpha;
            fx = ls_r.f_x;
            x.add_mut(&step.mul_scalar_mut(alpha));         
            df(&mut gvec, &x);            
            gnorm = gvec.norm2();            
        }  

        let f_x = f(&x);      

        OptimizerResult{
            x: x,
            f_x: f_x,
            iterations: iter
        }
    }
}

#[cfg(test)]
mod tests {    
    use super::*; 
    use crate::linalg::naive::dense_vector::DenseVector;
    use crate::optimization::line_search::Backtracking;
    use crate::optimization::FunctionOrder;

    #[test]
    fn gradient_descent() { 

        let x0 = DenseVector::from_array(&[-1., 1.]);
        let f = |x: &DenseVector| {                
            (1.0 - x.get(0)).powf(2.) + 100.0 * (x.get(1) - x.get(0).powf(2.)).powf(2.)
        };

        let df = |g: &mut DenseVector, x: &DenseVector| {                                         
            g.set(0, -2. * (1. - x.get(0)) - 400. * (x.get(1) - x.get(0).powf(2.)) * x.get(0));
            g.set(1, 200. * (x.get(1) - x.get(0).powf(2.)));                
        };

        let mut ls: Backtracking = Default::default();
        ls.order = FunctionOrder::THIRD;
        let optimizer: GradientDescent = Default::default();
        
        let result = optimizer.optimize(&f, &df, &x0, &ls);
        
        assert!((result.f_x - 0.0).abs() < EPSILON);
        assert!((result.x.get(0) - 1.0).abs() < EPSILON);
        assert!((result.x.get(1) - 1.0).abs() < EPSILON);

    }

}