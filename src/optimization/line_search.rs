use crate::math::EPSILON;
use crate::optimization::FunctionOrder;

pub trait LineSearchMethod {
    fn search<'a>(&self, f: &(dyn Fn(f64) -> f64), df: &(dyn Fn(f64) -> f64), alpha: f64, f0: f64, df0: f64) -> LineSearchResult;
}

#[derive(Debug, Clone)]
pub struct LineSearchResult {
    pub alpha: f64,
    pub f_x: f64
}

pub struct Backtracking {
    pub c1: f64,
    pub max_iterations: usize,
    pub max_infinity_iterations: usize,
    pub phi: f64,
    pub plo: f64,
    pub order: FunctionOrder
}

impl Default for Backtracking {
    fn default() -> Self { 
        Backtracking {
            c1: 1e-4,
            max_iterations: 1000,
            max_infinity_iterations: -EPSILON.log2() as usize,
            phi: 0.5,
            plo: 0.1,
            order: FunctionOrder::SECOND
        }
     }
}

impl LineSearchMethod for Backtracking {
    
    fn search<'a>(&self, f: &(dyn Fn(f64) -> f64), _: &(dyn Fn(f64) -> f64), alpha: f64, f0: f64, df0: f64) -> LineSearchResult {        

        let (mut a1, mut a2) = (alpha, alpha);
        let (mut fx0, mut fx1) = (f0, f(a1));             

        let mut iterfinite = 0;
        while !fx1.is_finite() && iterfinite < self.max_infinity_iterations {
            iterfinite += 1;
            a1 = a2;
            a2 = a1 / 2.;

            fx1 = f(a2);
        }

        let mut iteration = 0;        

        while fx1 > f0 + self.c1 * a2 * df0 {
            if iteration > self.max_iterations {
                panic!("Linesearch failed to converge, reached maximum iterations.");
            }

            let a_tmp;

            if self.order == FunctionOrder::SECOND || iteration == 0 {

                a_tmp = - (df0 * a2.powf(2.)) / (2. * (fx1 - f0 - df0*a2))
                
            } else {

                let div = 1. / (a1.powf(2.) * a2.powf(2.) * (a2 - a1));
                let a = (a1.powf(2.) * (fx1 - f0 - df0*a2) - a2.powf(2.)*(fx0 - f0 - df0*a1))*div;
                let b = (-a1.powf(3.) * (fx1 - f0 - df0*a2) + a2.powf(3.)*(fx0 - f0 - df0*a1))*div;                

                if (a - 0.).powf(2.).sqrt() <= EPSILON {
                    a_tmp = df0 / (2. * b);
                } else {
                    let d = f64::max(b.powf(2.) - 3. * a * df0, 0.);
                    a_tmp = (-b + d.sqrt()) / (3.*a); //root of quadratic equation
                }
            }

            a1 = a2;            
            a2 = f64::max(f64::min(a_tmp, a2*self.phi), a2*self.plo);

            fx0 = fx1;
            fx1 = f(a2);                      

            iteration += 1;
        }

        LineSearchResult {
            alpha: a2,
            f_x: fx1
        }

    }
}

#[cfg(test)]
mod tests {    
    use super::*;       

    #[test]
    fn backtracking() { 
            
            let f = |x: f64| -> f64 {                
                x.powf(2.) + x
            };

            let df = |x: f64| -> f64 {                                         
                2. * x + 1.
            };

            let ls: Backtracking = Default::default();     

            let mut x = -3.;
            let mut alpha = 1.;

            for _ in 0..10 {            
                let result = ls.search(&f, &df, alpha, f(x), df(x));                   
                alpha = result.alpha;
                x += alpha;                
            }        

            assert!(f(x).abs() < 0.01);
    }
}