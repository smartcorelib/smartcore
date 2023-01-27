// TODO: missing documentation

use std::default::Default;

use crate::linalg::basic::arrays::Array1;
use crate::numbers::floatnum::FloatNumber;
use crate::optimization::first_order::{FirstOrderOptimizer, OptimizerResult};
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{DF, F};

///
pub struct GradientDescent {
    ///
    pub max_iter: usize,
    ///
    pub g_rtol: f64,
    ///
    pub g_atol: f64,
}

///
impl Default for GradientDescent {
    fn default() -> Self {
        GradientDescent {
            max_iter: 10000,
            g_rtol: std::f64::EPSILON.sqrt(),
            g_atol: std::f64::EPSILON,
        }
    }
}

///
impl<T: FloatNumber> FirstOrderOptimizer<T> for GradientDescent {
    ///
    fn optimize<'a, X: Array1<T>, LS: LineSearchMethod<T>>(
        &self,
        f: &'a F<'_, T, X>,
        df: &'a DF<'_, X>,
        x0: &X,
        ls: &'a LS,
    ) -> OptimizerResult<T, X> {
        let mut x = x0.clone();
        let mut fx = f(&x);

        let mut gvec = x0.clone();
        let mut gnorm = gvec.norm2();

        let gtol = (gvec.norm2() * self.g_rtol).max(self.g_atol);

        let mut iter = 0;
        let mut alpha = T::one();
        df(&mut gvec, &x);

        while iter < self.max_iter && (iter == 0 || gnorm > gtol) {
            iter += 1;

            let mut step = gvec.neg();

            let f_alpha = |alpha: T| -> T {
                let mut dx = step.clone();
                dx.mul_scalar_mut(alpha);
                dx.add_mut(&x);
                f(&dx) // f(x) = f(x .+ gvec .* alpha)
            };

            let df_alpha = |alpha: T| -> T {
                let mut dx = step.clone();
                let mut dg = gvec.clone();
                dx.mul_scalar_mut(alpha);
                dx.add_mut(&x);
                df(&mut dg, &dx); //df(x) = df(x .+ gvec .* alpha)
                gvec.dot(&dg)
            };

            let df0 = step.dot(&gvec);

            let ls_r = ls.search(&f_alpha, &df_alpha, alpha, fx, df0);
            alpha = ls_r.alpha;
            fx = ls_r.f_x;
            step.mul_scalar_mut(alpha);
            x.add_mut(&step);
            df(&mut gvec, &x);
            gnorm = gvec.norm2();
        }

        let f_x = f(&x);

        OptimizerResult {
            x,
            f_x,
            iterations: iter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimization::line_search::Backtracking;
    use crate::optimization::FunctionOrder;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn gradient_descent() {
        let x0 = vec![-1., 1.];
        let f = |x: &Vec<f64>| (1.0 - x[0]).powf(2.) + 100.0 * (x[1] - x[0].powf(2.)).powf(2.);

        let df = |g: &mut Vec<f64>, x: &Vec<f64>| {
            g[0] = -2. * (1. - x[0]) - 400. * (x[1] - x[0].powf(2.)) * x[0];
            g[1] = 200. * (x[1] - x[0].powf(2.));
        };

        let ls: Backtracking<f64> = Backtracking::<f64> {
            order: FunctionOrder::THIRD,
            ..Default::default()
        };
        let optimizer: GradientDescent = Default::default();

        let result = optimizer.optimize(&f, &df, &x0, &ls);

        assert!((result.f_x - 0.0).abs() < 1e-5);
        assert!((result.x[0] - 1.0).abs() < 1e-2);
        assert!((result.x[1] - 1.0).abs() < 1e-2);
    }
}
