use std::default::Default;

use crate::linalg::basic::arrays::Array1;
use crate::numbers::floatnum::FloatNumber;
use crate::optimization::first_order::{FirstOrderOptimizer, OptimizerResult};
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{DF, F};

/// Gradient Descent optimization algorithm
pub struct GradientDescent {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Relative tolerance for the gradient norm
    pub g_rtol: f64,
    /// Absolute tolerance for the gradient norm
    pub g_atol: f64,
}

impl Default for GradientDescent {
    fn default() -> Self {
        GradientDescent {
            max_iter: 10000,
            g_rtol: f64::EPSILON.sqrt(),
            g_atol: f64::EPSILON,
        }
    }
}

/// Panic with a clear message when the gradient norm is NaN.
/// Called immediately after every `df` evaluation so degenerate inputs
/// (e.g. log(0), zero-variance features) are caught before they silently
/// corrupt the optimisation state.
#[inline]
fn assert_finite_gnorm<T: FloatNumber>(gnorm: T) {
    if gnorm.is_nan() {
        panic!(
            "Gradient norm is NaN — check the objective function for \
             degenerate inputs (e.g. log(0) or a zero-variance feature)."
        );
    }
}

impl<T: FloatNumber> FirstOrderOptimizer<T> for GradientDescent {
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

        // Evaluate the initial gradient FIRST, then compute gnorm from the
        // filled gvec.  Previously gnorm was computed before df() ran, so it
        // was always 0.0 on entry and the NaN check inside the loop was
        // never reached when df immediately produced NaN.
        df(&mut gvec, &x);
        let mut gnorm = gvec.norm2();
        assert_finite_gnorm(gnorm);

        let gtol = (gnorm * self.g_rtol).max(self.g_atol);

        let mut iter = 0;
        let mut alpha = T::one();

        while iter < self.max_iter && (iter == 0 || gnorm > gtol) {
            iter += 1;

            let mut step = gvec.neg();

            let f_alpha = |alpha: T| -> T {
                let mut dx = step.clone();
                dx.mul_scalar_mut(alpha);
                dx.add_mut(&x);
                f(&dx)
            };

            let df_alpha = |alpha: T| -> T {
                let mut dx = step.clone();
                let mut dg = gvec.clone();
                dx.mul_scalar_mut(alpha);
                dx.add_mut(&x);
                df(&mut dg, &dx);
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
            // Guard after every df evaluation — catches NaN introduced at any
            // iteration, not just the first.
            assert_finite_gnorm(gnorm);
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

    #[test]
    #[should_panic(expected = "Gradient norm is NaN")]
    fn gradient_descent_nan_gradient_panics() {
        // df always writes NaN — this simulates degenerate inputs such as
        // log(0) or a zero-variance feature column.  The panic must be
        // triggered on the very first df evaluation (before the loop), so
        // the optimizer can never return a silently-corrupted result.
        let x0 = vec![1.0f64];
        let f = |_x: &Vec<f64>| 0.0f64;
        let df = |g: &mut Vec<f64>, _x: &Vec<f64>| {
            g[0] = f64::NAN;
        };

        let ls: Backtracking<f64> = Backtracking::<f64> {
            order: FunctionOrder::THIRD,
            ..Default::default()
        };
        let optimizer: GradientDescent = Default::default();
        optimizer.optimize(&f, &df, &x0, &ls);
    }
}
