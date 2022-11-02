#![allow(clippy::suspicious_operation_groupings)]

// TODO: Add documentation
use std::default::Default;
use std::fmt::Debug;

use crate::linalg::basic::arrays::Array1;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;
use crate::optimization::first_order::{FirstOrderOptimizer, OptimizerResult};
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{DF, F};

///
pub struct LBFGS {
    ///
    pub max_iter: usize,
    ///
    pub g_rtol: f64,
    ///
    pub g_atol: f64,
    ///
    pub x_atol: f64,
    ///
    pub x_rtol: f64,
    ///
    pub f_abstol: f64,
    ///
    pub f_reltol: f64,
    ///
    pub successive_f_tol: usize,
    ///
    pub m: usize,
}

///
impl Default for LBFGS {
    ///
    fn default() -> Self {
        LBFGS {
            max_iter: 1000,
            g_rtol: 1e-8,
            g_atol: 1e-8,
            x_atol: 0f64,
            x_rtol: 0f64,
            f_abstol: 0f64,
            f_reltol: 0f64,
            successive_f_tol: 1,
            m: 10,
        }
    }
}

///
impl LBFGS {
    ///
    fn two_loops<T: FloatNumber + RealNumber, X: Array1<T>>(&self, state: &mut LBFGSState<T, X>) {
        let lower = state.iteration.max(self.m) - self.m;
        let upper = state.iteration;

        state.twoloop_q.copy_from(&state.x_df);

        for index in (lower..upper).rev() {
            let i = index.rem_euclid(self.m);
            let dgi = &state.dg_history[i];
            let dxi = &state.dx_history[i];
            state.twoloop_alpha[i] = state.rho[i] * dxi.dot(&state.twoloop_q);
            state
                .twoloop_q
                .sub_mut(&dgi.mul_scalar(state.twoloop_alpha[i]));
        }

        if state.iteration > 0 {
            let i = (upper - 1).rem_euclid(self.m);
            let dxi = &state.dx_history[i];
            let dgi = &state.dg_history[i];
            let mut div = dgi.abs();
            div.pow_mut(RealNumber::two());
            let scaling = dxi.dot(dgi) / div.sum();
            state.s.copy_from(&state.twoloop_q.mul_scalar(scaling));
        } else {
            state.s.copy_from(&state.twoloop_q);
        }

        for index in lower..upper {
            let i = index.rem_euclid(self.m);
            let dgi = &state.dg_history[i];
            let dxi = &state.dx_history[i];
            let beta = state.rho[i] * dgi.dot(&state.s);
            state
                .s
                .add_mut(&dxi.mul_scalar(state.twoloop_alpha[i] - beta));
        }

        state.s.mul_scalar_mut(-T::one());
    }

    ///
    fn init_state<T: FloatNumber + RealNumber, X: Array1<T>>(&self, x: &X) -> LBFGSState<T, X> {
        LBFGSState {
            x: x.clone(),
            x_prev: x.clone(),
            x_f: T::nan(),
            x_f_prev: T::nan(),
            x_df: x.clone(),
            x_df_prev: x.clone(),
            rho: vec![T::zero(); self.m],
            dx_history: vec![x.clone(); self.m],
            dg_history: vec![x.clone(); self.m],
            dx: x.clone(),
            dg: x.clone(),

            twoloop_q: x.clone(),
            twoloop_alpha: vec![T::zero(); self.m],
            iteration: 0,
            counter_f_tol: 0,
            s: x.clone(),
            alpha: T::one(),
        }
    }

    ///
    fn update_state<'a, T: FloatNumber + RealNumber, X: Array1<T>, LS: LineSearchMethod<T>>(
        &self,
        f: &'a F<'_, T, X>,
        df: &'a DF<'_, X>,
        ls: &'a LS,
        state: &mut LBFGSState<T, X>,
    ) {
        self.two_loops(state);

        df(&mut state.x_df_prev, &state.x);
        state.x_f_prev = f(&state.x);
        state.x_prev.copy_from(&state.x);

        let df0 = state.x_df.dot(&state.s);

        let f_alpha = |alpha: T| -> T {
            let mut dx = state.s.clone();
            dx.mul_scalar_mut(alpha);
            dx.add_mut(&state.x);
            f(&dx) // f(x) = f(x .+ gvec .* alpha)
        };

        let df_alpha = |alpha: T| -> T {
            let mut dx = state.s.clone();
            let mut dg = state.x_df.clone();
            dx.mul_scalar_mut(alpha);
            dx.add_mut(&state.x);
            df(&mut dg, &dx); //df(x) = df(x .+ gvec .* alpha)
            state.x_df.dot(&dg)
        };

        let ls_r = ls.search(&f_alpha, &df_alpha, T::one(), state.x_f_prev, df0);
        state.alpha = ls_r.alpha;

        state.s.mul_scalar_mut(state.alpha);
        state.dx.copy_from(&state.s);
        state.x.add_mut(&state.dx);
        state.x_f = f(&state.x);
        df(&mut state.x_df, &state.x);
    }

    ///
    fn assess_convergence<T: FloatNumber, X: Array1<T>>(
        &self,
        state: &mut LBFGSState<T, X>,
    ) -> bool {
        let (mut x_converged, mut g_converged) = (false, false);

        if state.x.max_diff(&state.x_prev) <= T::from_f64(self.x_atol).unwrap() {
            x_converged = true;
        }

        if state.x.max_diff(&state.x_prev)
            <= T::from_f64(self.x_rtol * state.x.norm(std::f64::INFINITY)).unwrap()
        {
            x_converged = true;
        }

        if (state.x_f - state.x_f_prev).abs() <= T::from_f64(self.f_abstol).unwrap() {
            state.counter_f_tol += 1;
        }

        if (state.x_f - state.x_f_prev).abs()
            <= T::from_f64(self.f_reltol).unwrap() * state.x_f.abs()
        {
            state.counter_f_tol += 1;
        }

        if state.x_df.norm(std::f64::INFINITY) <= self.g_atol {
            g_converged = true;
        }

        g_converged || x_converged || state.counter_f_tol > self.successive_f_tol
    }

    ///
    fn update_hessian<'a, T: FloatNumber, X: Array1<T>>(
        &self,
        _: &'a DF<'_, X>,
        state: &mut LBFGSState<T, X>,
    ) {
        state.dg = state.x_df.sub(&state.x_df_prev);
        let rho_iteration = T::one() / state.dx.dot(&state.dg);
        if !rho_iteration.is_infinite() {
            let idx = state.iteration.rem_euclid(self.m);
            state.dx_history[idx].copy_from(&state.dx);
            state.dg_history[idx].copy_from(&state.dg);
            state.rho[idx] = rho_iteration;
        }
    }
}

///
#[derive(Debug)]
struct LBFGSState<T: FloatNumber, X: Array1<T>> {
    x: X,
    x_prev: X,
    x_f: T,
    x_f_prev: T,
    x_df: X,
    x_df_prev: X,
    rho: Vec<T>,
    dx_history: Vec<X>,
    dg_history: Vec<X>,
    dx: X,
    dg: X,
    twoloop_q: X,
    twoloop_alpha: Vec<T>,
    iteration: usize,
    counter_f_tol: usize,
    s: X,
    alpha: T,
}

///
impl<T: FloatNumber + RealNumber> FirstOrderOptimizer<T> for LBFGS {
    ///
    fn optimize<'a, X: Array1<T>, LS: LineSearchMethod<T>>(
        &self,
        f: &F<'_, T, X>,
        df: &'a DF<'_, X>,
        x0: &X,
        ls: &'a LS,
    ) -> OptimizerResult<T, X> {
        let mut state = self.init_state(x0);

        df(&mut state.x_df, x0);

        let g_converged = state.x_df.norm(std::f64::INFINITY) < self.g_atol;
        let mut converged = g_converged;
        let stopped = false;

        while !converged && !stopped && state.iteration < self.max_iter {
            self.update_state(f, df, ls, &mut state);

            converged = self.assess_convergence(&mut state);

            if !converged {
                self.update_hessian(df, &mut state);
            }

            state.iteration += 1;
        }

        OptimizerResult {
            x: state.x,
            f_x: state.x_f,
            iterations: state.iteration,
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
    fn lbfgs() {
        let x0 = vec![0., 0.];
        let f = |x: &Vec<f64>| (1.0 - x[0]).powf(2.) + 100.0 * (x[1] - x[0].powf(2.)).powf(2.);

        let df = |g: &mut Vec<f64>, x: &Vec<f64>| {
            g[0] = -2. * (1. - x[0]) - 400. * (x[1] - x[0].powf(2.)) * x[0];
            g[1] = 200. * (x[1] - x[0].powf(2.));
        };
        let mut ls: Backtracking<f64> = Default::default();
        ls.order = FunctionOrder::THIRD;
        let optimizer: LBFGS = Default::default();

        let result = optimizer.optimize(&f, &df, &x0, &ls);

        assert!((result.f_x - 0.0).abs() < std::f64::EPSILON);
        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!((result.x[1] - 1.0).abs() < 1e-8);
        assert!(result.iterations <= 24);
    }
}
