#![allow(clippy::suspicious_operation_groupings)]
use std::default::Default;
use std::fmt::Debug;

use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::optimization::first_order::{FirstOrderOptimizer, OptimizerResult};
use crate::optimization::line_search::LineSearchMethod;
use crate::optimization::{DF, F};

pub struct LBFGS<T: RealNumber> {
    pub max_iter: usize,
    pub g_rtol: T,
    pub g_atol: T,
    pub x_atol: T,
    pub x_rtol: T,
    pub f_abstol: T,
    pub f_reltol: T,
    pub successive_f_tol: usize,
    pub m: usize,
}

impl<T: RealNumber> Default for LBFGS<T> {
    fn default() -> Self {
        LBFGS {
            max_iter: 1000,
            g_rtol: T::from(1e-8).unwrap(),
            g_atol: T::from(1e-8).unwrap(),
            x_atol: T::zero(),
            x_rtol: T::zero(),
            f_abstol: T::zero(),
            f_reltol: T::zero(),
            successive_f_tol: 1,
            m: 10,
        }
    }
}

impl<T: RealNumber> LBFGS<T> {
    fn two_loops<X: Matrix<T>>(&self, state: &mut LBFGSState<T, X>) {
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
            let scaling = dxi.dot(dgi) / dgi.abs().pow_mut(T::two()).sum();
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

    fn init_state<X: Matrix<T>>(&self, x: &X) -> LBFGSState<T, X> {
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

    fn update_state<'a, X: Matrix<T>, LS: LineSearchMethod<T>>(
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
            f(&dx.add_mut(&state.x)) // f(x) = f(x .+ gvec .* alpha)
        };

        let df_alpha = |alpha: T| -> T {
            let mut dx = state.s.clone();
            let mut dg = state.x_df.clone();
            dx.mul_scalar_mut(alpha);
            df(&mut dg, &dx.add_mut(&state.x)); //df(x) = df(x .+ gvec .* alpha)
            state.x_df.dot(&dg)
        };

        let ls_r = ls.search(&f_alpha, &df_alpha, T::one(), state.x_f_prev, df0);
        state.alpha = ls_r.alpha;

        state.dx.copy_from(state.s.mul_scalar_mut(state.alpha));
        state.x.add_mut(&state.dx);
        state.x_f = f(&state.x);
        df(&mut state.x_df, &state.x);
    }

    fn assess_convergence<X: Matrix<T>>(&self, state: &mut LBFGSState<T, X>) -> bool {
        let (mut x_converged, mut g_converged) = (false, false);

        if state.x.max_diff(&state.x_prev) <= self.x_atol {
            x_converged = true;
        }

        if state.x.max_diff(&state.x_prev) <= self.x_rtol * state.x.norm(T::infinity()) {
            x_converged = true;
        }

        if (state.x_f - state.x_f_prev).abs() <= self.f_abstol {
            state.counter_f_tol += 1;
        }

        if (state.x_f - state.x_f_prev).abs() <= self.f_reltol * state.x_f.abs() {
            state.counter_f_tol += 1;
        }

        if state.x_df.norm(T::infinity()) <= self.g_atol {
            g_converged = true;
        }

        g_converged || x_converged || state.counter_f_tol > self.successive_f_tol
    }

    fn update_hessian<'a, X: Matrix<T>>(&self, _: &'a DF<'_, X>, state: &mut LBFGSState<T, X>) {
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

#[derive(Debug)]
struct LBFGSState<T: RealNumber, X: Matrix<T>> {
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

impl<T: RealNumber> FirstOrderOptimizer<T> for LBFGS<T> {
    fn optimize<'a, X: Matrix<T>, LS: LineSearchMethod<T>>(
        &self,
        f: &F<'_, T, X>,
        df: &'a DF<'_, X>,
        x0: &X,
        ls: &'a LS,
    ) -> OptimizerResult<T, X> {
        let mut state = self.init_state(x0);

        df(&mut state.x_df, &x0);

        let g_converged = state.x_df.norm(T::infinity()) < self.g_atol;
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
    use crate::linalg::naive::dense_matrix::*;
    use crate::optimization::line_search::Backtracking;
    use crate::optimization::FunctionOrder;

    #[test]
    fn lbfgs() {
        let x0 = DenseMatrix::row_vector_from_array(&[0., 0.]);
        let f = |x: &DenseMatrix<f64>| {
            (1.0 - x.get(0, 0)).powf(2.) + 100.0 * (x.get(0, 1) - x.get(0, 0).powf(2.)).powf(2.)
        };

        let df = |g: &mut DenseMatrix<f64>, x: &DenseMatrix<f64>| {
            g.set(
                0,
                0,
                -2. * (1. - x.get(0, 0))
                    - 400. * (x.get(0, 1) - x.get(0, 0).powf(2.)) * x.get(0, 0),
            );
            g.set(0, 1, 200. * (x.get(0, 1) - x.get(0, 0).powf(2.)));
        };
        let mut ls: Backtracking<f64> = Default::default();
        ls.order = FunctionOrder::THIRD;
        let optimizer: LBFGS<f64> = Default::default();

        let result = optimizer.optimize(&f, &df, &x0, &ls);

        assert!((result.f_x - 0.0).abs() < std::f64::EPSILON);
        assert!((result.x.get(0, 0) - 1.0).abs() < 1e-8);
        assert!((result.x.get(0, 1) - 1.0).abs() < 1e-8);
        assert!(result.iterations <= 24);
    }
}
