//! # Logistic Regression
//!
//! As [Linear Regression](../linear_regression/index.html), logistic regression explains your outcome as a linear combination of predictor variables \\(X\\) but rather than modeling this response directly,
//! logistic regression models the probability that \\(y\\) belongs to a particular category, \\(Pr(y = 1|X) \\), as:
//!
//! \\[ Pr(y=1) \approx \frac{e^{\beta_0 + \sum_{i=1}^n \beta_iX_i}}{1 + e^{\beta_0 + \sum_{i=1}^n \beta_iX_i}} \\]
//!
//! `smartcore` uses [limited memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) method to find estimates of regression coefficients, \\(\beta\\)
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::linear::logistic_regression::*;
//!
//! //Iris data
//! let x = DenseMatrix::from_2d_array(&[
//!           &[5.1, 3.5, 1.4, 0.2],
//!           &[4.9, 3.0, 1.4, 0.2],
//!           &[4.7, 3.2, 1.3, 0.2],
//!           &[4.6, 3.1, 1.5, 0.2],
//!           &[5.0, 3.6, 1.4, 0.2],
//!           &[5.4, 3.9, 1.7, 0.4],
//!           &[4.6, 3.4, 1.4, 0.3],
//!           &[5.0, 3.4, 1.5, 0.2],
//!           &[4.4, 2.9, 1.4, 0.2],
//!           &[4.9, 3.1, 1.5, 0.1],
//!           &[7.0, 3.2, 4.7, 1.4],
//!           &[6.4, 3.2, 4.5, 1.5],
//!           &[6.9, 3.1, 4.9, 1.5],
//!           &[5.5, 2.3, 4.0, 1.3],
//!           &[6.5, 2.8, 4.6, 1.5],
//!           &[5.7, 2.8, 4.5, 1.3],
//!           &[6.3, 3.3, 4.7, 1.6],
//!           &[4.9, 2.4, 3.3, 1.0],
//!           &[6.6, 2.9, 4.6, 1.3],
//!           &[5.2, 2.7, 3.9, 1.4],
//!           ]);
//! let y: Vec<i32> = vec![
//!           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//! ];
//!
//! let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
//!
//! let y_hat = lr.predict(&x).unwrap();
//! ```
//!
//! ## References:
//! * ["Pattern Recognition and Machine Learning", C.M. Bishop, Linear Models for Classification](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 4.3 Logistic Regression](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["On the Limited Memory Method for Large Scale Optimization", Nocedal et al., Mathematical Programming, 1989](http://users.iems.northwestern.edu/~nocedal/PDFfiles/limited.pdf)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::cmp::Ordering;
use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2, MutArrayView1};
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;
use crate::optimization::first_order::lbfgs::LBFGS;
use crate::optimization::first_order::{FirstOrderOptimizer, OptimizerResult};
use crate::optimization::line_search::Backtracking;
use crate::optimization::FunctionOrder;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq, Default)]
/// Solver options for Logistic regression. Right now only LBFGS solver is supported.
pub enum LogisticRegressionSolverName {
    /// Limited-memory Broyden–Fletcher–Goldfarb–Shanno method, see [LBFGS paper](http://users.iems.northwestern.edu/~nocedal/lbfgsb.html)
    #[default]
    LBFGS,
}

/// Logistic Regression parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LogisticRegressionParameters<T: Number + FloatNumber> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Solver to use for estimation of regression coefficients.
    pub solver: LogisticRegressionSolverName,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Regularization parameter.
    pub alpha: T,
}

/// Logistic Regression grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LogisticRegressionSearchParameters<T: Number> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Solver to use for estimation of regression coefficients.
    pub solver: Vec<LogisticRegressionSolverName>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Regularization parameter.
    pub alpha: Vec<T>,
}

/// Logistic Regression grid search iterator
pub struct LogisticRegressionSearchParametersIterator<T: Number> {
    logistic_regression_search_parameters: LogisticRegressionSearchParameters<T>,
    current_solver: usize,
    current_alpha: usize,
}

impl<T: Number + FloatNumber> IntoIterator for LogisticRegressionSearchParameters<T> {
    type Item = LogisticRegressionParameters<T>;
    type IntoIter = LogisticRegressionSearchParametersIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        LogisticRegressionSearchParametersIterator {
            logistic_regression_search_parameters: self,
            current_solver: 0,
            current_alpha: 0,
        }
    }
}

impl<T: Number + FloatNumber> Iterator for LogisticRegressionSearchParametersIterator<T> {
    type Item = LogisticRegressionParameters<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_alpha == self.logistic_regression_search_parameters.alpha.len()
            && self.current_solver == self.logistic_regression_search_parameters.solver.len()
        {
            return None;
        }

        let next = LogisticRegressionParameters {
            solver: self.logistic_regression_search_parameters.solver[self.current_solver].clone(),
            alpha: self.logistic_regression_search_parameters.alpha[self.current_alpha],
        };

        if self.current_alpha + 1 < self.logistic_regression_search_parameters.alpha.len() {
            self.current_alpha += 1;
        } else if self.current_solver + 1 < self.logistic_regression_search_parameters.solver.len()
        {
            self.current_alpha = 0;
            self.current_solver += 1;
        } else {
            self.current_alpha += 1;
            self.current_solver += 1;
        }

        Some(next)
    }
}

impl<T: Number + FloatNumber> Default for LogisticRegressionSearchParameters<T> {
    fn default() -> Self {
        let default_params = LogisticRegressionParameters::default();

        LogisticRegressionSearchParameters {
            solver: vec![default_params.solver],
            alpha: vec![default_params.alpha],
        }
    }
}

/// Logistic Regression
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct LogisticRegression<
    TX: Number + FloatNumber + RealNumber,
    TY: Number + Ord,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    coefficients: Option<X>,
    intercept: Option<X>,
    classes: Option<Vec<TY>>,
    num_attributes: usize,
    num_classes: usize,
    _phantom_tx: PhantomData<TX>,
    _phantom_y: PhantomData<Y>,
}

trait ObjectiveFunction<T: Number + FloatNumber, X: Array2<T>> {
    ///
    fn f(&self, w_bias: &[T]) -> T;

    ///
    #[allow(clippy::ptr_arg)]
    fn df(&self, g: &mut Vec<T>, w_bias: &Vec<T>);

    ///
    #[allow(clippy::ptr_arg)]
    fn partial_dot(w: &[T], x: &X, v_col: usize, m_row: usize) -> T {
        let mut sum = T::zero();
        let p = x.shape().1;
        for i in 0..p {
            sum += *x.get((m_row, i)) * w[i + v_col];
        }

        sum + w[p + v_col]
    }
}

struct BinaryObjectiveFunction<'a, T: Number + FloatNumber, X: Array2<T>> {
    x: &'a X,
    y: Vec<usize>,
    alpha: T,
    _phantom_t: PhantomData<T>,
}

impl<T: Number + FloatNumber> LogisticRegressionParameters<T> {
    /// Solver to use for estimation of regression coefficients.
    pub fn with_solver(mut self, solver: LogisticRegressionSolverName) -> Self {
        self.solver = solver;
        self
    }
    /// Regularization parameter.
    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }
}

impl<T: Number + FloatNumber> Default for LogisticRegressionParameters<T> {
    fn default() -> Self {
        LogisticRegressionParameters {
            solver: LogisticRegressionSolverName::default(),
            alpha: T::zero(),
        }
    }
}

impl<TX: Number + FloatNumber + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    PartialEq for LogisticRegression<TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        if self.num_classes != other.num_classes
            || self.num_attributes != other.num_attributes
            || self.classes().len() != other.classes().len()
        {
            false
        } else {
            for i in 0..self.classes().len() {
                if self.classes()[i] != other.classes()[i] {
                    return false;
                }
            }

            self.coefficients()
                .iterator(0)
                .zip(other.coefficients().iterator(0))
                .all(|(&a, &b)| (a - b).abs() <= TX::epsilon())
                && self
                    .intercept()
                    .iterator(0)
                    .zip(other.intercept().iterator(0))
                    .all(|(&a, &b)| (a - b).abs() <= TX::epsilon())
        }
    }
}

impl<'a, T: Number + FloatNumber, X: Array2<T>> ObjectiveFunction<T, X>
    for BinaryObjectiveFunction<'a, T, X>
{
    fn f(&self, w_bias: &[T]) -> T {
        let mut f = T::zero();
        let (n, p) = self.x.shape();

        for i in 0..n {
            let wx = BinaryObjectiveFunction::partial_dot(w_bias, self.x, 0, i);
            f += wx.ln_1pe() - (T::from(self.y[i]).unwrap()) * wx;
        }

        if self.alpha > T::zero() {
            let mut w_squared = T::zero();
            for w_bias_i in w_bias.iter().take(p) {
                w_squared += *w_bias_i * *w_bias_i;
            }
            f += T::from_f64(0.5).unwrap() * self.alpha * w_squared;
        }

        f
    }

    fn df(&self, g: &mut Vec<T>, w_bias: &Vec<T>) {
        g.copy_from(&Vec::zeros(g.len()));

        let (n, p) = self.x.shape();

        for i in 0..n {
            let wx = BinaryObjectiveFunction::partial_dot(w_bias, self.x, 0, i);

            let dyi = (T::from(self.y[i]).unwrap()) - wx.sigmoid();
            for (j, g_j) in g.iter_mut().enumerate().take(p) {
                *g_j -= dyi * *self.x.get((i, j));
            }
            g[p] -= dyi;
        }

        if self.alpha > T::zero() {
            for i in 0..p {
                let w = w_bias[i];
                g[i] += self.alpha * w;
            }
        }
    }
}

struct MultiClassObjectiveFunction<'a, T: Number + FloatNumber, X: Array2<T>> {
    x: &'a X,
    y: Vec<usize>,
    k: usize,
    alpha: T,
    _phantom_t: PhantomData<T>,
}

impl<'a, T: Number + FloatNumber + RealNumber, X: Array2<T>> ObjectiveFunction<T, X>
    for MultiClassObjectiveFunction<'a, T, X>
{
    fn f(&self, w_bias: &[T]) -> T {
        let mut f = T::zero();
        let mut prob = vec![T::zero(); self.k];
        let (n, p) = self.x.shape();
        for i in 0..n {
            for (j, prob_j) in prob.iter_mut().enumerate().take(self.k) {
                *prob_j = MultiClassObjectiveFunction::partial_dot(w_bias, self.x, j * (p + 1), i);
            }
            prob.softmax_mut();
            f -= prob[self.y[i]].ln();
        }

        if self.alpha > T::zero() {
            let mut w_squared = T::zero();
            for i in 0..self.k {
                for j in 0..p {
                    let wi = w_bias[i * (p + 1) + j];
                    w_squared += wi * wi;
                }
            }
            f += T::from_f64(0.5).unwrap() * self.alpha * w_squared;
        }

        f
    }

    fn df(&self, g: &mut Vec<T>, w: &Vec<T>) {
        g.copy_from(&Vec::zeros(g.len()));

        let mut prob = vec![T::zero(); self.k];
        let (n, p) = self.x.shape();

        for i in 0..n {
            for (j, prob_j) in prob.iter_mut().enumerate().take(self.k) {
                *prob_j = MultiClassObjectiveFunction::partial_dot(w, self.x, j * (p + 1), i);
            }

            prob.softmax_mut();

            for j in 0..self.k {
                let yi = (if self.y[i] == j { T::one() } else { T::zero() }) - prob[j];

                for l in 0..p {
                    let pos = j * (p + 1);
                    g[pos + l] -= yi * *self.x.get((i, l));
                }
                g[j * (p + 1) + p] -= yi;
            }
        }

        if self.alpha > T::zero() {
            for i in 0..self.k {
                for j in 0..p {
                    let pos = i * (p + 1);
                    let wi = w[pos + j];
                    g[pos + j] += self.alpha * wi;
                }
            }
        }
    }
}

impl<TX: Number + FloatNumber + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimator<X, Y, LogisticRegressionParameters<TX>>
    for LogisticRegression<TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            coefficients: Option::None,
            intercept: Option::None,
            classes: Option::None,
            num_attributes: 0,
            num_classes: 0,
            _phantom_tx: PhantomData,
            _phantom_y: PhantomData,
        }
    }

    fn fit(x: &X, y: &Y, parameters: LogisticRegressionParameters<TX>) -> Result<Self, Failed> {
        LogisticRegression::fit(x, y, parameters)
    }
}

impl<TX: Number + FloatNumber + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    Predictor<X, Y> for LogisticRegression<TX, TY, X, Y>
{
    fn predict(&self, x: &X) -> Result<Y, Failed> {
        self.predict(x)
    }
}

impl<TX: Number + FloatNumber + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    LogisticRegression<TX, TY, X, Y>
{
    /// Fits Logistic Regression to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target class values
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.    
    pub fn fit(
        x: &X,
        y: &Y,
        parameters: LogisticRegressionParameters<TX>,
    ) -> Result<LogisticRegression<TX, TY, X, Y>, Failed> {
        let (x_nrows, num_attributes) = x.shape();
        let y_nrows = y.shape();

        if x_nrows != y_nrows {
            return Err(Failed::fit(
                "Number of rows of X doesn\'t match number of rows of Y",
            ));
        }

        let classes = y.unique();

        let k = classes.len();

        let mut yi: Vec<usize> = vec![0; y_nrows];

        for (i, yi_i) in yi.iter_mut().enumerate().take(y_nrows) {
            let yc = y.get(i);
            *yi_i = classes.iter().position(|c| yc == c).unwrap();
        }

        match k.cmp(&2) {
            Ordering::Less => Err(Failed::fit(&format!(
                "incorrect number of classes: {k}. Should be >= 2."
            ))),
            Ordering::Equal => {
                let x0 = Vec::zeros(num_attributes + 1);

                let objective = BinaryObjectiveFunction {
                    x,
                    y: yi,
                    alpha: parameters.alpha,
                    _phantom_t: PhantomData,
                };

                let result = Self::minimize(x0, objective);

                let weights = X::from_iterator(result.x.into_iter(), 1, num_attributes + 1, 0);
                let coefficients = weights.slice(0..1, 0..num_attributes);
                let intercept = weights.slice(0..1, num_attributes..num_attributes + 1);

                Ok(LogisticRegression {
                    coefficients: Some(X::from_slice(coefficients.as_ref())),
                    intercept: Some(X::from_slice(intercept.as_ref())),
                    classes: Some(classes),
                    num_attributes,
                    num_classes: k,
                    _phantom_tx: PhantomData,
                    _phantom_y: PhantomData,
                })
            }
            Ordering::Greater => {
                let x0 = Vec::zeros((num_attributes + 1) * k);

                let objective = MultiClassObjectiveFunction {
                    x,
                    y: yi,
                    k,
                    alpha: parameters.alpha,
                    _phantom_t: PhantomData,
                };

                let result = Self::minimize(x0, objective);
                let weights = X::from_iterator(result.x.into_iter(), k, num_attributes + 1, 0);
                let coefficients = weights.slice(0..k, 0..num_attributes);
                let intercept = weights.slice(0..k, num_attributes..num_attributes + 1);

                Ok(LogisticRegression {
                    coefficients: Some(X::from_slice(coefficients.as_ref())),
                    intercept: Some(X::from_slice(intercept.as_ref())),
                    classes: Some(classes),
                    num_attributes,
                    num_classes: k,
                    _phantom_tx: PhantomData,
                    _phantom_y: PhantomData,
                })
            }
        }
    }

    /// Predict class labels for samples in `x`.
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        let n = x.shape().0;
        let mut result = Y::zeros(n);
        if self.num_classes == 2 {
            let y_hat = x.ab(false, self.coefficients(), true);
            let intercept = *self.intercept().get((0, 0));
            for (i, y_hat_i) in y_hat.iterator(0).enumerate().take(n) {
                result.set(
                    i,
                    self.classes()[usize::from(
                        RealNumber::sigmoid(*y_hat_i + intercept) > RealNumber::half(),
                    )],
                );
            }
        } else {
            let mut y_hat = x.matmul(&self.coefficients().transpose());
            for r in 0..n {
                for c in 0..self.num_classes {
                    y_hat.set((r, c), *y_hat.get((r, c)) + *self.intercept().get((c, 0)));
                }
            }
            let class_idxs = y_hat.argmax(1);
            for (i, class_i) in class_idxs.iter().enumerate().take(n) {
                result.set(i, self.classes()[*class_i]);
            }
        }
        Ok(result)
    }

    /// Get estimates regression coefficients, this create a sharable reference
    pub fn coefficients(&self) -> &X {
        self.coefficients.as_ref().unwrap()
    }

    /// Get estimate of intercept, this create a sharable reference
    pub fn intercept(&self) -> &X {
        self.intercept.as_ref().unwrap()
    }

    /// Get classes, this create a sharable reference
    pub fn classes(&self) -> &Vec<TY> {
        self.classes.as_ref().unwrap()
    }

    fn minimize(
        x0: Vec<TX>,
        objective: impl ObjectiveFunction<TX, X>,
    ) -> OptimizerResult<TX, Vec<TX>> {
        let f = |w: &Vec<TX>| -> TX { objective.f(w) };

        let df = |g: &mut Vec<TX>, w: &Vec<TX>| objective.df(g, w);

        let ls: Backtracking<TX> = Backtracking {
            order: FunctionOrder::THIRD,
            ..Default::default()
        };
        let optimizer: LBFGS = Default::default();

        optimizer.optimize(&f, &df, &x0, &ls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "datasets")]
    use crate::dataset::generator::make_blobs;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn search_parameters() {
        let parameters = LogisticRegressionSearchParameters {
            alpha: vec![0., 1.],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        assert_eq!(iter.next().unwrap().alpha, 0.);
        assert_eq!(
            iter.next().unwrap().solver,
            LogisticRegressionSolverName::LBFGS
        );
        assert!(iter.next().is_none());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn multiclass_objective_f() {
        let x = DenseMatrix::from_2d_array(&[
            &[1., -5.],
            &[2., 5.],
            &[3., -2.],
            &[1., 2.],
            &[2., 0.],
            &[6., -5.],
            &[7., 5.],
            &[6., -2.],
            &[7., 2.],
            &[6., 0.],
            &[8., -5.],
            &[9., 5.],
            &[10., -2.],
            &[8., 2.],
            &[9., 0.],
        ]);

        let y = vec![0, 0, 1, 1, 2, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1];

        let objective = MultiClassObjectiveFunction {
            x: &x,
            y: y.clone(),
            k: 3,
            alpha: 0.0,
            _phantom_t: PhantomData,
        };

        let mut g = vec![0f64; 9];

        objective.df(&mut g, &vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        objective.df(&mut g, &vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        assert!((g[0] + 33.000068218163484).abs() < std::f64::EPSILON);

        let f = objective.f(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        assert!((f - 408.0052230582765).abs() < std::f64::EPSILON);

        let objective_reg = MultiClassObjectiveFunction {
            x: &x,
            y,
            k: 3,
            alpha: 1.0,
            _phantom_t: PhantomData,
        };

        let f = objective_reg.f(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        assert!((f - 487.5052).abs() < 1e-4);

        objective_reg.df(&mut g, &vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        assert!((g[0].abs() - 32.0).abs() < 1e-4);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn binary_objective_f() {
        let x = DenseMatrix::from_2d_array(&[
            &[1., -5.],
            &[2., 5.],
            &[3., -2.],
            &[1., 2.],
            &[2., 0.],
            &[6., -5.],
            &[7., 5.],
            &[6., -2.],
            &[7., 2.],
            &[6., 0.],
            &[8., -5.],
            &[9., 5.],
            &[10., -2.],
            &[8., 2.],
            &[9., 0.],
        ]);

        let y = vec![0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1];

        let objective = BinaryObjectiveFunction {
            x: &x,
            y: y.clone(),
            alpha: 0.0,
            _phantom_t: PhantomData,
        };

        let mut g = vec![0f64; 3];

        objective.df(&mut g, &vec![1., 2., 3.]);
        objective.df(&mut g, &vec![1., 2., 3.]);

        assert!((g[0] - 26.051064349381285).abs() < std::f64::EPSILON);
        assert!((g[1] - 10.239000702928523).abs() < std::f64::EPSILON);
        assert!((g[2] - 3.869294270156324).abs() < std::f64::EPSILON);

        let f = objective.f(&[1., 2., 3.]);

        assert!((f - 59.76994756647412).abs() < std::f64::EPSILON);

        let objective_reg = BinaryObjectiveFunction {
            x: &x,
            y,
            alpha: 1.0,
            _phantom_t: PhantomData,
        };

        let f = objective_reg.f(&[1., 2., 3.]);
        assert!((f - 62.2699).abs() < 1e-4);

        objective_reg.df(&mut g, &vec![1., 2., 3.]);
        assert!((g[0] - 27.0511).abs() < 1e-4);
        assert!((g[1] - 12.239).abs() < 1e-4);
        assert!((g[2] - 3.8693).abs() < 1e-4);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn lr_fit_predict() {
        let x: DenseMatrix<f64> = DenseMatrix::from_2d_array(&[
            &[1., -5.],
            &[2., 5.],
            &[3., -2.],
            &[1., 2.],
            &[2., 0.],
            &[6., -5.],
            &[7., 5.],
            &[6., -2.],
            &[7., 2.],
            &[6., 0.],
            &[8., -5.],
            &[9., 5.],
            &[10., -2.],
            &[8., 2.],
            &[9., 0.],
        ]);
        let y: Vec<i32> = vec![0, 0, 1, 1, 2, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1];

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

        assert_eq!(lr.coefficients().shape(), (3, 2));
        assert_eq!(lr.intercept().shape(), (3, 1));

        assert!((*lr.coefficients().get((0, 0)) - 0.0435).abs() < 1e-4);
        assert!(
            (*lr.intercept().get((0, 0)) - 0.1250).abs() < 1e-4,
            "expected to be least than 1e-4, got {}",
            (*lr.intercept().get((0, 0)) - 0.1250).abs()
        );

        let y_hat = lr.predict(&x).unwrap();

        assert_eq!(y_hat, vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[cfg(feature = "datasets")]
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn lr_fit_predict_multiclass() {
        let blobs = make_blobs(15, 4, 3);

        let x: DenseMatrix<f32> = DenseMatrix::from_iterator(blobs.data.into_iter(), 15, 4, 0);
        let y: Vec<i32> = blobs.target.into_iter().map(|v| v as i32).collect();

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

        let y_hat = lr.predict(&x).unwrap();

        assert_eq!(y_hat, vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);

        let lr_reg = LogisticRegression::fit(
            &x,
            &y,
            LogisticRegressionParameters::default().with_alpha(10.0),
        )
        .unwrap();

        let reg_coeff_sum: f32 = lr_reg.coefficients().abs().iter().sum();
        let coeff: f32 = lr.coefficients().abs().iter().sum();

        assert!(reg_coeff_sum < coeff);
    }

    #[cfg(feature = "datasets")]
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn lr_fit_predict_binary() {
        let blobs = make_blobs(20, 4, 2);

        let x = DenseMatrix::from_iterator(blobs.data.into_iter(), 20, 4, 0);
        let y: Vec<i32> = blobs.target.into_iter().map(|v| v as i32).collect();

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

        let y_hat = lr.predict(&x).unwrap();

        assert_eq!(
            y_hat,
            vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        );

        let lr_reg = LogisticRegression::fit(
            &x,
            &y,
            LogisticRegressionParameters::default().with_alpha(10.0),
        )
        .unwrap();

        let reg_coeff_sum: f32 = lr_reg.coefficients().abs().iter().sum();
        let coeff: f32 = lr.coefficients().abs().iter().sum();

        assert!(reg_coeff_sum < coeff);
    }

    //TODO: serialization for the new DenseMatrix needs to be implemented
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let x = DenseMatrix::from_2d_array(&[
            &[1., -5.],
            &[2., 5.],
            &[3., -2.],
            &[1., 2.],
            &[2., 0.],
            &[6., -5.],
            &[7., 5.],
            &[6., -2.],
            &[7., 2.],
            &[6., 0.],
            &[8., -5.],
            &[9., 5.],
            &[10., -2.],
            &[8., 2.],
            &[9., 0.],
        ]);
        let y: Vec<i32> = vec![0, 0, 1, 1, 2, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1];

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

        let deserialized_lr: LogisticRegression<f64, i32, DenseMatrix<f64>, Vec<i32>> =
            serde_json::from_str(&serde_json::to_string(&lr).unwrap()).unwrap();

        assert_eq!(lr, deserialized_lr);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn lr_fit_predict_iris() {
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[5.7, 2.8, 4.5, 1.3],
            &[6.3, 3.3, 4.7, 1.6],
            &[4.9, 2.4, 3.3, 1.0],
            &[6.6, 2.9, 4.6, 1.3],
            &[5.2, 2.7, 3.9, 1.4],
        ]);
        let y: Vec<i32> = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
        let lr_reg = LogisticRegression::fit(
            &x,
            &y,
            LogisticRegressionParameters::default().with_alpha(1.0),
        )
        .unwrap();

        let y_hat = lr.predict(&x).unwrap();

        let error: i32 = y
            .into_iter()
            .zip(y_hat.into_iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(error <= 1);

        let reg_coeff_sum: f32 = lr_reg.coefficients().abs().iter().sum();
        let coeff: f32 = lr.coefficients().abs().iter().sum();

        assert!(reg_coeff_sum < coeff);
    }
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn lr_fit_predict_random() {
        let x: DenseMatrix<f32> = DenseMatrix::rand(52181, 94);
        let y1: Vec<i32> = vec![1; 2181];
        let y2: Vec<i32> = vec![0; 50000];
        let y: Vec<i32> = y1.into_iter().chain(y2.into_iter()).collect();

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
        let lr_reg = LogisticRegression::fit(
            &x,
            &y,
            LogisticRegressionParameters::default().with_alpha(1.0),
        )
        .unwrap();

        let y_hat = lr.predict(&x).unwrap();
        let y_hat_reg = lr_reg.predict(&x).unwrap();

        assert_eq!(y.len(), y_hat.len());
        assert_eq!(y.len(), y_hat_reg.len());
    }

    #[test]
    fn test_logit() {
        let x: &DenseMatrix<f64> = &DenseMatrix::rand(52181, 94);
        let y1: Vec<u32> = vec![1; 2181];
        let y2: Vec<u32> = vec![0; 50000];
        let y: &Vec<u32> = &(y1.into_iter().chain(y2.into_iter()).collect());
        println!("y vec height: {:?}", y.len());
        println!("x matrix shape: {:?}", x.shape());
     
	    let lr = LogisticRegression::fit(x, y, Default::default()).unwrap();
        let y_hat = lr.predict(&x).unwrap();

        println!("y_hat shape: {:?}", y_hat.shape());

        assert_eq!(y_hat.shape(), 52181);

    }
}
