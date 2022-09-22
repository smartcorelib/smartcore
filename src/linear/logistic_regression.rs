//! # Logistic Regression
//!
//! As [Linear Regression](../linear_regression/index.html), logistic regression explains your outcome as a linear combination of predictor variables \\(X\\) but rather than modeling this response directly,
//! logistic regression models the probability that \\(y\\) belongs to a particular category, \\(Pr(y = 1|X) \\), as:
//!
//! \\[ Pr(y=1) \approx \frac{e^{\beta_0 + \sum_{i=1}^n \beta_iX_i}}{1 + e^{\beta_0 + \sum_{i=1}^n \beta_iX_i}} \\]
//!
//! SmartCore uses [limited memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) method to find estimates of regression coefficients, \\(\beta\\)
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
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
//! let y: Vec<f64> = vec![
//!           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::optimization::first_order::lbfgs::LBFGS;
use crate::optimization::first_order::{FirstOrderOptimizer, OptimizerResult};
use crate::optimization::line_search::Backtracking;
use crate::optimization::FunctionOrder;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq)]
/// Solver options for Logistic regression. Right now only LBFGS solver is supported.
pub enum LogisticRegressionSolverName {
    /// Limited-memory Broyden–Fletcher–Goldfarb–Shanno method, see [LBFGS paper](http://users.iems.northwestern.edu/~nocedal/lbfgsb.html)
    LBFGS,
}

impl Default for LogisticRegressionSolverName {
    fn default() -> Self {
        LogisticRegressionSolverName::LBFGS
    }
}

/// Logistic Regression parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LogisticRegressionParameters<T: RealNumber> {
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
pub struct LogisticRegressionSearchParameters<T: RealNumber> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Solver to use for estimation of regression coefficients.
    pub solver: Vec<LogisticRegressionSolverName>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Regularization parameter.
    pub alpha: Vec<T>,
}

/// Logistic Regression grid search iterator
pub struct LogisticRegressionSearchParametersIterator<T: RealNumber> {
    logistic_regression_search_parameters: LogisticRegressionSearchParameters<T>,
    current_solver: usize,
    current_alpha: usize,
}

impl<T: RealNumber> IntoIterator for LogisticRegressionSearchParameters<T> {
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

impl<T: RealNumber> Iterator for LogisticRegressionSearchParametersIterator<T> {
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

impl<T: RealNumber> Default for LogisticRegressionSearchParameters<T> {
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
pub struct LogisticRegression<T: RealNumber, M: Matrix<T>> {
    coefficients: M,
    intercept: M,
    classes: Vec<T>,
    num_attributes: usize,
    num_classes: usize,
}

trait ObjectiveFunction<T: RealNumber, M: Matrix<T>> {
    fn f(&self, w_bias: &M) -> T;
    fn df(&self, g: &mut M, w_bias: &M);

    fn partial_dot(w: &M, x: &M, v_col: usize, m_row: usize) -> T {
        let mut sum = T::zero();
        let p = x.shape().1;
        for i in 0..p {
            sum += x.get(m_row, i) * w.get(0, i + v_col);
        }

        sum + w.get(0, p + v_col)
    }
}

struct BinaryObjectiveFunction<'a, T: RealNumber, M: Matrix<T>> {
    x: &'a M,
    y: Vec<usize>,
    alpha: T,
}

impl<T: RealNumber> LogisticRegressionParameters<T> {
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

impl<T: RealNumber> Default for LogisticRegressionParameters<T> {
    fn default() -> Self {
        LogisticRegressionParameters {
            solver: LogisticRegressionSolverName::default(),
            alpha: T::zero(),
        }
    }
}

impl<T: RealNumber, M: Matrix<T>> PartialEq for LogisticRegression<T, M> {
    fn eq(&self, other: &Self) -> bool {
        if self.num_classes != other.num_classes
            || self.num_attributes != other.num_attributes
            || self.classes.len() != other.classes.len()
        {
            false
        } else {
            for i in 0..self.classes.len() {
                if (self.classes[i] - other.classes[i]).abs() > T::epsilon() {
                    return false;
                }
            }

            self.coefficients == other.coefficients && self.intercept == other.intercept
        }
    }
}

impl<'a, T: RealNumber, M: Matrix<T>> ObjectiveFunction<T, M>
    for BinaryObjectiveFunction<'a, T, M>
{
    fn f(&self, w_bias: &M) -> T {
        let mut f = T::zero();
        let (n, p) = self.x.shape();

        for i in 0..n {
            let wx = BinaryObjectiveFunction::partial_dot(w_bias, self.x, 0, i);
            f += wx.ln_1pe() - (T::from(self.y[i]).unwrap()) * wx;
        }

        if self.alpha > T::zero() {
            let mut w_squared = T::zero();
            for i in 0..p {
                let w = w_bias.get(0, i);
                w_squared += w * w;
            }
            f += T::half() * self.alpha * w_squared;
        }

        f
    }

    fn df(&self, g: &mut M, w_bias: &M) {
        g.copy_from(&M::zeros(1, g.shape().1));

        let (n, p) = self.x.shape();

        for i in 0..n {
            let wx = BinaryObjectiveFunction::partial_dot(w_bias, self.x, 0, i);

            let dyi = (T::from(self.y[i]).unwrap()) - wx.sigmoid();
            for j in 0..p {
                g.set(0, j, g.get(0, j) - dyi * self.x.get(i, j));
            }
            g.set(0, p, g.get(0, p) - dyi);
        }

        if self.alpha > T::zero() {
            for i in 0..p {
                let w = w_bias.get(0, i);
                g.set(0, i, g.get(0, i) + self.alpha * w);
            }
        }
    }
}

struct MultiClassObjectiveFunction<'a, T: RealNumber, M: Matrix<T>> {
    x: &'a M,
    y: Vec<usize>,
    k: usize,
    alpha: T,
}

impl<'a, T: RealNumber, M: Matrix<T>> ObjectiveFunction<T, M>
    for MultiClassObjectiveFunction<'a, T, M>
{
    fn f(&self, w_bias: &M) -> T {
        let mut f = T::zero();
        let mut prob = M::zeros(1, self.k);
        let (n, p) = self.x.shape();
        for i in 0..n {
            for j in 0..self.k {
                prob.set(
                    0,
                    j,
                    MultiClassObjectiveFunction::partial_dot(w_bias, self.x, j * (p + 1), i),
                );
            }
            prob.softmax_mut();
            f -= prob.get(0, self.y[i]).ln();
        }

        if self.alpha > T::zero() {
            let mut w_squared = T::zero();
            for i in 0..self.k {
                for j in 0..p {
                    let wi = w_bias.get(0, i * (p + 1) + j);
                    w_squared += wi * wi;
                }
            }
            f += T::half() * self.alpha * w_squared;
        }

        f
    }

    fn df(&self, g: &mut M, w: &M) {
        g.copy_from(&M::zeros(1, g.shape().1));

        let mut prob = M::zeros(1, self.k);
        let (n, p) = self.x.shape();

        for i in 0..n {
            for j in 0..self.k {
                prob.set(
                    0,
                    j,
                    MultiClassObjectiveFunction::partial_dot(w, self.x, j * (p + 1), i),
                );
            }

            prob.softmax_mut();

            for j in 0..self.k {
                let yi = (if self.y[i] == j { T::one() } else { T::zero() }) - prob.get(0, j);

                for l in 0..p {
                    let pos = j * (p + 1);
                    g.set(0, pos + l, g.get(0, pos + l) - yi * self.x.get(i, l));
                }
                g.set(0, j * (p + 1) + p, g.get(0, j * (p + 1) + p) - yi);
            }
        }

        if self.alpha > T::zero() {
            for i in 0..self.k {
                for j in 0..p {
                    let pos = i * (p + 1);
                    let wi = w.get(0, pos + j);
                    g.set(0, pos + j, g.get(0, pos + j) + self.alpha * wi);
                }
            }
        }
    }
}

impl<T: RealNumber, M: Matrix<T>>
    SupervisedEstimator<M, M::RowVector, LogisticRegressionParameters<T>>
    for LogisticRegression<T, M>
{
    fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: LogisticRegressionParameters<T>,
    ) -> Result<Self, Failed> {
        LogisticRegression::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>> Predictor<M, M::RowVector> for LogisticRegression<T, M> {
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber, M: Matrix<T>> LogisticRegression<T, M> {
    /// Fits Logistic Regression to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target class values
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.    
    pub fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: LogisticRegressionParameters<T>,
    ) -> Result<LogisticRegression<T, M>, Failed> {
        let y_m = M::from_row_vector(y.clone());
        let (x_nrows, num_attributes) = x.shape();
        let (_, y_nrows) = y_m.shape();

        if x_nrows != y_nrows {
            return Err(Failed::fit(
                "Number of rows of X doesn\'t match number of rows of Y",
            ));
        }

        let classes = y_m.unique();

        let k = classes.len();

        let mut yi: Vec<usize> = vec![0; y_nrows];

        for (i, yi_i) in yi.iter_mut().enumerate().take(y_nrows) {
            let yc = y_m.get(0, i);
            *yi_i = classes.iter().position(|c| yc == *c).unwrap();
        }

        match k.cmp(&2) {
            Ordering::Less => Err(Failed::fit(&format!(
                "incorrect number of classes: {}. Should be >= 2.",
                k
            ))),
            Ordering::Equal => {
                let x0 = M::zeros(1, num_attributes + 1);

                let objective = BinaryObjectiveFunction {
                    x,
                    y: yi,
                    alpha: parameters.alpha,
                };

                let result = LogisticRegression::minimize(x0, objective);

                let weights = result.x;

                Ok(LogisticRegression {
                    coefficients: weights.slice(0..1, 0..num_attributes),
                    intercept: weights.slice(0..1, num_attributes..num_attributes + 1),
                    classes,
                    num_attributes,
                    num_classes: k,
                })
            }
            Ordering::Greater => {
                let x0 = M::zeros(1, (num_attributes + 1) * k);

                let objective = MultiClassObjectiveFunction {
                    x,
                    y: yi,
                    k,
                    alpha: parameters.alpha,
                };

                let result = LogisticRegression::minimize(x0, objective);
                let weights = result.x.reshape(k, num_attributes + 1);

                Ok(LogisticRegression {
                    coefficients: weights.slice(0..k, 0..num_attributes),
                    intercept: weights.slice(0..k, num_attributes..num_attributes + 1),
                    classes,
                    num_attributes,
                    num_classes: k,
                })
            }
        }
    }

    /// Predict class labels for samples in `x`.
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        let n = x.shape().0;
        let mut result = M::zeros(1, n);
        if self.num_classes == 2 {
            let y_hat: Vec<T> = x.ab(false, &self.coefficients, true).get_col_as_vec(0);
            let intercept = self.intercept.get(0, 0);
            for (i, y_hat_i) in y_hat.iter().enumerate().take(n) {
                result.set(
                    0,
                    i,
                    self.classes[if (*y_hat_i + intercept).sigmoid() > T::half() {
                        1
                    } else {
                        0
                    }],
                );
            }
        } else {
            let mut y_hat = x.matmul(&self.coefficients.transpose());
            for r in 0..n {
                for c in 0..self.num_classes {
                    y_hat.set(r, c, y_hat.get(r, c) + self.intercept.get(c, 0));
                }
            }
            let class_idxs = y_hat.argmax();
            for (i, class_i) in class_idxs.iter().enumerate().take(n) {
                result.set(0, i, self.classes[*class_i]);
            }
        }
        Ok(result.to_row_vector())
    }

    /// Get estimates regression coefficients
    pub fn coefficients(&self) -> &M {
        &self.coefficients
    }

    /// Get estimate of intercept
    pub fn intercept(&self) -> &M {
        &self.intercept
    }

    fn minimize(x0: M, objective: impl ObjectiveFunction<T, M>) -> OptimizerResult<T, M> {
        let f = |w: &M| -> T { objective.f(w) };

        let df = |g: &mut M, w: &M| objective.df(g, w);

        let ls: Backtracking<T> = Backtracking {
            order: FunctionOrder::THIRD,
            ..Default::default()
        };
        let optimizer: LBFGS<T> = Default::default();

        optimizer.optimize(&f, &df, &x0, &ls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::generator::make_blobs;
    use crate::linalg::naive::dense_matrix::*;
    use crate::metrics::accuracy;

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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        };

        let mut g: DenseMatrix<f64> = DenseMatrix::zeros(1, 9);

        objective.df(
            &mut g,
            &DenseMatrix::row_vector_from_array(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );
        objective.df(
            &mut g,
            &DenseMatrix::row_vector_from_array(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );

        assert!((g.get(0, 0) + 33.000068218163484).abs() < std::f64::EPSILON);

        let f = objective.f(&DenseMatrix::row_vector_from_array(&[
            1., 2., 3., 4., 5., 6., 7., 8., 9.,
        ]));

        assert!((f - 408.0052230582765).abs() < std::f64::EPSILON);

        let objective_reg = MultiClassObjectiveFunction {
            x: &x,
            y: y.clone(),
            k: 3,
            alpha: 1.0,
        };

        let f = objective_reg.f(&DenseMatrix::row_vector_from_array(&[
            1., 2., 3., 4., 5., 6., 7., 8., 9.,
        ]));
        assert!((f - 487.5052).abs() < 1e-4);

        objective_reg.df(
            &mut g,
            &DenseMatrix::row_vector_from_array(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );
        assert!((g.get(0, 0).abs() - 32.0).abs() < 1e-4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        };

        let mut g: DenseMatrix<f64> = DenseMatrix::zeros(1, 3);

        objective.df(&mut g, &DenseMatrix::row_vector_from_array(&[1., 2., 3.]));
        objective.df(&mut g, &DenseMatrix::row_vector_from_array(&[1., 2., 3.]));

        assert!((g.get(0, 0) - 26.051064349381285).abs() < std::f64::EPSILON);
        assert!((g.get(0, 1) - 10.239000702928523).abs() < std::f64::EPSILON);
        assert!((g.get(0, 2) - 3.869294270156324).abs() < std::f64::EPSILON);

        let f = objective.f(&DenseMatrix::row_vector_from_array(&[1., 2., 3.]));

        assert!((f - 59.76994756647412).abs() < std::f64::EPSILON);

        let objective_reg = BinaryObjectiveFunction {
            x: &x,
            y: y.clone(),
            alpha: 1.0,
        };

        let f = objective_reg.f(&DenseMatrix::row_vector_from_array(&[1., 2., 3.]));
        assert!((f - 62.2699).abs() < 1e-4);

        objective_reg.df(&mut g, &DenseMatrix::row_vector_from_array(&[1., 2., 3.]));
        assert!((g.get(0, 0) - 27.0511).abs() < 1e-4);
        assert!((g.get(0, 1) - 12.239).abs() < 1e-4);
        assert!((g.get(0, 2) - 3.8693).abs() < 1e-4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lr_fit_predict() {
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
        let y: Vec<f64> = vec![0., 0., 1., 1., 2., 1., 1., 0., 0., 2., 1., 1., 0., 0., 1.];

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

        assert_eq!(lr.coefficients().shape(), (3, 2));
        assert_eq!(lr.intercept().shape(), (3, 1));

        assert!((lr.coefficients().get(0, 0) - 0.0435).abs() < 1e-4);
        assert!((lr.intercept().get(0, 0) - 0.1250).abs() < 1e-4);

        let y_hat = lr.predict(&x).unwrap();

        assert_eq!(
            y_hat,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lr_fit_predict_multiclass() {
        let blobs = make_blobs(15, 4, 3);

        let x = DenseMatrix::from_vec(15, 4, &blobs.data);
        let y = blobs.target;

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

        let y_hat = lr.predict(&x).unwrap();

        assert!(accuracy(&y_hat, &y) > 0.9);

        let lr_reg = LogisticRegression::fit(
            &x,
            &y,
            LogisticRegressionParameters::default().with_alpha(10.0),
        )
        .unwrap();

        assert!(lr_reg.coefficients().abs().sum() < lr.coefficients().abs().sum());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lr_fit_predict_binary() {
        let blobs = make_blobs(20, 4, 2);

        let x = DenseMatrix::from_vec(20, 4, &blobs.data);
        let y = blobs.target;

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

        let y_hat = lr.predict(&x).unwrap();

        assert!(accuracy(&y_hat, &y) > 0.9);

        let lr_reg = LogisticRegression::fit(
            &x,
            &y,
            LogisticRegressionParameters::default().with_alpha(10.0),
        )
        .unwrap();

        assert!(lr_reg.coefficients().abs().sum() < lr.coefficients().abs().sum());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let y: Vec<f64> = vec![0., 0., 1., 1., 2., 1., 1., 0., 0., 2., 1., 1., 0., 0., 1.];

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();

        let deserialized_lr: LogisticRegression<f64, DenseMatrix<f64>> =
            serde_json::from_str(&serde_json::to_string(&lr).unwrap()).unwrap();

        assert_eq!(lr, deserialized_lr);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let y: Vec<f64> = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
        let lr_reg = LogisticRegression::fit(
            &x,
            &y,
            LogisticRegressionParameters::default().with_alpha(1.0),
        )
        .unwrap();

        let y_hat = lr.predict(&x).unwrap();

        let error: f64 = y
            .into_iter()
            .zip(y_hat.into_iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(error <= 1.0);
        assert!(lr_reg.coefficients().abs().sum() < lr.coefficients().abs().sum());
    }
}
