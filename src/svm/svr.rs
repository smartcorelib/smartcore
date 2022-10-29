//! # Epsilon-Support Vector Regression.
//!
//! Support Vector Regression (SVR) is a popular algorithm used for regression that uses the same principle as SVM.
//!
//! Just like [SVC](../svc/index.html) SVR finds optimal decision boundary, \\(f(x)\\) that separates all training instances with the largest margin.
//! Unlike SVC, in \\(\epsilon\\)-SVR regression the goal is to find a function \\(f(x)\\) that has at most \\(\epsilon\\) deviation from the
//! known targets \\(y_i\\) for all the training data. To find this function, we need to find solution to this optimization problem:
//!
//! \\[\underset{w, \zeta}{minimize} \space \space \frac{1}{2} \lVert \vec{w} \rVert^2 + C\sum_{i=1}^m \zeta_i \\]
//!
//! subject to:
//!
//! \\[\lvert y_i - \langle\vec{w}, \vec{x}_i \rangle - b \rvert \leq \epsilon + \zeta_i \\]
//! \\[\lvert \langle\vec{w}, \vec{x}_i \rangle + b - y_i \rvert \leq \epsilon + \zeta_i \\]
//! \\[\zeta_i \geq 0 for \space any \space i = 1, ... , m\\]
//!
//! Where \\( m \\) is a number of training samples, \\( y_i \\) is a target value and \\(\langle\vec{w}, \vec{x}_i \rangle + b\\) is a decision boundary.
//!
//! The parameter `C` > 0 determines the trade-off between the flatness of \\(f(x)\\) and the amount up to which deviations larger than \\(\epsilon\\) are tolerated
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::linear::linear_regression::*;
//! use smartcore::svm::Kernels;
//! use smartcore::svm::svr::{SVR, SVRParameters};
//!
//! // Longley dataset (https://www.statsmodels.org/stable/datasets/generated/longley.html)
//! let x = DenseMatrix::from_2d_array(&[
//!               &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
//!               &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
//!               &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
//!               &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
//!               &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
//!               &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
//!               &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
//!               &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
//!               &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
//!               &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
//!               &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
//!               &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
//!               &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
//!               &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
//!               &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
//!               &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
//!          ]);
//!
//! let y: Vec<f64> = vec![83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0,
//!           100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2, 115.7, 116.9];
//! 
//! let knl = Kernels::linear();
//! let params = &SVRParameters::default().with_eps(2.0).with_c(10.0).with_kernel(&knl);
//! // let svr = SVR::fit(&x, &y, params).unwrap();
//!
//! // let y_hat = svr.predict(&x).unwrap();
//! ```
//!
//! ## References:
//!
//! * ["Support Vector Machines", Kowalczyk A., 2017](https://www.svm-tutorial.com/2017/10/support-vector-machines-succinctly-released/)
//! * ["A Fast Algorithm for Training Support Vector Machines", Platt J.C., 1998](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)
//! * ["Working Set Selection Using Second Order Information for Training Support Vector Machines", Rong-En Fan et al., 2005](https://www.jmlr.org/papers/volume6/fan05a/fan05a.pdf)
//! * ["A tutorial on support vector regression", Smola A.J., Scholkopf B., 2003](https://alex.smola.org/papers/2004/SmoSch04.pdf)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::marker::PhantomData;

use num::Bounded;
use num_traits::float::Float;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{PredictorBorrow, SupervisedEstimatorBorrow};
use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, Array2, MutArray};
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;
use crate::svm::Kernel;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// SVR Parameters
pub struct SVRParameters<'a, T: Number + RealNumber> {
    /// Epsilon in the epsilon-SVR model.
    pub eps: T,
    /// Regularization parameter.
    pub c: T,
    /// Tolerance for stopping criterion.
    pub tol: T,
    #[serde(skip_deserializing)]
    /// The kernel function.
    pub kernel: Option<&'a dyn Kernel<'a>>,
}

// /// SVR grid search parameters
// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// #[derive(Debug, Clone)]
// pub struct SVRSearchParameters<T: Number + RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
//     /// Epsilon in the epsilon-SVR model.
//     pub eps: Vec<T>,
//     /// Regularization parameter.
//     pub c: Vec<T>,
//     /// Tolerance for stopping eps.
//     pub tol: Vec<T>,
//     /// The kernel function.
//     pub kernel: Vec<K>,
//     /// Unused parameter.
//     m: PhantomData<M>,
// }

// /// SVR grid search iterator
// pub struct SVRSearchParametersIterator<T: Number + RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
//     svr_search_parameters: SVRSearchParameters<T, M, K>,
//     current_eps: usize,
//     current_c: usize,
//     current_tol: usize,
//     current_kernel: usize,
// }

// impl<T: Number + RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> IntoIterator
//     for SVRSearchParameters<T, M, K>
// {
//     type Item = SVRParameters<T, M, K>;
//     type IntoIter = SVRSearchParametersIterator<T, M, K>;

//     fn into_iter(self) -> Self::IntoIter {
//         SVRSearchParametersIterator {
//             svr_search_parameters: self,
//             current_eps: 0,
//             current_c: 0,
//             current_tol: 0,
//             current_kernel: 0,
//         }
//     }
// }

// impl<T: Number + RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Iterator
//     for SVRSearchParametersIterator<T, M, K>
// {
//     type Item = SVRParameters<T, M, K>;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.current_eps == self.svr_search_parameters.eps.len()
//             && self.current_c == self.svr_search_parameters.c.len()
//             && self.current_tol == self.svr_search_parameters.tol.len()
//             && self.current_kernel == self.svr_search_parameters.kernel.len()
//         {
//             return None;
//         }

//         let next = SVRParameters::<T, M, K> {
//             eps: self.svr_search_parameters.eps[self.current_eps],
//             c: self.svr_search_parameters.c[self.current_c],
//             tol: self.svr_search_parameters.tol[self.current_tol],
//             kernel: self.svr_search_parameters.kernel[self.current_kernel].clone(),
//             m: PhantomData,
//         };

//         if self.current_eps + 1 < self.svr_search_parameters.eps.len() {
//             self.current_eps += 1;
//         } else if self.current_c + 1 < self.svr_search_parameters.c.len() {
//             self.current_eps = 0;
//             self.current_c += 1;
//         } else if self.current_tol + 1 < self.svr_search_parameters.tol.len() {
//             self.current_eps = 0;
//             self.current_c = 0;
//             self.current_tol += 1;
//         } else if self.current_kernel + 1 < self.svr_search_parameters.kernel.len() {
//             self.current_eps = 0;
//             self.current_c = 0;
//             self.current_tol = 0;
//             self.current_kernel += 1;
//         } else {
//             self.current_eps += 1;
//             self.current_c += 1;
//             self.current_tol += 1;
//             self.current_kernel += 1;
//         }

//         Some(next)
//     }
// }

// impl<T: Number + RealNumber, M: Matrix<T>> Default for SVRSearchParameters<T, M, LinearKernel> {
//     fn default() -> Self {
//         let default_params: SVRParameters<T, M, LinearKernel> = SVRParameters::default();

//         SVRSearchParameters {
//             eps: vec![default_params.eps],
//             c: vec![default_params.c],
//             tol: vec![default_params.tol],
//             kernel: vec![default_params.kernel],
//             m: PhantomData,
//         }
//     }
// }

// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// #[derive(Debug)]
// #[cfg_attr(
//     feature = "serde",
//     serde(bound(
//         serialize = "M::RowVector: Serialize, K: Serialize, T: Serialize",
//         deserialize = "M::RowVector: Deserialize<'de>, K: Deserialize<'de>, T: Deserialize<'de>",
//     ))
// )]

/// Epsilon-Support Vector Regression
pub struct SVR<'a, T: Number + RealNumber, X: Array2<T>, Y: Array1<T>> {
    instances: Option<Vec<Vec<f64>>>,
    parameters: Option<&'a SVRParameters<'a, T>>,
    w: Option<Vec<T>>,
    b: T,
    phantom: PhantomData<(X, Y)>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
struct SupportVector<T> {
    index: usize,
    x: Vec<f64>,
    alpha: [T; 2],
    grad: [T; 2],
    k: f64,
}

/// Sequential Minimal Optimization algorithm
struct Optimizer<'a, T: Number + RealNumber> {
    tol: T,
    c: T,
    parameters: Option<&'a SVRParameters<'a, T>>,
    svmin: usize,
    svmax: usize,
    gmin: T,
    gmax: T,
    gminindex: usize,
    gmaxindex: usize,
    tau: T,
    sv: Vec<SupportVector<T>>,
}

struct Cache<T: Clone> {
    data: Vec<RefCell<Option<Vec<T>>>>,
}

impl<'a, T: Number + RealNumber> SVRParameters<'a, T> {
    /// Epsilon in the epsilon-SVR model.
    pub fn with_eps(mut self, eps: T) -> Self {
        self.eps = eps;
        self
    }
    /// Regularization parameter.
    pub fn with_c(mut self, c: T) -> Self {
        self.c = c;
        self
    }
    /// Tolerance for stopping criterion.
    pub fn with_tol(mut self, tol: T) -> Self {
        self.tol = tol;
        self
    }
    /// The kernel function.
    pub fn with_kernel(mut self, kernel: &'a (dyn Kernel<'a>)) -> Self {
        self.kernel = Some(kernel);
        self
    }
}

impl<'a, T: Number + RealNumber> Default for SVRParameters<'a, T> {
    fn default() -> Self {
        SVRParameters {
            eps: T::from_f64(0.1).unwrap(),
            c: T::one(),
            tol: T::from_f64(1e-3).unwrap(),
            kernel: Option::None,
        }
    }
}

impl<'a, T: Number + RealNumber, X: Array2<T>, Y: Array1<T>>
    SupervisedEstimatorBorrow<'a, X, Y, SVRParameters<'a, T>> for SVR<'a, T, X, Y>
{
    fn new() -> Self {
        Self {
            instances: Option::None,
            parameters: Option::None,
            w: Option::None,
            b: T::zero(),
            phantom: PhantomData,
        }
    }
    fn fit(x: &'a X, y: &'a Y, parameters: &'a SVRParameters<'a, T>) -> Result<Self, Failed> {
        SVR::fit(x, y, parameters)
    }
}

impl<'a, T: Number + RealNumber, X: Array2<T>, Y: Array1<T>> PredictorBorrow<'a, X, T>
    for SVR<'a, T, X, Y>
{
    fn predict(&self, x: &'a X) -> Result<Vec<T>, Failed> {
        self.predict(x)
    }
}

impl<'a, T: Number + RealNumber, X: Array2<T>, Y: Array1<T>> SVR<'a, T, X, Y> {
    /// Fits SVR to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target values
    /// * `kernel` - the kernel function
    /// * `parameters` - optional parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(
        x: &'a X,
        y: &'a Y,
        parameters: &'a SVRParameters<'a, T>,
    ) -> Result<SVR<'a, T, X, Y>, Failed> {
        let (n, _) = x.shape();

        if n != y.shape() {
            return Err(Failed::fit(
                "Number of rows of X doesn\'t match number of rows of Y",
            ));
        }

        if parameters.kernel.is_none() {
            return Err(Failed::because(
                FailedError::ParametersError,
                "kernel should be defined at this point, please use `with_kernel()`",
            ));
        }

        let optimizer: Optimizer<'a, T> = Optimizer::new(x, y, parameters);

        let (support_vectors, weight, b) = optimizer.smo();

        Ok(SVR {
            instances: Some(support_vectors),
            parameters: Some(parameters),
            w: Some(weight),
            b,
            phantom: PhantomData,
        })
    }

    /// Predict target values from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &'a X) -> Result<Vec<T>, Failed> {
        let (n, _) = x.shape();

        let mut y_hat: Vec<T> = Vec::<T>::zeros(n);

        for i in 0..n {
            y_hat.set(
                i,
                self.predict_for_row(Vec::from_iterator(
                    x.get_row(i).iterator(0).copied(), n)),
            );
        }

        Ok(y_hat)
    }

    pub(crate) fn predict_for_row(&self, x: Vec<T>) -> T {
        let mut f = self.b;

        for i in 0..self.instances.as_ref().unwrap().len() {
            f += self.w.as_ref().unwrap()[i]
                * T::from(self
                    .parameters
                    .as_ref()
                    .unwrap()
                    .kernel
                    .as_ref()
                    .unwrap()
                    .apply(
                        &x.iter().map(|e| e.to_f64().unwrap()).collect(),
                        &self.instances.as_ref().unwrap()[i],
                    )
                    .unwrap()
                ).unwrap()
        }

        T::from(f).unwrap()
    }
}

impl<'a, T: Number + RealNumber, X: Array2<T>, Y: Array1<T>> PartialEq for SVR<'a, T, X, Y> {
    fn eq(&self, other: &Self) -> bool {
        if (self.b - other.b).abs() > T::epsilon() * T::two()
            || self.w.as_ref().unwrap().len() != other.w.as_ref().unwrap().len()
            || self.instances.as_ref().unwrap().len() != other.instances.as_ref().unwrap().len()
        {
            false
        } else {
            for i in 0..self.w.as_ref().unwrap().len() {
                if (self.w.as_ref().unwrap()[i] - other.w.as_ref().unwrap()[i]).abs()
                    > T::epsilon()
                {
                    return false;
                }
            }
            for i in 0..self.instances.as_ref().unwrap().len() {
                if !self.instances.as_ref().unwrap()[i]
                    .approximate_eq(&other.instances.as_ref().unwrap()[i], f64::epsilon())
                {
                    return false;
                }
            }
            true
        }
    }
}

impl<T: Number + RealNumber> SupportVector<T> {
    fn new(i: usize, x: Vec<f64>, y: T, eps: T, k: f64) -> SupportVector<T> {
        SupportVector {
            index: i,
            x,
            grad: [eps + y, eps - y],
            k,
            alpha: [T::zero(), T::zero()],
        }
    }
}

impl<'a, T: Number + RealNumber> Optimizer<'a, T> {
    fn new<X: Array2<T>, Y: Array1<T>>(x: &'a X, y: &'a Y, parameters: &'a SVRParameters<'a, T>) -> Optimizer<'a, T> {
        let (n, _) = x.shape();

        let mut support_vectors: Vec<SupportVector<T>> = Vec::with_capacity(n);

        // initialize support vectors with kernel value (k)
        for i in 0..n {
            let k = parameters
                .kernel
                .as_ref()
                .unwrap()
                .apply(
                    &Vec::from_iterator(x.iterator(0).map(|e| e.to_f64().unwrap()), n),
                    &Vec::from_iterator(x.iterator(0).map(|e| e.to_f64().unwrap()), n),
                )
                .unwrap();
            support_vectors.push(SupportVector::<T>::new(
                i,
                Vec::from_iterator(x.get_row(i).iterator(0).map(|e| e.to_f64().unwrap()), n),
                T::from(*y.get(i)).unwrap(),
                parameters.eps,
                k,
            ));
        }

        Optimizer {
            tol: parameters.tol,
            c: parameters.c,
            parameters: Some(parameters),
            svmin: 0,
            svmax: 0,
            gmin: <T as Bounded>::max_value(),
            gmax: <T as Bounded>::min_value(),
            gminindex: 0,
            gmaxindex: 0,
            tau: T::from_f64(1e-12).unwrap(),
            sv: support_vectors,
        }
    }

    fn find_min_max_gradient(&mut self) {
        // self.gmin = <T as Bounded>::max_value()();
        // self.gmax = <T as Bounded>::min_value();

        for i in 0..self.sv.len() {
            let v = &self.sv[i];
            let g = -v.grad[0];
            let a = v.alpha[0];
            if g < self.gmin && a > T::zero() {
                self.gmin = g;
                self.gminindex = 0;
                self.svmin = i;
            }
            if g > self.gmax && a < self.c {
                self.gmax = g;
                self.gmaxindex = 0;
                self.svmax = i;
            }

            let g = v.grad[1];
            let a = v.alpha[1];
            if g < self.gmin && a < self.c {
                self.gmin = g;
                self.gminindex = 1;
                self.svmin = i;
            }
            if g > self.gmax && a > T::zero() {
                self.gmax = g;
                self.gmaxindex = 1;
                self.svmax = i;
            }
        }
    }

    /// Solves the quadratic programming (QP) problem that arises during the training of support-vector machines (SVM) algorithm.
    /// Returns:
    /// * support vectors (computed with f64)
    /// * hyperplane parameters: w and b (computed with T)
    fn smo(mut self) -> (Vec<Vec<f64>>, Vec<T>, T) {
        let cache: Cache<f64> = Cache::new(self.sv.len());

        self.find_min_max_gradient();

        while self.gmax - self.gmin > self.tol {
            let v1 = self.svmax;
            let i = self.gmaxindex;
            let old_alpha_i = self.sv[v1].alpha[i];

            let k1 = cache.get(self.sv[v1].index, || {
                self.sv
                    .iter()
                    .map(|vi| {
                        self.parameters
                            .unwrap()
                            .kernel
                            .as_ref()
                            .unwrap()
                            .apply(&self.sv[v1].x, &vi.x)
                            .unwrap()
                    })
                    .collect()
            });

            let mut v2 = self.svmin;
            let mut j = self.gminindex;
            let mut old_alpha_j = self.sv[v2].alpha[j];

            let mut best = T::zero();
            let gi = if i == 0 {
                -self.sv[v1].grad[0]
            } else {
                self.sv[v1].grad[1]
            };
            for jj in 0..self.sv.len() {
                let v = &self.sv[jj];
                let mut curv = self.sv[v1].k + v.k - 2f64 * k1[v.index];
                if curv <= 0f64 {
                    curv = self.tau.to_f64().unwrap();
                }

                let mut gj = -v.grad[0];
                if v.alpha[0] > T::zero() && gj < gi {
                    let gain = -((gi - gj) * (gi - gj)) / T::from(curv).unwrap();
                    if gain < best {
                        best = gain;
                        v2 = jj;
                        j = 0;
                        old_alpha_j = self.sv[v2].alpha[0];
                    }
                }

                gj = v.grad[1];
                if v.alpha[1] < self.c && gj < gi {
                    let gain = -((gi - gj) * (gi - gj)) / T::from(curv).unwrap();
                    if gain < best {
                        best = gain;
                        v2 = jj;
                        j = 1;
                        old_alpha_j = self.sv[v2].alpha[1];
                    }
                }
            }

            let k2 = cache.get(self.sv[v2].index, || {
                self.sv
                    .iter()
                    .map(|vi| {
                        self.parameters
                            .unwrap()
                            .kernel
                            .as_ref()
                            .unwrap()
                            .apply(&self.sv[v2].x, &vi.x)
                            .unwrap()
                    })
                    .collect()
            });

            let mut curv = self.sv[v1].k + self.sv[v2].k - 2f64 * k1[self.sv[v2].index];
            if curv <= 0f64 {
                curv = self.tau.to_f64().unwrap();
            }

            if i != j {
                let delta = (-self.sv[v1].grad[i] - self.sv[v2].grad[j]) / T::from(curv).unwrap();
                let diff = self.sv[v1].alpha[i] - self.sv[v2].alpha[j];
                self.sv[v1].alpha[i] += delta;
                self.sv[v2].alpha[j] += delta;

                if diff > T::zero() {
                    if self.sv[v2].alpha[j] < T::zero() {
                        self.sv[v2].alpha[j] = T::zero();
                        self.sv[v1].alpha[i] = diff;
                    }
                } else if self.sv[v1].alpha[i] < T::zero() {
                    self.sv[v1].alpha[i] = T::zero();
                    self.sv[v2].alpha[j] = -diff;
                }

                if diff > T::zero() {
                    if self.sv[v1].alpha[i] > self.c {
                        self.sv[v1].alpha[i] = self.c;
                        self.sv[v2].alpha[j] = self.c - diff;
                    }
                } else if self.sv[v2].alpha[j] > self.c {
                    self.sv[v2].alpha[j] = self.c;
                    self.sv[v1].alpha[i] = self.c + diff;
                }
            } else {
                let delta = (self.sv[v1].grad[i] - self.sv[v2].grad[j]) / T::from(curv).unwrap();
                let sum = self.sv[v1].alpha[i] + self.sv[v2].alpha[j];
                self.sv[v1].alpha[i] -= delta;
                self.sv[v2].alpha[j] += delta;

                if sum > self.c {
                    if self.sv[v1].alpha[i] > self.c {
                        self.sv[v1].alpha[i] = self.c;
                        self.sv[v2].alpha[j] = sum - self.c;
                    }
                } else if self.sv[v2].alpha[j] < T::zero() {
                    self.sv[v2].alpha[j] = T::zero();
                    self.sv[v1].alpha[i] = sum;
                }

                if sum > self.c {
                    if self.sv[v2].alpha[j] > self.c {
                        self.sv[v2].alpha[j] = self.c;
                        self.sv[v1].alpha[i] = sum - self.c;
                    }
                } else if self.sv[v1].alpha[i] < T::zero() {
                    self.sv[v1].alpha[i] = T::zero();
                    self.sv[v2].alpha[j] = sum;
                }
            }

            let delta_alpha_i = self.sv[v1].alpha[i] - old_alpha_i;
            let delta_alpha_j = self.sv[v2].alpha[j] - old_alpha_j;

            let si = T::two() * T::from_usize(i).unwrap() - T::one();
            let sj = T::two() * T::from_usize(j).unwrap() - T::one();
            for v in self.sv.iter_mut() {
                v.grad[0] -= si * T::from(k1[v.index]).unwrap() * delta_alpha_i + sj * T::from(k2[v.index]).unwrap() * delta_alpha_j;
                v.grad[1] += si * T::from(k1[v.index]).unwrap() * delta_alpha_i + sj * T::from(k2[v.index]).unwrap() * delta_alpha_j;
            }

            self.find_min_max_gradient();
        }

        let b = -(self.gmax + self.gmin) / T::two();

        let mut support_vectors: Vec<Vec<f64>> = Vec::new();
        let mut w: Vec<T> = Vec::new();

        for v in self.sv {
            if v.alpha[0] != v.alpha[1] {
                support_vectors.push(v.x);
                w.push(v.alpha[1] - v.alpha[0]);
            }
        }

        (support_vectors, w, b)
    }
}

impl<T: Clone> Cache<T> {
    fn new(n: usize) -> Cache<T> {
        Cache {
            data: vec![RefCell::new(None); n],
        }
    }

    fn get<F: Fn() -> Vec<T>>(&self, i: usize, or: F) -> Ref<'_, Vec<T>> {
        if self.data[i].borrow().is_none() {
            self.data[i].replace(Some(or()));
        }
        Ref::map(self.data[i].borrow(), |v| v.as_ref().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::metrics::mean_squared_error;
    #[cfg(feature = "serde")]
    use crate::svm::*;

    // #[test]
    // fn search_parameters() {
    //     let parameters: SVRSearchParameters<f64, DenseMatrix<f64>, LinearKernel> =
    //         SVRSearchParameters {
    //             eps: vec![0., 1.],
    //             kernel: vec![LinearKernel {}],
    //             ..Default::default()
    //         };
    //     let mut iter = parameters.into_iter();
    //     let next = iter.next().unwrap();
    //     assert_eq!(next.eps, 0.);
    //     assert_eq!(next.kernel, LinearKernel {});
    //     let next = iter.next().unwrap();
    //     assert_eq!(next.eps, 1.);
    //     assert_eq!(next.kernel, LinearKernel {});
    //     assert!(iter.next().is_none());
    // }

    // TODO: had to disable this test as it runs for too long
    // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    // #[test]
    // fn svr_fit_predict() {
    //     let x = DenseMatrix::from_2d_array(&[
    //         &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
    //         &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
    //         &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
    //         &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
    //         &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
    //         &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
    //         &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
    //         &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
    //         &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
    //         &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
    //         &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
    //         &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
    //         &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
    //         &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
    //         &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
    //         &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
    //     ]);

    //     let y: Vec<f64> = vec![
    //         83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
    //         114.2, 115.7, 116.9,
    //     ];

    //     let knl = Kernels::linear();
    //     let y_hat = SVR::fit(&x, &y, &SVRParameters::default()
    //          .with_eps(2.0)
    //          .with_c(10.0)
    //          .with_kernel(&knl)
    //     )
    //         .and_then(|lr| lr.predict(&x))
    //         .unwrap();

    //     assert!(mean_squared_error(&y_hat, &y) < 2.5);
    // }

    // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    // #[test]
    // #[cfg(feature = "serde")]
    // fn svr_serde() {
    //     let x = DenseMatrix::from_2d_array(&[
    //         &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
    //         &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
    //         &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
    //         &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
    //         &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
    //         &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
    //         &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
    //         &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
    //         &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
    //         &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
    //         &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
    //         &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
    //         &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
    //         &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
    //         &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
    //         &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
    //     ]);

    //     let y: Vec<f64> = vec![
    //         83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
    //         114.2, 115.7, 116.9,
    //     ];

    //     let svr = SVR::fit(&x, &y, Default::default()).unwrap();

    //     let deserialized_svr: SVR<f64, DenseMatrix<f64>, LinearKernel> =
    //         serde_json::from_str(&serde_json::to_string(&svr).unwrap()).unwrap();

    //     assert_eq!(svr, deserialized_svr);
    // }
}
