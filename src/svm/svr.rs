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
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::linear::linear_regression::*;
//! use smartcore::svm::*;
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
//! let svr = SVR::fit(&x, &y, SVRParameters::default().with_eps(2.0).with_c(10.0)).unwrap();
//!
//! let y_hat = svr.predict(&x).unwrap();
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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Predictor, SupervisedEstimator};
use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::svm::{Kernel, Kernels, LinearKernel};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// SVR Parameters
pub struct SVRParameters<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
    /// Epsilon in the epsilon-SVR model.
    pub eps: T,
    /// Regularization parameter.
    pub c: T,
    /// Tolerance for stopping criterion.
    pub tol: T,
    /// The kernel function.
    pub kernel: K,
    /// Unused parameter.
    m: PhantomData<M>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "M::RowVector: Serialize, K: Serialize, T: Serialize",
        deserialize = "M::RowVector: Deserialize<'de>, K: Deserialize<'de>, T: Deserialize<'de>",
    ))
)]

/// Epsilon-Support Vector Regression
pub struct SVR<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
    kernel: K,
    instances: Vec<M::RowVector>,
    w: Vec<T>,
    b: T,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
struct SupportVector<T: RealNumber, V: BaseVector<T>> {
    index: usize,
    x: V,
    alpha: [T; 2],
    grad: [T; 2],
    k: T,
}

/// Sequential Minimal Optimization algorithm
struct Optimizer<'a, T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
    tol: T,
    c: T,
    svmin: usize,
    svmax: usize,
    gmin: T,
    gmax: T,
    gminindex: usize,
    gmaxindex: usize,
    tau: T,
    sv: Vec<SupportVector<T, M::RowVector>>,
    kernel: &'a K,
}

struct Cache<T: Clone> {
    data: Vec<RefCell<Option<Vec<T>>>>,
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> SVRParameters<T, M, K> {
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
    pub fn with_kernel<KK: Kernel<T, M::RowVector>>(&self, kernel: KK) -> SVRParameters<T, M, KK> {
        SVRParameters {
            eps: self.eps,
            c: self.c,
            tol: self.tol,
            kernel,
            m: PhantomData,
        }
    }
}

impl<T: RealNumber, M: Matrix<T>> Default for SVRParameters<T, M, LinearKernel> {
    fn default() -> Self {
        SVRParameters {
            eps: T::from_f64(0.1).unwrap(),
            c: T::one(),
            tol: T::from_f64(1e-3).unwrap(),
            kernel: Kernels::linear(),
            m: PhantomData,
        }
    }
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>>
    SupervisedEstimator<M, M::RowVector, SVRParameters<T, M, K>> for SVR<T, M, K>
{
    fn fit(x: &M, y: &M::RowVector, parameters: SVRParameters<T, M, K>) -> Result<Self, Failed> {
        SVR::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Predictor<M, M::RowVector>
    for SVR<T, M, K>
{
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> SVR<T, M, K> {
    /// Fits SVR to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - target values
    /// * `kernel` - the kernel function
    /// * `parameters` - optional parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: SVRParameters<T, M, K>,
    ) -> Result<SVR<T, M, K>, Failed> {
        let (n, _) = x.shape();

        if n != y.len() {
            return Err(Failed::fit(
                &"Number of rows of X doesn\'t match number of rows of Y".to_string(),
            ));
        }

        let optimizer = Optimizer::new(x, y, &parameters.kernel, &parameters);

        let (support_vectors, weight, b) = optimizer.smo();

        Ok(SVR {
            kernel: parameters.kernel,
            instances: support_vectors,
            w: weight,
            b,
        })
    }

    /// Predict target values from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        let (n, _) = x.shape();

        let mut y_hat = M::RowVector::zeros(n);

        for i in 0..n {
            y_hat.set(i, self.predict_for_row(x.get_row(i)));
        }

        Ok(y_hat)
    }

    pub(in crate) fn predict_for_row(&self, x: M::RowVector) -> T {
        let mut f = self.b;

        for i in 0..self.instances.len() {
            f += self.w[i] * self.kernel.apply(&x, &self.instances[i]);
        }

        f
    }
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> PartialEq for SVR<T, M, K> {
    fn eq(&self, other: &Self) -> bool {
        if (self.b - other.b).abs() > T::epsilon() * T::two()
            || self.w.len() != other.w.len()
            || self.instances.len() != other.instances.len()
        {
            false
        } else {
            for i in 0..self.w.len() {
                if (self.w[i] - other.w[i]).abs() > T::epsilon() {
                    return false;
                }
            }
            for i in 0..self.instances.len() {
                if !self.instances[i].approximate_eq(&other.instances[i], T::epsilon()) {
                    return false;
                }
            }
            true
        }
    }
}

impl<T: RealNumber, V: BaseVector<T>> SupportVector<T, V> {
    fn new<K: Kernel<T, V>>(i: usize, x: V, y: T, eps: T, k: &K) -> SupportVector<T, V> {
        let k_v = k.apply(&x, &x);
        SupportVector {
            index: i,
            x,
            grad: [eps + y, eps - y],
            k: k_v,
            alpha: [T::zero(), T::zero()],
        }
    }
}

impl<'a, T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Optimizer<'a, T, M, K> {
    fn new(
        x: &M,
        y: &M::RowVector,
        kernel: &'a K,
        parameters: &SVRParameters<T, M, K>,
    ) -> Optimizer<'a, T, M, K> {
        let (n, _) = x.shape();

        let mut support_vectors: Vec<SupportVector<T, M::RowVector>> = Vec::with_capacity(n);

        for i in 0..n {
            support_vectors.push(SupportVector::new(
                i,
                x.get_row(i),
                y.get(i),
                parameters.eps,
                kernel,
            ));
        }

        Optimizer {
            tol: parameters.tol,
            c: parameters.c,
            svmin: 0,
            svmax: 0,
            gmin: T::max_value(),
            gmax: T::min_value(),
            gminindex: 0,
            gmaxindex: 0,
            tau: T::from_f64(1e-12).unwrap(),
            sv: support_vectors,
            kernel,
        }
    }

    fn find_min_max_gradient(&mut self) {
        self.gmin = T::max_value();
        self.gmax = T::min_value();

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

    /// Solvs the quadratic programming (QP) problem that arises during the training of support-vector machines (SVM) algorithm.
    /// Returns:
    /// * support vectors
    /// * hyperplane parameters: w and b
    fn smo(mut self) -> (Vec<M::RowVector>, Vec<T>, T) {
        let cache: Cache<T> = Cache::new(self.sv.len());

        self.find_min_max_gradient();

        while self.gmax - self.gmin > self.tol {
            let v1 = self.svmax;
            let i = self.gmaxindex;
            let old_alpha_i = self.sv[v1].alpha[i];

            let k1 = cache.get(self.sv[v1].index, || {
                self.sv
                    .iter()
                    .map(|vi| self.kernel.apply(&self.sv[v1].x, &vi.x))
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
                let mut curv = self.sv[v1].k + v.k - T::two() * k1[v.index];
                if curv <= T::zero() {
                    curv = self.tau;
                }

                let mut gj = -v.grad[0];
                if v.alpha[0] > T::zero() && gj < gi {
                    let gain = -((gi - gj) * (gi - gj)) / curv;
                    if gain < best {
                        best = gain;
                        v2 = jj;
                        j = 0;
                        old_alpha_j = self.sv[v2].alpha[0];
                    }
                }

                gj = v.grad[1];
                if v.alpha[1] < self.c && gj < gi {
                    let gain = -((gi - gj) * (gi - gj)) / curv;
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
                    .map(|vi| self.kernel.apply(&self.sv[v2].x, &vi.x))
                    .collect()
            });

            let mut curv = self.sv[v1].k + self.sv[v2].k - T::two() * k1[self.sv[v2].index];
            if curv <= T::zero() {
                curv = self.tau;
            }

            if i != j {
                let delta = (-self.sv[v1].grad[i] - self.sv[v2].grad[j]) / curv;
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
                let delta = (self.sv[v1].grad[i] - self.sv[v2].grad[j]) / curv;
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
                v.grad[0] -= si * k1[v.index] * delta_alpha_i + sj * k2[v.index] * delta_alpha_j;
                v.grad[1] += si * k1[v.index] * delta_alpha_i + sj * k2[v.index] * delta_alpha_j;
            }

            self.find_min_max_gradient();
        }

        let b = -(self.gmax + self.gmin) / T::two();

        let mut support_vectors: Vec<M::RowVector> = Vec::new();
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
    use crate::linalg::naive::dense_matrix::*;
    use crate::metrics::mean_squared_error;
    use crate::svm::*;

    #[test]
    fn svr_fit_predict() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
            &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);

        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let y_hat = SVR::fit(&x, &y, SVRParameters::default().with_eps(2.0).with_c(10.0))
            .and_then(|lr| lr.predict(&x))
            .unwrap();

        assert!(mean_squared_error(&y_hat, &y) < 2.5);
    }

    #[test]
    fn svr_serde() {
        let x = DenseMatrix::from_2d_array(&[
            &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
            &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
            &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
            &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
            &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
            &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
            &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
            &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
            &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
            &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
            &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
            &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
            &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
            &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
            &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
            &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
        ]);

        let y: Vec<f64> = vec![
            83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6,
            114.2, 115.7, 116.9,
        ];

        let svr = SVR::fit(&x, &y, Default::default()).unwrap();

        let deserialized_svr: SVR<f64, DenseMatrix<f64>, LinearKernel> =
            serde_json::from_str(&serde_json::to_string(&svr).unwrap()).unwrap();

        assert_eq!(svr, deserialized_svr);
    }
}
