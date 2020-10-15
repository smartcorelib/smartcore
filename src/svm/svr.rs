//! # Epsilon-Support Vector Regression.
//!
//! Example
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
//! let svr = SVR::fit(&x, &y,
//!             LinearKernel {},
//!             SVRParameters {
//!                 eps: 2.0,
//!                 c: 10.0,
//!                 tol: 1e-3,
//!             }).unwrap();
//!
//! let y_hat = svr.predict(&x).unwrap();
//! ```
use std::cell::{Ref, RefCell};
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::linalg::Matrix;
use crate::math::num::RealNumber;
use crate::svm::Kernel;

#[derive(Serialize, Deserialize, Debug)]

/// SVR Parameters
pub struct SVRParameters<T: RealNumber> {
    /// Epsilon in the epsilon-SVR model
    pub eps: T,
    /// Regularization parameter.
    pub c: T,
    /// Tolerance for stopping criterion
    pub tol: T,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "M::RowVector: Serialize, K: Serialize, T: Serialize",
    deserialize = "M::RowVector: Deserialize<'de>, K: Deserialize<'de>, T: Deserialize<'de>",
))]

/// Epsilon-Support Vector Regression
pub struct SVR<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
    kernel: K,
    instances: Vec<M::RowVector>,
    w: Vec<T>,
    b: T,
}

#[derive(Serialize, Deserialize, Debug)]
struct SupportVector<T: RealNumber, V: BaseVector<T>> {
    index: usize,
    x: V,
    alpha: [T; 2],
    grad: [T; 2],
    k: T,
}

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

impl<T: RealNumber> Default for SVRParameters<T> {
    fn default() -> Self {
        SVRParameters {
            eps: T::from_f64(0.1).unwrap(),
            c: T::one(),
            tol: T::from_f64(1e-3).unwrap(),
        }
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
        kernel: K,
        parameters: SVRParameters<T>,
    ) -> Result<SVR<T, M, K>, Failed> {
        let (n, _) = x.shape();

        if n != y.len() {
            return Err(Failed::fit(&format!(
                "Number of rows of X doesn't match number of rows of Y"
            )));
        }

        let optimizer = Optimizer::optimize(x, y, &kernel, &parameters);

        let (support_vectors, weight, b) = optimizer.smo();

        Ok(SVR {
            kernel: kernel,
            instances: support_vectors,
            w: weight,
            b: b,
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

        return f;
    }
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> PartialEq for SVR<T, M, K> {
    fn eq(&self, other: &Self) -> bool {
        if self.b != other.b
            || self.w.len() != other.w.len()
            || self.instances.len() != other.instances.len()
        {
            return false;
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
            return true;
        }
    }
}

impl<T: RealNumber, V: BaseVector<T>> SupportVector<T, V> {
    fn new<K: Kernel<T, V>>(i: usize, x: V, y: T, eps: T, k: &K) -> SupportVector<T, V> {
        let k_v = k.apply(&x, &x);
        SupportVector {
            index: i,
            x: x,
            grad: [eps + y, eps - y],
            k: k_v,
            alpha: [T::zero(), T::zero()],
        }
    }
}

impl<'a, T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Optimizer<'a, T, M, K> {
    fn optimize(
        x: &M,
        y: &M::RowVector,
        kernel: &'a K,
        parameters: &SVRParameters<T>,
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
            kernel: kernel,
        }
    }

    fn minmax(&mut self) {
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

    fn smo(mut self) -> (Vec<M::RowVector>, Vec<T>, T) {
        let cache: Cache<T> = Cache::new(self.sv.len());

        self.minmax();

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
                } else {
                    if self.sv[v1].alpha[i] < T::zero() {
                        self.sv[v1].alpha[i] = T::zero();
                        self.sv[v2].alpha[j] = -diff;
                    }
                }

                if diff > T::zero() {
                    if self.sv[v1].alpha[i] > self.c {
                        self.sv[v1].alpha[i] = self.c;
                        self.sv[v2].alpha[j] = self.c - diff;
                    }
                } else {
                    if self.sv[v2].alpha[j] > self.c {
                        self.sv[v2].alpha[j] = self.c;
                        self.sv[v1].alpha[i] = self.c + diff;
                    }
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
                } else {
                    if self.sv[v2].alpha[j] < T::zero() {
                        self.sv[v2].alpha[j] = T::zero();
                        self.sv[v1].alpha[i] = sum;
                    }
                }

                if sum > self.c {
                    if self.sv[v2].alpha[j] > self.c {
                        self.sv[v2].alpha[j] = self.c;
                        self.sv[v1].alpha[i] = sum - self.c;
                    }
                } else {
                    if self.sv[v1].alpha[i] < T::zero() {
                        self.sv[v1].alpha[i] = T::zero();
                        self.sv[v2].alpha[j] = sum;
                    }
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

            self.minmax();
        }

        let b = -(self.gmax + self.gmin) / T::two();

        let mut result: Vec<M::RowVector> = Vec::new();
        let mut alpha: Vec<T> = Vec::new();

        for v in self.sv {
            if v.alpha[0] != v.alpha[1] {
                result.push(v.x);
                alpha.push(v.alpha[1] - v.alpha[0]);
            }
        }

        (result, alpha, b)
    }
}

impl<T: Clone> Cache<T> {
    fn new(n: usize) -> Cache<T> {
        Cache {
            data: vec![RefCell::new(None); n],
        }
    }

    fn get<F: Fn() -> Vec<T>>(&self, i: usize, or: F) -> Ref<Vec<T>> {
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

        let y_hat = SVR::fit(
            &x,
            &y,
            LinearKernel {},
            SVRParameters {
                eps: 2.0,
                c: 10.0,
                tol: 1e-3,
            },
        )
        .and_then(|lr| lr.predict(&x))
        .unwrap();

        println!("{:?}", y_hat);

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

        let svr = SVR::fit(&x, &y, LinearKernel {}, Default::default()).unwrap();

        let deserialized_svr: SVR<f64, DenseMatrix<f64>, LinearKernel> =
            serde_json::from_str(&serde_json::to_string(&svr).unwrap()).unwrap();

        assert_eq!(svr, deserialized_svr);
    }
}
