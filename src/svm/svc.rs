//! # Support Vector Classifier.
//!
//! Support Vector Classifier (SVC) is a binary classifier that uses an optimal hyperplane to separate the points in the input variable space by their class.
//!
//! During training, SVC chooses a Maximal-Margin hyperplane that can separate all training instances with the largest margin.
//! The margin is calculated as the perpendicular distance from the boundary to only the closest points. Hence, only these points are relevant in defining
//! the hyperplane and in the construction of the classifier. These points are called the support vectors.
//!
//! While SVC selects a hyperplane with the largest margin it allows some points in the training data to violate the separating boundary.
//! The parameter `C` > 0 gives you control over how SVC will handle violating points. The bigger the value of this parameter the more we penalize the algorithm
//! for incorrectly classified points. In other words, setting this parameter to a small value will result in a classifier that allows for a big number
//! of misclassified samples. Mathematically, SVC optimization problem can be defined as:
//!
//! \\[\underset{w, \zeta}{minimize} \space \space \frac{1}{2} \lVert \vec{w} \rVert^2 + C\sum_{i=1}^m \zeta_i \\]
//!
//! subject to:
//!
//! \\[y_i(\langle\vec{w}, \vec{x}_i \rangle + b) \geq 1 - \zeta_i \\]
//! \\[\zeta_i \geq 0 for \space any \space i = 1, ... , m\\]
//!
//! Where \\( m \\) is a number of training samples, \\( y_i \\) is a label value (either 1 or -1) and \\(\langle\vec{w}, \vec{x}_i \rangle + b\\) is a decision boundary.
//!
//! To solve this optimization problem, SmartCore uses an [approximate SVM solver](https://leon.bottou.org/projects/lasvm).
//! The optimizer reaches accuracies similar to that of a real SVM after performing two passes through the training examples. You can choose the number of passes
//! through the data that the algorithm takes by changing the `epoch` parameter of the classifier.
//!
//! Example:
//!
//! ```
//! use smartcore::linalg::naive::dense_matrix::*;
//! use smartcore::svm::Kernels;
//! use smartcore::svm::svc::{SVC, SVCParameters};
//!
//! // Iris dataset
//! let x = DenseMatrix::from_2d_array(&[
//!            &[5.1, 3.5, 1.4, 0.2],
//!            &[4.9, 3.0, 1.4, 0.2],
//!            &[4.7, 3.2, 1.3, 0.2],
//!            &[4.6, 3.1, 1.5, 0.2],
//!            &[5.0, 3.6, 1.4, 0.2],
//!            &[5.4, 3.9, 1.7, 0.4],
//!            &[4.6, 3.4, 1.4, 0.3],
//!            &[5.0, 3.4, 1.5, 0.2],
//!            &[4.4, 2.9, 1.4, 0.2],
//!            &[4.9, 3.1, 1.5, 0.1],
//!            &[7.0, 3.2, 4.7, 1.4],
//!            &[6.4, 3.2, 4.5, 1.5],
//!            &[6.9, 3.1, 4.9, 1.5],
//!            &[5.5, 2.3, 4.0, 1.3],
//!            &[6.5, 2.8, 4.6, 1.5],
//!            &[5.7, 2.8, 4.5, 1.3],
//!            &[6.3, 3.3, 4.7, 1.6],
//!            &[4.9, 2.4, 3.3, 1.0],
//!            &[6.6, 2.9, 4.6, 1.3],
//!            &[5.2, 2.7, 3.9, 1.4],
//!         ]);
//! let y = vec![ 0., 0., 0., 0., 0., 0., 0., 0.,
//!            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.];
//!
//! let svc = SVC::fit(&x, &y, SVCParameters::default().with_c(200.0)).unwrap();
//!
//! let y_hat = svc.predict(&x).unwrap();
//! ```
//!
//! ## References:
//!
//! * ["Support Vector Machines", Kowalczyk A., 2017](https://www.svm-tutorial.com/2017/10/support-vector-machines-succinctly-released/)
//! * ["Fast Kernel Classifiers with Online and Active Learning", Bordes A., Ertekin S., Weston J., Bottou L., 2005](https://www.jmlr.org/papers/volume6/bordes05a/bordes05a.pdf)
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;

use rand::seq::SliceRandom;

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
/// SVC Parameters
pub struct SVCParameters<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
    /// Number of epochs.
    pub epoch: usize,
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
/// Support Vector Classifier
pub struct SVC<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
    classes: Vec<T>,
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
    alpha: T,
    grad: T,
    cmin: T,
    cmax: T,
    k: T,
}

struct Cache<'a, T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
    kernel: &'a K,
    data: HashMap<(usize, usize), T>,
    phantom: PhantomData<M>,
}

struct Optimizer<'a, T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> {
    x: &'a M,
    y: &'a M::RowVector,
    parameters: &'a SVCParameters<T, M, K>,
    svmin: usize,
    svmax: usize,
    gmin: T,
    gmax: T,
    tau: T,
    sv: Vec<SupportVector<T, M::RowVector>>,
    kernel: &'a K,
    recalculate_minmax_grad: bool,
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> SVCParameters<T, M, K> {
    /// Number of epochs.
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = epoch;
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
    pub fn with_kernel<KK: Kernel<T, M::RowVector>>(&self, kernel: KK) -> SVCParameters<T, M, KK> {
        SVCParameters {
            epoch: self.epoch,
            c: self.c,
            tol: self.tol,
            kernel,
            m: PhantomData,
        }
    }
}

impl<T: RealNumber, M: Matrix<T>> Default for SVCParameters<T, M, LinearKernel> {
    fn default() -> Self {
        SVCParameters {
            epoch: 2,
            c: T::one(),
            tol: T::from_f64(1e-3).unwrap(),
            kernel: Kernels::linear(),
            m: PhantomData,
        }
    }
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>>
    SupervisedEstimator<M, M::RowVector, SVCParameters<T, M, K>> for SVC<T, M, K>
{
    fn fit(x: &M, y: &M::RowVector, parameters: SVCParameters<T, M, K>) -> Result<Self, Failed> {
        SVC::fit(x, y, parameters)
    }
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Predictor<M, M::RowVector>
    for SVC<T, M, K>
{
    fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        self.predict(x)
    }
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> SVC<T, M, K> {
    /// Fits SVC to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - class labels
    /// * `parameters` - optional parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(
        x: &M,
        y: &M::RowVector,
        parameters: SVCParameters<T, M, K>,
    ) -> Result<SVC<T, M, K>, Failed> {
        let (n, _) = x.shape();

        if n != y.len() {
            return Err(Failed::fit(
                "Number of rows of X doesn\'t match number of rows of Y",
            ));
        }

        let classes = y.unique();

        if classes.len() != 2 {
            return Err(Failed::fit(&format!(
                "Incorrect number of classes {}",
                classes.len()
            )));
        }

        // Make sure class labels are either 1 or -1
        let mut y = y.clone();
        for i in 0..y.len() {
            let y_v = y.get(i);
            if y_v != -T::one() || y_v != T::one() {
                match y_v == classes[0] {
                    true => y.set(i, -T::one()),
                    false => y.set(i, T::one()),
                }
            }
        }

        let optimizer = Optimizer::new(x, &y, &parameters.kernel, &parameters);

        let (support_vectors, weight, b) = optimizer.optimize();

        Ok(SVC {
            classes,
            kernel: parameters.kernel,
            instances: support_vectors,
            w: weight,
            b,
        })
    }

    /// Predicts estimated class labels from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &M) -> Result<M::RowVector, Failed> {
        let (n, _) = x.shape();

        let mut y_hat = M::RowVector::zeros(n);

        for i in 0..n {
            let cls_idx = match self.predict_for_row(x.get_row(i)) == T::one() {
                false => self.classes[0],
                true => self.classes[1],
            };
            y_hat.set(i, cls_idx);
        }

        Ok(y_hat)
    }

    fn predict_for_row(&self, x: M::RowVector) -> T {
        let mut f = self.b;

        for i in 0..self.instances.len() {
            f += self.w[i] * self.kernel.apply(&x, &self.instances[i]);
        }

        if f > T::zero() {
            T::one()
        } else {
            -T::one()
        }
    }
}

impl<T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> PartialEq for SVC<T, M, K> {
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
    fn new<K: Kernel<T, V>>(i: usize, x: V, y: T, g: T, c: T, k: &K) -> SupportVector<T, V> {
        let k_v = k.apply(&x, &x);
        let (cmin, cmax) = if y > T::zero() {
            (T::zero(), c)
        } else {
            (-c, T::zero())
        };
        SupportVector {
            index: i,
            x,
            grad: g,
            k: k_v,
            alpha: T::zero(),
            cmin,
            cmax,
        }
    }
}

impl<'a, T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Cache<'a, T, M, K> {
    fn new(kernel: &'a K) -> Cache<'a, T, M, K> {
        Cache {
            kernel,
            data: HashMap::new(),
            phantom: PhantomData,
        }
    }

    fn get(&mut self, i: &SupportVector<T, M::RowVector>, j: &SupportVector<T, M::RowVector>) -> T {
        let idx_i = i.index;
        let idx_j = j.index;
        #[allow(clippy::or_fun_call)]
        let entry = self
            .data
            .entry((idx_i, idx_j))
            .or_insert(self.kernel.apply(&i.x, &j.x));
        *entry
    }

    fn insert(&mut self, key: (usize, usize), value: T) {
        self.data.insert(key, value);
    }

    fn drop(&mut self, idxs_to_drop: HashSet<usize>) {
        self.data.retain(|k, _| !idxs_to_drop.contains(&k.0));
    }
}

impl<'a, T: RealNumber, M: Matrix<T>, K: Kernel<T, M::RowVector>> Optimizer<'a, T, M, K> {
    fn new(
        x: &'a M,
        y: &'a M::RowVector,
        kernel: &'a K,
        parameters: &'a SVCParameters<T, M, K>,
    ) -> Optimizer<'a, T, M, K> {
        let (n, _) = x.shape();

        Optimizer {
            x,
            y,
            parameters,
            svmin: 0,
            svmax: 0,
            gmin: T::max_value(),
            gmax: T::min_value(),
            tau: T::from_f64(1e-12).unwrap(),
            sv: Vec::with_capacity(n),
            kernel,
            recalculate_minmax_grad: true,
        }
    }

    fn optimize(mut self) -> (Vec<M::RowVector>, Vec<T>, T) {
        let (n, _) = self.x.shape();

        let mut cache = Cache::new(self.kernel);

        self.initialize(&mut cache);

        let tol = self.parameters.tol;
        let good_enough = T::from_i32(1000).unwrap();

        for _ in 0..self.parameters.epoch {
            for i in Self::permutate(n) {
                self.process(i, self.x.get_row(i), self.y.get(i), &mut cache);
                loop {
                    self.reprocess(tol, &mut cache);
                    self.find_min_max_gradient();
                    if self.gmax - self.gmin < good_enough {
                        break;
                    }
                }
            }
        }

        self.finish(&mut cache);

        let mut support_vectors: Vec<M::RowVector> = Vec::new();
        let mut w: Vec<T> = Vec::new();

        let b = (self.gmax + self.gmin) / T::two();

        for v in self.sv {
            support_vectors.push(v.x);
            w.push(v.alpha);
        }

        (support_vectors, w, b)
    }

    fn initialize(&mut self, cache: &mut Cache<'_, T, M, K>) {
        let (n, _) = self.x.shape();
        let few = 5;
        let mut cp = 0;
        let mut cn = 0;

        for i in Self::permutate(n) {
            if self.y.get(i) == T::one() && cp < few {
                if self.process(i, self.x.get_row(i), self.y.get(i), cache) {
                    cp += 1;
                }
            } else if self.y.get(i) == -T::one()
                && cn < few
                && self.process(i, self.x.get_row(i), self.y.get(i), cache)
            {
                cn += 1;
            }

            if cp >= few && cn >= few {
                break;
            }
        }
    }

    fn process(&mut self, i: usize, x: M::RowVector, y: T, cache: &mut Cache<'_, T, M, K>) -> bool {
        for j in 0..self.sv.len() {
            if self.sv[j].index == i {
                return true;
            }
        }

        let mut g = y;

        let mut cache_values: Vec<((usize, usize), T)> = Vec::new();

        for v in self.sv.iter() {
            let k = self.kernel.apply(&v.x, &x);
            cache_values.push(((i, v.index), k));
            g -= v.alpha * k;
        }

        self.find_min_max_gradient();

        if self.gmin < self.gmax
            && ((y > T::zero() && g < self.gmin) || (y < T::zero() && g > self.gmax))
        {
            return false;
        }

        for v in cache_values {
            cache.insert(v.0, v.1);
        }

        self.sv.insert(
            0,
            SupportVector::new(i, x, y, g, self.parameters.c, self.kernel),
        );

        if y > T::zero() {
            self.smo(None, Some(0), T::zero(), cache);
        } else {
            self.smo(Some(0), None, T::zero(), cache);
        }

        true
    }

    fn reprocess(&mut self, tol: T, cache: &mut Cache<'_, T, M, K>) -> bool {
        let status = self.smo(None, None, tol, cache);
        self.clean(cache);
        status
    }

    fn finish(&mut self, cache: &mut Cache<'_, T, M, K>) {
        let mut max_iter = self.sv.len();

        while self.smo(None, None, self.parameters.tol, cache) && max_iter > 0 {
            max_iter -= 1;
        }

        self.clean(cache);
    }

    fn find_min_max_gradient(&mut self) {
        if !self.recalculate_minmax_grad {
            return;
        }

        self.gmin = T::max_value();
        self.gmax = T::min_value();

        for i in 0..self.sv.len() {
            let v = &self.sv[i];
            let g = v.grad;
            let a = v.alpha;
            if g < self.gmin && a > v.cmin {
                self.gmin = g;
                self.svmin = i;
            }
            if g > self.gmax && a < v.cmax {
                self.gmax = g;
                self.svmax = i;
            }
        }

        self.recalculate_minmax_grad = false
    }

    fn clean(&mut self, cache: &mut Cache<'_, T, M, K>) {
        self.find_min_max_gradient();

        let gmax = self.gmax;
        let gmin = self.gmin;

        let mut idxs_to_drop: HashSet<usize> = HashSet::new();

        self.sv.retain(|v| {
            if v.alpha == T::zero()
                && ((v.grad >= gmax && T::zero() >= v.cmax)
                    || (v.grad <= gmin && T::zero() <= v.cmin))
            {
                idxs_to_drop.insert(v.index);
                return false;
            };
            true
        });

        cache.drop(idxs_to_drop);
        self.recalculate_minmax_grad = true;
    }

    fn permutate(n: usize) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let mut range: Vec<usize> = (0..n).collect();
        range.shuffle(&mut rng);
        range
    }

    fn select_pair(
        &mut self,
        idx_1: Option<usize>,
        idx_2: Option<usize>,
        cache: &mut Cache<'_, T, M, K>,
    ) -> Option<(usize, usize, T)> {
        match (idx_1, idx_2) {
            (None, None) => {
                if self.gmax > -self.gmin {
                    self.select_pair(None, Some(self.svmax), cache)
                } else {
                    self.select_pair(Some(self.svmin), None, cache)
                }
            }
            (Some(idx_1), None) => {
                let sv1 = &self.sv[idx_1];
                let mut idx_2 = None;
                let mut k_v_12 = None;
                let km = sv1.k;
                let gm = sv1.grad;
                let mut best = T::zero();
                for i in 0..self.sv.len() {
                    let v = &self.sv[i];
                    let z = v.grad - gm;
                    let k = cache.get(sv1, v);
                    let mut curv = km + v.k - T::two() * k;
                    if curv <= T::zero() {
                        curv = self.tau;
                    }
                    let mu = z / curv;
                    if (mu > T::zero() && v.alpha < v.cmax) || (mu < T::zero() && v.alpha > v.cmin)
                    {
                        let gain = z * mu;
                        if gain > best {
                            best = gain;
                            idx_2 = Some(i);
                            k_v_12 = Some(k);
                        }
                    }
                }

                idx_2.map(|idx_2| {
                    (
                        idx_1,
                        idx_2,
                        k_v_12.unwrap_or_else(|| {
                            self.kernel.apply(&self.sv[idx_1].x, &self.sv[idx_2].x)
                        }),
                    )
                })
            }
            (None, Some(idx_2)) => {
                let mut idx_1 = None;
                let sv2 = &self.sv[idx_2];
                let mut k_v_12 = None;
                let km = sv2.k;
                let gm = sv2.grad;
                let mut best = T::zero();
                for i in 0..self.sv.len() {
                    let v = &self.sv[i];
                    let z = gm - v.grad;
                    let k = cache.get(sv2, v);
                    let mut curv = km + v.k - T::two() * k;
                    if curv <= T::zero() {
                        curv = self.tau;
                    }

                    let mu = z / curv;
                    if (mu > T::zero() && v.alpha > v.cmin) || (mu < T::zero() && v.alpha < v.cmax)
                    {
                        let gain = z * mu;
                        if gain > best {
                            best = gain;
                            idx_1 = Some(i);
                            k_v_12 = Some(k);
                        }
                    }
                }

                idx_1.map(|idx_1| {
                    (
                        idx_1,
                        idx_2,
                        k_v_12.unwrap_or_else(|| {
                            self.kernel.apply(&self.sv[idx_1].x, &self.sv[idx_2].x)
                        }),
                    )
                })
            }
            (Some(idx_1), Some(idx_2)) => Some((
                idx_1,
                idx_2,
                self.kernel.apply(&self.sv[idx_1].x, &self.sv[idx_2].x),
            )),
        }
    }

    fn smo(
        &mut self,
        idx_1: Option<usize>,
        idx_2: Option<usize>,
        tol: T,
        cache: &mut Cache<'_, T, M, K>,
    ) -> bool {
        match self.select_pair(idx_1, idx_2, cache) {
            Some((idx_1, idx_2, k_v_12)) => {
                let mut curv = self.sv[idx_1].k + self.sv[idx_2].k - T::two() * k_v_12;
                if curv <= T::zero() {
                    curv = self.tau;
                }

                let mut step = (self.sv[idx_2].grad - self.sv[idx_1].grad) / curv;

                if step >= T::zero() {
                    let mut ostep = self.sv[idx_1].alpha - self.sv[idx_1].cmin;
                    if ostep < step {
                        step = ostep;
                    }
                    ostep = self.sv[idx_2].cmax - self.sv[idx_2].alpha;
                    if ostep < step {
                        step = ostep;
                    }
                } else {
                    let mut ostep = self.sv[idx_2].cmin - self.sv[idx_2].alpha;
                    if ostep > step {
                        step = ostep;
                    }
                    ostep = self.sv[idx_1].alpha - self.sv[idx_1].cmax;
                    if ostep > step {
                        step = ostep;
                    }
                }

                self.update(idx_1, idx_2, step, cache);

                self.gmax - self.gmin > tol
            }
            None => false,
        }
    }

    fn update(&mut self, v1: usize, v2: usize, step: T, cache: &mut Cache<'_, T, M, K>) {
        self.sv[v1].alpha -= step;
        self.sv[v2].alpha += step;

        for i in 0..self.sv.len() {
            let k2 = cache.get(&self.sv[v2], &self.sv[i]);
            let k1 = cache.get(&self.sv[v1], &self.sv[i]);
            self.sv[i].grad -= step * (k2 - k1);
        }

        self.recalculate_minmax_grad = true;
        self.find_min_max_gradient();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::naive::dense_matrix::*;
    use crate::metrics::accuracy;
    #[cfg(feature = "serde")]
    use crate::svm::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn svc_fit_predict() {
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
            0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let y_hat = SVC::fit(
            &x,
            &y,
            SVCParameters::default()
                .with_c(200.0)
                .with_kernel(Kernels::linear()),
        )
        .and_then(|lr| lr.predict(&x))
        .unwrap();

        assert!(accuracy(&y_hat, &y) >= 0.9);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn svc_fit_predict_rbf() {
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
            -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1.,
        ];

        let y_hat = SVC::fit(
            &x,
            &y,
            SVCParameters::default()
                .with_c(1.0)
                .with_kernel(Kernels::rbf(0.7)),
        )
        .and_then(|lr| lr.predict(&x))
        .unwrap();

        assert!(accuracy(&y_hat, &y) >= 0.9);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "serde")]
    fn svc_serde() {
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
            -1., -1., -1., -1., -1., -1., -1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let svc = SVC::fit(&x, &y, Default::default()).unwrap();

        let deserialized_svc: SVC<f64, DenseMatrix<f64>, LinearKernel> =
            serde_json::from_str(&serde_json::to_string(&svc).unwrap()).unwrap();

        assert_eq!(svc, deserialized_svc);
    }
}
