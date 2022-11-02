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
//! use smartcore::linalg::basic::matrix::DenseMatrix;
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
//! let y = vec![ -1, -1, -1, -1, -1, -1, -1, -1,
//!            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
//!
//! let knl = Kernels::linear();
//! let params = &SVCParameters::default().with_c(200.0).with_kernel(&knl);
//! let svc = SVC::fit(&x, &y, params).unwrap();
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

use num::Bounded;
use rand::seq::SliceRandom;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{PredictorBorrow, SupervisedEstimatorBorrow};
use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, Array2, MutArray};
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;
use crate::rand_custom::get_rng_impl;
use crate::svm::Kernel;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// SVC Parameters
pub struct SVCParameters<
    'a,
    TX: Number + RealNumber,
    TY: Number + Ord,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of epochs.
    pub epoch: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Regularization parameter.
    pub c: TX,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Tolerance for stopping criterion.
    pub tol: TX,
    #[cfg_attr(feature = "serde", serde(skip_deserializing))]
    /// The kernel function.
    pub kernel: Option<&'a dyn Kernel<'a>>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Unused parameter.
    m: PhantomData<(X, Y, TY)>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Controls the pseudo random number generation for shuffling the data for probability estimates
    seed: Option<u64>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "TX: Serialize, TY: Serialize, X: Serialize, Y: Serialize",
        deserialize = "TX: Deserialize<'de>, TY: Deserialize<'de>, X: Deserialize<'de>, Y: Deserialize<'de>",
    ))
)]
/// Support Vector Classifier
pub struct SVC<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> {
    classes: Option<Vec<TY>>,
    instances: Option<Vec<Vec<TX>>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    parameters: Option<&'a SVCParameters<'a, TX, TY, X, Y>>,
    w: Option<Vec<TX>>,
    b: Option<TX>,
    phantomdata: PhantomData<(X, Y)>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
struct SupportVector<TX: Number + RealNumber> {
    index: usize,
    x: Vec<TX>,
    alpha: f64,
    grad: f64,
    cmin: f64,
    cmax: f64,
    k: f64,
}

struct Cache<TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> {
    data: HashMap<(usize, usize), f64>,
    phantom: PhantomData<(X, Y, TY, TX)>,
}

struct Optimizer<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> {
    x: &'a X,
    y: &'a Y,
    parameters: &'a SVCParameters<'a, TX, TY, X, Y>,
    svmin: usize,
    svmax: usize,
    gmin: TX,
    gmax: TX,
    tau: TX,
    sv: Vec<SupportVector<TX>>,
    recalculate_minmax_grad: bool,
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    SVCParameters<'a, TX, TY, X, Y>
{
    /// Number of epochs.
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = epoch;
        self
    }
    /// Regularization parameter.
    pub fn with_c(mut self, c: TX) -> Self {
        self.c = c;
        self
    }
    /// Tolerance for stopping criterion.
    pub fn with_tol(mut self, tol: TX) -> Self {
        self.tol = tol;
        self
    }
    /// The kernel function.
    pub fn with_kernel(mut self, kernel: &'a (dyn Kernel<'a>)) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Seed for the pseudo random number generator.
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> Default
    for SVCParameters<'a, TX, TY, X, Y>
{
    fn default() -> Self {
        SVCParameters {
            epoch: 2,
            c: TX::one(),
            tol: TX::from_f64(1e-3).unwrap(),
            kernel: Option::None,
            m: PhantomData,
            seed: Option::None,
        }
    }
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimatorBorrow<'a, X, Y, SVCParameters<'a, TX, TY, X, Y>> for SVC<'a, TX, TY, X, Y>
{
    fn new() -> Self {
        Self {
            classes: Option::None,
            instances: Option::None,
            parameters: Option::None,
            w: Option::None,
            b: Option::None,
            phantomdata: PhantomData,
        }
    }
    fn fit(
        x: &'a X,
        y: &'a Y,
        parameters: &'a SVCParameters<'a, TX, TY, X, Y>,
    ) -> Result<Self, Failed> {
        SVC::fit(x, y, parameters)
    }
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    PredictorBorrow<'a, X, TX> for SVC<'a, TX, TY, X, Y>
{
    fn predict(&self, x: &'a X) -> Result<Vec<TX>, Failed> {
        Ok(self.predict(x).unwrap())
    }
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX> + 'a, Y: Array1<TY> + 'a>
    SVC<'a, TX, TY, X, Y>
{
    /// Fits SVC to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - class labels
    /// * `parameters` - optional parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(
        x: &'a X,
        y: &'a Y,
        parameters: &'a SVCParameters<'a, TX, TY, X, Y>,
    ) -> Result<SVC<'a, TX, TY, X, Y>, Failed> {
        let (n, _) = x.shape();

        if parameters.kernel.is_none() {
            return Err(Failed::because(
                FailedError::ParametersError,
                "kernel should be defined at this point, please use `with_kernel()`",
            ));
        }

        if n != y.shape() {
            return Err(Failed::fit(
                "Number of rows of X doesn\'t match number of rows of Y",
            ));
        }

        let classes = y.unique();

        if classes.len() != 2 {
            return Err(Failed::fit(&format!(
                "Incorrect number of classes: {}",
                classes.len()
            )));
        }

        // Make sure class labels are either 1 or -1
        for e in y.iterator(0) {
            let y_v = e.to_i32().unwrap();
            if y_v != -1 && y_v != 1 {
                return Err(Failed::because(
                    FailedError::ParametersError,
                    "Class labels must be 1 or -1",
                ));
            }
        }

        let optimizer: Optimizer<'_, TX, TY, X, Y> = Optimizer::new(x, y, parameters);

        let (support_vectors, weight, b) = optimizer.optimize();

        Ok(SVC::<'a> {
            classes: Some(classes),
            instances: Some(support_vectors),
            parameters: Some(parameters),
            w: Some(weight),
            b: Some(b),
            phantomdata: PhantomData,
        })
    }

    /// Predicts estimated class labels from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &'a X) -> Result<Vec<TX>, Failed> {
        let mut y_hat: Vec<TX> = self.decision_function(x)?;

        for i in 0..y_hat.len() {
            let cls_idx = match *y_hat.get(i).unwrap() > TX::zero() {
                false => TX::from(self.classes.as_ref().unwrap()[0]).unwrap(),
                true => TX::from(self.classes.as_ref().unwrap()[1]).unwrap(),
            };

            y_hat.set(i, cls_idx);
        }

        Ok(y_hat)
    }

    /// Evaluates the decision function for the rows in `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn decision_function(&self, x: &'a X) -> Result<Vec<TX>, Failed> {
        let (n, _) = x.shape();
        let mut y_hat: Vec<TX> = Array1::zeros(n);

        for i in 0..n {
            let row_pred: TX =
                self.predict_for_row(Vec::from_iterator(x.get_row(i).iterator(0).copied(), n));
            y_hat.set(i, row_pred);
        }

        Ok(y_hat)
    }

    fn predict_for_row(&self, x: Vec<TX>) -> TX {
        let mut f = self.b.unwrap();

        for i in 0..self.instances.as_ref().unwrap().len() {
            f += self.w.as_ref().unwrap()[i]
                * TX::from(
                    self.parameters
                        .as_ref()
                        .unwrap()
                        .kernel
                        .as_ref()
                        .unwrap()
                        .apply(
                            &x.iter().map(|e| e.to_f64().unwrap()).collect(),
                            &self.instances.as_ref().unwrap()[i]
                                .iter()
                                .map(|e| e.to_f64().unwrap())
                                .collect(),
                        )
                        .unwrap(),
                )
                .unwrap();
        }

        f
    }
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for SVC<'a, TX, TY, X, Y>
{
    fn eq(&self, other: &Self) -> bool {
        if (self.b.unwrap().sub(other.b.unwrap())).abs() > TX::epsilon() * TX::two()
            || self.w.as_ref().unwrap().len() != other.w.as_ref().unwrap().len()
            || self.instances.as_ref().unwrap().len() != other.instances.as_ref().unwrap().len()
        {
            false
        } else {
            if !self
                .w
                .as_ref()
                .unwrap()
                .approximate_eq(other.w.as_ref().unwrap(), TX::epsilon())
            {
                return false;
            }
            for i in 0..self.w.as_ref().unwrap().len() {
                if (self.w.as_ref().unwrap()[i].sub(other.w.as_ref().unwrap()[i])).abs()
                    > TX::epsilon()
                {
                    return false;
                }
            }
            for i in 0..self.instances.as_ref().unwrap().len() {
                if !(self.instances.as_ref().unwrap()[i] == other.instances.as_ref().unwrap()[i]) {
                    return false;
                }
            }
            true
        }
    }
}

impl<TX: Number + RealNumber> SupportVector<TX> {
    fn new(i: usize, x: Vec<TX>, y: TX, g: f64, c: f64, k_v: f64) -> SupportVector<TX> {
        let (cmin, cmax) = if y > TX::zero() {
            (0f64, c)
        } else {
            (-c, 0f64)
        };
        SupportVector {
            index: i,
            x,
            grad: g,
            k: k_v,
            alpha: 0f64,
            cmin,
            cmax,
        }
    }
}

impl<TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> Cache<TX, TY, X, Y> {
    fn new() -> Cache<TX, TY, X, Y> {
        Cache {
            data: HashMap::new(),
            phantom: PhantomData,
        }
    }

    fn get(&mut self, i: &SupportVector<TX>, j: &SupportVector<TX>, or_insert: f64) -> f64 {
        let idx_i = i.index;
        let idx_j = j.index;
        #[allow(clippy::or_fun_call)]
        let entry = self.data.entry((idx_i, idx_j)).or_insert(or_insert);
        *entry
    }

    fn insert(&mut self, key: (usize, usize), value: f64) {
        self.data.insert(key, value);
    }

    fn drop(&mut self, idxs_to_drop: HashSet<usize>) {
        self.data.retain(|k, _| !idxs_to_drop.contains(&k.0));
    }
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    Optimizer<'a, TX, TY, X, Y>
{
    fn new(
        x: &'a X,
        y: &'a Y,
        parameters: &'a SVCParameters<'a, TX, TY, X, Y>,
    ) -> Optimizer<'a, TX, TY, X, Y> {
        let (n, _) = x.shape();

        Optimizer {
            x,
            y,
            parameters,
            svmin: 0,
            svmax: 0,
            gmin: <TX as Bounded>::max_value(),
            gmax: <TX as Bounded>::min_value(),
            tau: TX::from_f64(1e-12).unwrap(),
            sv: Vec::with_capacity(n),
            recalculate_minmax_grad: true,
        }
    }

    fn optimize(mut self) -> (Vec<Vec<TX>>, Vec<TX>, TX) {
        let (n, _) = self.x.shape();

        let mut cache: Cache<TX, TY, X, Y> = Cache::new();

        self.initialize(&mut cache);

        let tol = self.parameters.tol;
        let good_enough = TX::from_i32(1000).unwrap();

        for _ in 0..self.parameters.epoch {
            for i in self.permutate(n) {
                self.process(
                    i,
                    Vec::from_iterator(self.x.get_row(i).iterator(0).copied(), n),
                    *self.y.get(i),
                    &mut cache,
                );
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

        let mut support_vectors: Vec<Vec<TX>> = Vec::new();
        let mut w: Vec<TX> = Vec::new();

        let b = (self.gmax + self.gmin) / TX::two();

        for v in self.sv {
            support_vectors.push(v.x);
            w.push(TX::from(v.alpha).unwrap());
        }

        (support_vectors, w, b)
    }

    fn initialize(&mut self, cache: &mut Cache<TX, TY, X, Y>) {
        let (n, _) = self.x.shape();
        let few = 5;
        let mut cp = 0;
        let mut cn = 0;

        for i in self.permutate(n) {
            if *self.y.get(i) == TY::one() && cp < few {
                if self.process(
                    i,
                    Vec::from_iterator(self.x.get_row(i).iterator(0).copied(), n),
                    *self.y.get(i),
                    cache,
                ) {
                    cp += 1;
                }
            } else if *self.y.get(i) == TY::from(-1).unwrap()
                && cn < few
                && self.process(
                    i,
                    Vec::from_iterator(self.x.get_row(i).iterator(0).copied(), n),
                    *self.y.get(i),
                    cache,
                )
            {
                cn += 1;
            }

            if cp >= few && cn >= few {
                break;
            }
        }
    }

    fn process(&mut self, i: usize, x: Vec<TX>, y: TY, cache: &mut Cache<TX, TY, X, Y>) -> bool {
        for j in 0..self.sv.len() {
            if self.sv[j].index == i {
                return true;
            }
        }

        let mut g: f64 = y.to_f64().unwrap();

        let mut cache_values: Vec<((usize, usize), TX)> = Vec::new();

        for v in self.sv.iter() {
            let k = self
                .parameters
                .kernel
                .as_ref()
                .unwrap()
                .apply(
                    &v.x.iter().map(|e| e.to_f64().unwrap()).collect(),
                    &x.iter().map(|e| e.to_f64().unwrap()).collect(),
                )
                .unwrap();
            cache_values.push(((i, v.index), TX::from(k).unwrap()));
            g -= v.alpha * k;
        }

        self.find_min_max_gradient();

        if self.gmin < self.gmax
            && ((y > TY::zero() && g < self.gmin.to_f64().unwrap())
                || (y < TY::zero() && g > self.gmax.to_f64().unwrap()))
        {
            return false;
        }

        for v in cache_values {
            cache.insert(v.0, v.1.to_f64().unwrap());
        }

        let x_f64 = x.iter().map(|e| e.to_f64().unwrap()).collect();
        let k_v = self
            .parameters
            .kernel
            .as_ref()
            .expect("Kernel should be defined at this point, use with_kernel() on parameters")
            .apply(&x_f64, &x_f64)
            .unwrap();

        self.sv.insert(
            0,
            SupportVector::<TX>::new(
                i,
                x.to_vec(),
                TX::from(y).unwrap(),
                g,
                self.parameters.c.to_f64().unwrap(),
                k_v,
            ),
        );

        if y > TY::zero() {
            self.smo(None, Some(0), TX::zero(), cache);
        } else {
            self.smo(Some(0), None, TX::zero(), cache);
        }

        true
    }

    fn reprocess(&mut self, tol: TX, cache: &mut Cache<TX, TY, X, Y>) -> bool {
        let status = self.smo(None, None, tol, cache);
        self.clean(cache);
        status
    }

    fn finish(&mut self, cache: &mut Cache<TX, TY, X, Y>) {
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

        self.gmin = <TX as Bounded>::max_value();
        self.gmax = <TX as Bounded>::min_value();

        for i in 0..self.sv.len() {
            let v = &self.sv[i];
            let g = v.grad;
            let a = v.alpha;
            if g < self.gmin.to_f64().unwrap() && a > v.cmin {
                self.gmin = TX::from(g).unwrap();
                self.svmin = i;
            }
            if g > self.gmax.to_f64().unwrap() && a < v.cmax {
                self.gmax = TX::from(g).unwrap();
                self.svmax = i;
            }
        }

        self.recalculate_minmax_grad = false
    }

    fn clean(&mut self, cache: &mut Cache<TX, TY, X, Y>) {
        self.find_min_max_gradient();

        let gmax = self.gmax;
        let gmin = self.gmin;

        let mut idxs_to_drop: HashSet<usize> = HashSet::new();

        self.sv.retain(|v| {
            if v.alpha == 0f64
                && ((TX::from(v.grad).unwrap() >= gmax && TX::zero() >= TX::from(v.cmax).unwrap())
                    || (TX::from(v.grad).unwrap() <= gmin
                        && TX::zero() <= TX::from(v.cmin).unwrap()))
            {
                idxs_to_drop.insert(v.index);
                return false;
            };
            true
        });

        cache.drop(idxs_to_drop);
        self.recalculate_minmax_grad = true;
    }

    fn permutate(&self, n: usize) -> Vec<usize> {
        let mut rng = get_rng_impl(self.parameters.seed);
        let mut range: Vec<usize> = (0..n).collect();
        range.shuffle(&mut rng);
        range
    }

    fn select_pair(
        &mut self,
        idx_1: Option<usize>,
        idx_2: Option<usize>,
        cache: &mut Cache<TX, TY, X, Y>,
    ) -> Option<(usize, usize, f64)> {
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
                let mut best = 0f64;
                for i in 0..self.sv.len() {
                    let v = &self.sv[i];
                    let z = v.grad - gm;
                    let k = cache.get(
                        sv1,
                        v,
                        self.parameters
                            .kernel
                            .as_ref()
                            .unwrap()
                            .apply(
                                &sv1.x.iter().map(|e| e.to_f64().unwrap()).collect(),
                                &v.x.iter().map(|e| e.to_f64().unwrap()).collect(),
                            )
                            .unwrap(),
                    );
                    let mut curv = km + v.k - 2f64 * k;
                    if curv <= 0f64 {
                        curv = self.tau.to_f64().unwrap();
                    }
                    let mu = z / curv;
                    if (mu > 0f64 && v.alpha < v.cmax) || (mu < 0f64 && v.alpha > v.cmin) {
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
                            self.parameters
                                .kernel
                                .as_ref()
                                .unwrap()
                                .apply(
                                    &self.sv[idx_1]
                                        .x
                                        .iter()
                                        .map(|e| e.to_f64().unwrap())
                                        .collect(),
                                    &self.sv[idx_2]
                                        .x
                                        .iter()
                                        .map(|e| e.to_f64().unwrap())
                                        .collect(),
                                )
                                .unwrap()
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
                let mut best = 0f64;
                for i in 0..self.sv.len() {
                    let v = &self.sv[i];
                    let z = gm - v.grad;
                    let k = cache.get(
                        sv2,
                        v,
                        self.parameters
                            .kernel
                            .as_ref()
                            .unwrap()
                            .apply(
                                &sv2.x.iter().map(|e| e.to_f64().unwrap()).collect(),
                                &v.x.iter().map(|e| e.to_f64().unwrap()).collect(),
                            )
                            .unwrap(),
                    );
                    let mut curv = km + v.k - 2f64 * k;
                    if curv <= 0f64 {
                        curv = self.tau.to_f64().unwrap();
                    }

                    let mu = z / curv;
                    if (mu > 0f64 && v.alpha > v.cmin) || (mu < 0f64 && v.alpha < v.cmax) {
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
                            self.parameters
                                .kernel
                                .as_ref()
                                .unwrap()
                                .apply(
                                    &self.sv[idx_1]
                                        .x
                                        .iter()
                                        .map(|e| e.to_f64().unwrap())
                                        .collect(),
                                    &self.sv[idx_2]
                                        .x
                                        .iter()
                                        .map(|e| e.to_f64().unwrap())
                                        .collect(),
                                )
                                .unwrap()
                        }),
                    )
                })
            }
            (Some(idx_1), Some(idx_2)) => Some((
                idx_1,
                idx_2,
                self.parameters
                    .kernel
                    .as_ref()
                    .unwrap()
                    .apply(
                        &self.sv[idx_1]
                            .x
                            .iter()
                            .map(|e| e.to_f64().unwrap())
                            .collect(),
                        &self.sv[idx_2]
                            .x
                            .iter()
                            .map(|e| e.to_f64().unwrap())
                            .collect(),
                    )
                    .unwrap(),
            )),
        }
    }

    fn smo(
        &mut self,
        idx_1: Option<usize>,
        idx_2: Option<usize>,
        tol: TX,
        cache: &mut Cache<TX, TY, X, Y>,
    ) -> bool {
        match self.select_pair(idx_1, idx_2, cache) {
            Some((idx_1, idx_2, k_v_12)) => {
                let mut curv = self.sv[idx_1].k + self.sv[idx_2].k - 2f64 * k_v_12;
                if curv <= 0f64 {
                    curv = self.tau.to_f64().unwrap();
                }

                let mut step = (self.sv[idx_2].grad - self.sv[idx_1].grad) / curv;

                if step >= 0f64 {
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

                self.update(idx_1, idx_2, TX::from(step).unwrap(), cache);

                self.gmax - self.gmin > tol
            }
            None => false,
        }
    }

    fn update(&mut self, v1: usize, v2: usize, step: TX, cache: &mut Cache<TX, TY, X, Y>) {
        self.sv[v1].alpha -= step.to_f64().unwrap();
        self.sv[v2].alpha += step.to_f64().unwrap();

        for i in 0..self.sv.len() {
            let k2 = cache.get(
                &self.sv[v2],
                &self.sv[i],
                self.parameters
                    .kernel
                    .as_ref()
                    .unwrap()
                    .apply(
                        &self.sv[v2].x.iter().map(|e| e.to_f64().unwrap()).collect(),
                        &self.sv[i].x.iter().map(|e| e.to_f64().unwrap()).collect(),
                    )
                    .unwrap(),
            );
            let k1 = cache.get(
                &self.sv[v1],
                &self.sv[i],
                self.parameters
                    .kernel
                    .as_ref()
                    .unwrap()
                    .apply(
                        &self.sv[v1].x.iter().map(|e| e.to_f64().unwrap()).collect(),
                        &self.sv[i].x.iter().map(|e| e.to_f64().unwrap()).collect(),
                    )
                    .unwrap(),
            );
            self.sv[i].grad -= step.to_f64().unwrap() * (k2 - k1);
        }

        self.recalculate_minmax_grad = true;
        self.find_min_max_gradient();
    }
}

#[cfg(test)]
mod tests {
    use num::ToPrimitive;

    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
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

        let y: Vec<i32> = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let knl = Kernels::linear();
        let params = SVCParameters::default()
            .with_c(200.0)
            .with_kernel(&knl)
            .with_seed(Some(100));

        let y_hat = SVC::fit(&x, &y, &params)
            .and_then(|lr| lr.predict(&x))
            .unwrap();
        let acc = accuracy(&y, &(y_hat.iter().map(|e| e.to_i32().unwrap()).collect()));

        assert!(
            acc >= 0.9,
            "accuracy ({}) is not larger or equal to 0.9",
            acc
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn svc_fit_decision_function() {
        let x = DenseMatrix::from_2d_array(&[&[4.0, 0.0], &[0.0, 4.0], &[8.0, 0.0], &[0.0, 8.0]]);

        let x2 = DenseMatrix::from_2d_array(&[
            &[3.0, 3.0],
            &[4.0, 4.0],
            &[6.0, 6.0],
            &[10.0, 10.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
        ]);

        let y: Vec<i32> = vec![-1, -1, 1, 1];

        let y_hat = SVC::fit(
            &x,
            &y,
            &SVCParameters::default()
                .with_c(200.0)
                .with_kernel(&Kernels::linear()),
        )
        .and_then(|lr| lr.decision_function(&x2))
        .unwrap();

        // x can be classified by a straight line through [6.0, 0.0] and [0.0, 6.0],
        // so the score should increase as points get further away from that line
        assert!(y_hat[1] < y_hat[2]);
        assert!(y_hat[2] < y_hat[3]);

        // for negative scores the score should decrease
        assert!(y_hat[4] > y_hat[5]);

        // y_hat[0] is on the line, so its score should be close to 0
        assert!(num::Float::abs(y_hat[0]) <= 0.1);
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

        let y: Vec<i32> = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let y_hat = SVC::fit(
            &x,
            &y,
            &SVCParameters::default()
                .with_c(1.0)
                .with_kernel(&Kernels::rbf().with_gamma(0.7)),
        )
        .and_then(|lr| lr.predict(&x))
        .unwrap();

        let acc = accuracy(&y, &(y_hat.iter().map(|e| e.to_i32().unwrap()).collect()));

        assert!(
            acc >= 0.9,
            "accuracy ({}) is not larger or equal to 0.9",
            acc
        );
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

        let y: Vec<i32> = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let knl = Kernels::linear();
        let params = SVCParameters::default().with_kernel(&knl);
        let svc = SVC::fit(&x, &y, &params).unwrap();

        // serialization
        let serialized_svc = &serde_json::to_string(&svc).unwrap();

        println!("{:?}", serialized_svc);
    }
}
