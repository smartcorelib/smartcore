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
//! To solve this optimization problem, `smartcore` uses an [approximate SVM solver](https://leon.bottou.org/projects/lasvm).
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
//!         ]).unwrap();
//! let y = vec![ -1, -1, -1, -1, -1, -1, -1, -1,
//!            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
//!
//! let knl = Kernels::linear();
//! let parameters = &SVCParameters::default().with_c(200.0).with_kernel(knl);
//! let svc = SVC::fit(&x, &y, parameters).unwrap();
//!
//! let y_hat = svc.predict(&x).unwrap();
//!
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
use crate::linalg::basic::arrays::{Array, Array1, Array2, MutArray};
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;
use crate::rand_custom::get_rng_impl;
use crate::svm::Kernel;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// Configuration for a multi-class Support Vector Machine (SVM) classifier.
/// This struct holds the indices of the data points relevant to a specific binary
/// classification problem within a multi-class context, and the two classes
/// being discriminated.
struct MultiClassConfig<TY: Number + Ord> {
    /// The indices of the data points from the original dataset that belong to the two `classes`.
    indices: Vec<usize>,
    /// A tuple representing the two classes that this configuration is designed to distinguish.
    classes: (TY, TY),
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    SupervisedEstimatorBorrow<'a, X, Y, SVCParameters<TX, TY, X, Y>>
    for MultiClassSVC<'a, TX, TY, X, Y>
{
    /// Creates a new, empty `MultiClassSVC` instance.
    fn new() -> Self {
        Self {
            classifiers: Option::None,
        }
    }

    /// Fits the `MultiClassSVC` model to the provided data and parameters.
    ///
    /// This method delegates the fitting process to the inherent `MultiClassSVC::fit` method.
    ///
    /// # Arguments
    /// * `x` - A reference to the input features (2D array).
    /// * `y` - A reference to the target labels (1D array).
    /// * `parameters` - A reference to the `SVCParameters` controlling the SVM training.
    ///
    /// # Returns
    /// A `Result` indicating success (`Self`) or failure (`Failed`).
    fn fit(
        x: &'a X,
        y: &'a Y,
        parameters: &'a SVCParameters<TX, TY, X, Y>,
    ) -> Result<Self, Failed> {
        MultiClassSVC::fit(x, y, parameters)
    }
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    PredictorBorrow<'a, X, TX> for MultiClassSVC<'a, TX, TY, X, Y>
{
    /// Predicts the class labels for new data points.
    ///
    /// This method delegates the prediction process to the inherent `MultiClassSVC::predict` method.
    ///
    /// # Arguments
    /// * `x` - A reference to the input features (2D array) for which to make predictions.
    ///
    /// # Returns
    /// A `Result` containing a `Vec` of predicted class labels (`TX`) or a `Failed` error.
    fn predict(&self, x: &'a X) -> Result<Vec<TX>, Failed> {
        Ok(self.predict(x).unwrap())
    }
}

/// A multi-class Support Vector Machine (SVM) classifier.
///
/// This struct implements a multi-class SVM using the "one-vs-one" strategy,
/// where a separate binary SVC classifier is trained for every pair of classes.
///
/// # Type Parameters
/// * `'a` - Lifetime parameter for borrowed data.
/// * `TX` - The numeric type of the input features (must implement `Number` and `RealNumber`).
/// * `TY` - The numeric type of the target labels (must implement `Number` and `Ord`).
/// * `X` - The type representing the 2D array of input features (e.g., a matrix).
/// * `Y` - The type representing the 1D array of target labels (e.g., a vector).
pub struct MultiClassSVC<
    'a,
    TX: Number + RealNumber,
    TY: Number + Ord,
    X: Array2<TX>,
    Y: Array1<TY>,
> {
    /// An optional vector of binary `SVC` classifiers.
    classifiers: Option<Vec<SVC<'a, TX, TY, X, Y>>>,
}

impl<'a, TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    MultiClassSVC<'a, TX, TY, X, Y>
{
    /// Fits the `MultiClassSVC` model to the provided data using a one-vs-one strategy.
    ///
    /// This method identifies all unique classes in the target labels `y` and then
    /// trains a binary `SVC` for every unique pair of classes. For each pair, it
    /// extracts the relevant data points and their labels, and then trains a
    /// specialized `SVC` for that binary classification task.
    ///
    /// # Arguments
    /// * `x` - A reference to the input features (2D array).
    /// * `y` - A reference to the target labels (1D array).
    /// * `parameters` - A reference to the `SVCParameters` controlling the SVM training for each individual binary classifier.
    ///  
    ///
    /// # Returns
    /// A `Result` indicating success (`MultiClassSVC`) or failure (`Failed`).
    pub fn fit(
        x: &'a X,
        y: &'a Y,
        parameters: &'a SVCParameters<TX, TY, X, Y>,
    ) -> Result<MultiClassSVC<'a, TX, TY, X, Y>, Failed> {
        let unique_classes = y.unique();
        let mut classifiers = Vec::new();
        // Iterate through all unique pairs of classes (one-vs-one strategy)
        for i in 0..unique_classes.len() {
            for j in i..unique_classes.len() {
                if i == j {
                    continue;
                }
                let class0 = unique_classes[j];
                let class1 = unique_classes[i];

                let mut indices = Vec::new();
                // Collect indices of data points belonging to the current pair of classes
                for (index, v) in y.iterator(0).enumerate() {
                    if *v == class0 || *v == class1 {
                        indices.push(index)
                    }
                }
                let classes = (class0, class1);
                let multiclass_config = MultiClassConfig { classes, indices };
                // Fit a binary SVC for the current pair of classes
                let svc = SVC::multiclass_fit(x, y, parameters, multiclass_config).unwrap();
                classifiers.push(svc);
            }
        }
        Ok(Self {
            classifiers: Some(classifiers),
        })
    }

    /// Predicts the class labels for new data points using the trained multi-class SVM.
    ///
    /// This method uses a "voting" scheme (majority vote) among all the binary
    /// classifiers to determine the final prediction for each data point.
    ///
    /// # Arguments
    /// * `x` - A reference to the input features (2D array) for which to make predictions.
    ///
    /// # Returns
    /// A `Result` containing a `Vec` of predicted class labels (`TX`) or a `Failed` error.
    ///
    pub fn predict(&self, x: &X) -> Result<Vec<TX>, Failed> {
        // Initialize a HashMap for each data point to store votes for each class
        let mut polls = vec![HashMap::new(); x.shape().0];
        // Retrieve the trained binary classifiers
        let classifiers = self.classifiers.as_ref().unwrap();

        // Iterate through each binary classifier
        for i in 0..classifiers.len() {
            let svc = classifiers.get(i).unwrap();
            let predictions = svc.predict(x).unwrap(); // call SVC::predict for each binary classifier

            // For each prediction from the current binary classifier
            for (j, prediction) in predictions.iter().enumerate() {
                let prediction = prediction.to_i32().unwrap();
                let poll = polls.get_mut(j).unwrap(); // Get the poll for the current data point
                                                      // Increment the vote for the predicted class
                if let Some(count) = poll.get_mut(&prediction) {
                    *count += 1
                } else {
                    poll.insert(prediction, 1);
                }
            }
        }

        // Determine the final prediction for each data point based on majority vote
        Ok(polls
            .iter()
            .map(|v| {
                // Find the class with the maximum votes for each data point
                TX::from(*v.iter().max_by_key(|(_, class)| *class).unwrap().0).unwrap()
            })
            .collect())
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
/// SVC Parameters
pub struct SVCParameters<TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> {
    /// Number of epochs.
    pub epoch: usize,
    /// Regularization parameter.
    pub c: TX,
    /// Tolerance for stopping criterion.
    pub tol: TX,
    /// The kernel function.
    #[cfg_attr(
        all(feature = "serde", target_arch = "wasm32"),
        serde(skip_serializing, skip_deserializing)
    )]
    pub kernel: Option<Box<dyn Kernel>>,
    /// Unused parameter.
    m: PhantomData<(X, Y, TY)>,
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
    classes: Option<(TY, TY)>,
    instances: Option<Vec<Vec<TX>>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    parameters: Option<&'a SVCParameters<TX, TY, X, Y>>,
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
    indices: Option<Vec<usize>>,
    parameters: &'a SVCParameters<TX, TY, X, Y>,
    classes: &'a (TY, TY),
    svmin: usize,
    svmax: usize,
    gmin: TX,
    gmax: TX,
    tau: TX,
    sv: Vec<SupportVector<TX>>,
    recalculate_minmax_grad: bool,
}

impl<TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>>
    SVCParameters<TX, TY, X, Y>
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
    pub fn with_kernel<K: Kernel + 'static>(mut self, kernel: K) -> Self {
        self.kernel = Some(Box::new(kernel));
        self
    }
    /// Seed for the pseudo random number generator.
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.seed = seed;
        self
    }
}

impl<TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> Default
    for SVCParameters<TX, TY, X, Y>
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
    SupervisedEstimatorBorrow<'a, X, Y, SVCParameters<TX, TY, X, Y>> for SVC<'a, TX, TY, X, Y>
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
        parameters: &'a SVCParameters<TX, TY, X, Y>,
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
    /// Fits a binary Support Vector Classifier (SVC) to the provided data.
    ///
    /// This is the primary `fit` method for a standalone binary SVC. It expects
    /// the target labels `y` to contain exactly two unique classes. If more or
    /// fewer than two classes are found, it returns an error. It then extracts
    /// these two classes and proceeds to optimize and fit the SVC model.
    ///
    /// # Arguments
    /// * `x` - A reference to the input features (2D array) of the training data.
    /// * `y` - A reference to the target labels (1D array) of the training data. `y` must contain exactly two unique class labels.
    /// * `parameters` - A reference to the `SVCParameters` controlling the training process.
    ///
    /// # Returns
    /// A `Result` which is:
    /// - `Ok(SVC<'a, TX, TY, X, Y>)`: A new, fitted binary SVC instance.
    /// - `Err(Failed)`: If the number of unique classes in `y` is not exactly two, or if the underlying optimization fails.
    pub fn fit(
        x: &'a X,
        y: &'a Y,
        parameters: &'a SVCParameters<TX, TY, X, Y>,
    ) -> Result<SVC<'a, TX, TY, X, Y>, Failed> {
        let classes = y.unique();
        // Validate that there are exactly two unique classes in the target labels.
        if classes.len() != 2 {
            return Err(Failed::fit(&format!(
                "Incorrect number of classes: {}. A binary SVC requires exactly two classes.",
                classes.len()
            )));
        }
        let classes = (classes[0], classes[1]);
        let svc = Self::optimize_and_fit(x, y, parameters, classes, None);
        svc
    }

    /// Fits a binary Support Vector Classifier (SVC) specifically for multi-class scenarios.
    ///
    /// This function is intended to be called by a multi-class strategy (e.g., one-vs-one)
    /// to train individual binary SVCs. It takes a `MultiClassConfig` which specifies
    /// the two classes this SVC should discriminate and the subset of data indices
    /// relevant to these classes. It then delegates the actual optimization and fitting
    /// to `optimize_and_fit`.
    ///
    /// # Arguments
    /// * `x` - A reference to the input features (2D array) of the training data.
    /// * `y` - A reference to the target labels (1D array) of the training data.
    /// * `parameters` - A reference to the `SVCParameters` controlling the training process (e.g., kernel, C-value, tolerance).
    /// * `multiclass_config` - A `MultiClassConfig` struct containing:
    ///     - `classes`: A tuple `(class0, class1)` specifying the two classes this SVC should distinguish.
    ///     - `indices`: A `Vec<usize>` containing the indices of the data points in `x` and `y that belong to either `class0` or `class1`.`
    ///
    /// # Returns
    /// A `Result` which is:
    /// - `Ok(SVC<'a, TX, TY, X, Y>)`: A new, fitted binary SVC instance.
    /// - `Err(Failed)`: If the fitting process encounters an error (e.g., invalid parameters).
    fn multiclass_fit(
        x: &'a X,
        y: &'a Y,
        parameters: &'a SVCParameters<TX, TY, X, Y>,
        multiclass_config: MultiClassConfig<TY>,
    ) -> Result<SVC<'a, TX, TY, X, Y>, Failed> {
        let classes = multiclass_config.classes;
        let indices = multiclass_config.indices;
        let svc = Self::optimize_and_fit(x, y, parameters, classes, Some(indices));
        svc
    }

    /// Internal function to optimize and fit the Support Vector Classifier.
    ///
    /// This is the core logic for training a binary SVC. It performs several checks
    /// (e.g., kernel presence, data shape consistency) and then initializes an
    /// `Optimizer` to find the support vectors, weights (`w`), and bias (`b`).
    ///
    /// # Arguments
    /// * `x` - A reference to the input features (2D array) of the training data.
    /// * `y` - A reference to the target labels (1D array) of the training data.
    /// * `parameters` - A reference to the `SVCParameters` defining the SVM model's configuration.
    /// * `classes` - A tuple `(class0, class1)` representing the two distinct class labels that the SVC will learn to separate.
    /// * `indices` - An `Option<Vec<usize>>`. If `Some`, it contains the specific indices of data points from `x` and `y` that should be used for training this binary classifier. If `None`, all data points in `x` and `y` are considered.
    /// # Returns
    /// A `Result` which is:
    /// - `Ok(SVC<'a, TX, TY, X, Y>)`: A new `SVC` instance populated with the learned model components (support vectors, weights, bias).
    /// - `Err(Failed)`: If any of the validation checks fail (e.g., missing kernel, mismatched data shapes), or if the optimization process fails.
    fn optimize_and_fit(
        x: &'a X,
        y: &'a Y,
        parameters: &'a SVCParameters<TX, TY, X, Y>,
        classes: (TY, TY),
        indices: Option<Vec<usize>>,
    ) -> Result<SVC<'a, TX, TY, X, Y>, Failed> {
        let (n_samples, _) = x.shape();

        // Validate that a kernel has been defined in the parameters.
        if parameters.kernel.is_none() {
            return Err(Failed::because(
                FailedError::ParametersError,
                "kernel should be defined at this point, please use `with_kernel()`",
            ));
        }

        // Validate that the number of samples in X matches the number of labels in Y.
        if n_samples != y.shape() {
            return Err(Failed::fit(
                "Number of rows of X doesn't match number of rows of Y",
            ));
        }

        let optimizer: Optimizer<'_, TX, TY, X, Y> =
            Optimizer::new(x, y, indices, parameters, &classes);

        // Perform the optimization to find the support vectors, weight vector, and bias.
        // This is where the core SVM algorithm (e.g., SMO) would run.
        let (support_vectors, weight, b) = optimizer.optimize();

        // Construct and return the fitted SVC model.
        Ok(SVC::<'a> {
            classes: Some(classes), // Store the two classes the SVC was trained on.
            instances: Some(support_vectors), // Store the data points that are support vectors.
            parameters: Some(parameters), // Reference to the parameters used for fitting.
            w: Some(weight),        // The learned weight vector (for linear kernels).
            b: Some(b),             // The learned bias term.
            phantomdata: PhantomData, // Placeholder for type parameters not directly stored.
        })
    }
    /// Predicts estimated class labels from `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn predict(&self, x: &'a X) -> Result<Vec<TX>, Failed> {
        let mut y_hat: Vec<TX> = self.decision_function(x)?;

        for i in 0..y_hat.len() {
            let cls_idx = match *y_hat.get(i) > TX::zero() {
                false => TX::from(self.classes.as_ref().unwrap().0).unwrap(),
                true => TX::from(self.classes.as_ref().unwrap().1).unwrap(),
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

        let mut row = Vec::with_capacity(n);
        for i in 0..n {
            row.clear();
            row.extend(x.get_row(i).iterator(0).copied());
            let row_pred: TX = self.predict_for_row(&row);
            y_hat.set(i, row_pred);
        }

        Ok(y_hat)
    }

    fn predict_for_row(&self, x: &[TX]) -> TX {
        let mut f = self.b.unwrap();

        let xi: Vec<_> = x.iter().map(|e| e.to_f64().unwrap()).collect();
        for i in 0..self.instances.as_ref().unwrap().len() {
            let xj: Vec<_> = self.instances.as_ref().unwrap()[i]
                .iter()
                .map(|e| e.to_f64().unwrap())
                .collect();
            f += self.w.as_ref().unwrap()[i]
                * TX::from(
                    self.parameters
                        .as_ref()
                        .unwrap()
                        .kernel
                        .as_ref()
                        .unwrap()
                        .apply(&xi, &xj)
                        .unwrap(),
                )
                .unwrap();
        }

        f
    }
}

impl<TX: Number + RealNumber, TY: Number + Ord, X: Array2<TX>, Y: Array1<TY>> PartialEq
    for SVC<'_, TX, TY, X, Y>
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
        indices: Option<Vec<usize>>,
        parameters: &'a SVCParameters<TX, TY, X, Y>,
        classes: &'a (TY, TY),
    ) -> Optimizer<'a, TX, TY, X, Y> {
        let (n, _) = x.shape();

        Optimizer {
            x,
            y,
            indices,
            parameters,
            classes,
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

        let mut x = Vec::with_capacity(n);
        for _ in 0..self.parameters.epoch {
            for i in self.permutate(n) {
                x.clear();
                x.extend(self.x.get_row(i).iterator(0).take(n).copied());
                let y = if *self.y.get(i) == self.classes.1 {
                    1
                } else {
                    -1
                } as f64;
                self.process(i, &x, y, &mut cache);
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

        let mut x = Vec::with_capacity(n);
        for i in self.permutate(n) {
            x.clear();
            x.extend(self.x.get_row(i).iterator(0).take(n).copied());
            let y = if *self.y.get(i) == self.classes.1 {
                1
            } else {
                -1
            } as f64;
            if y == 1.0 && cp < few {
                if self.process(i, &x, y, cache) {
                    cp += 1;
                }
            } else if y == -1.0 && cn < few && self.process(i, &x, y, cache) {
                cn += 1;
            }

            if cp >= few && cn >= few {
                break;
            }
        }
    }

    fn process(&mut self, i: usize, x: &[TX], y: f64, cache: &mut Cache<TX, TY, X, Y>) -> bool {
        for j in 0..self.sv.len() {
            if self.sv[j].index == i {
                return true;
            }
        }

        let mut g = y;

        let mut cache_values: Vec<((usize, usize), TX)> = Vec::new();

        for v in self.sv.iter() {
            let xi: Vec<_> = v.x.iter().map(|e| e.to_f64().unwrap()).collect();
            let xj: Vec<_> = x.iter().map(|e| e.to_f64().unwrap()).collect();
            let k = self
                .parameters
                .kernel
                .as_ref()
                .unwrap()
                .apply(&xi, &xj)
                .unwrap();
            cache_values.push(((i, v.index), TX::from(k).unwrap()));
            g -= v.alpha * k;
        }

        self.find_min_max_gradient();

        if self.gmin < self.gmax
            && ((y > 0.0 && g < self.gmin.to_f64().unwrap())
                || (y < 0.0 && g > self.gmax.to_f64().unwrap()))
        {
            return false;
        }

        for v in cache_values {
            cache.insert(v.0, v.1.to_f64().unwrap());
        }

        let x_f64: Vec<_> = x.iter().map(|e| e.to_f64().unwrap()).collect();
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

        if y > 0.0 {
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
        let mut range = if let Some(indices) = self.indices.clone() {
            indices
        } else {
            (0..n).collect::<Vec<usize>>()
        };
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
                let xi: Vec<_> = sv1.x.iter().map(|e| e.to_f64().unwrap()).collect();
                for i in 0..self.sv.len() {
                    let v = &self.sv[i];
                    let xj: Vec<_> = v.x.iter().map(|e| e.to_f64().unwrap()).collect();
                    let z = v.grad - gm;
                    let k = cache.get(
                        sv1,
                        v,
                        self.parameters
                            .kernel
                            .as_ref()
                            .unwrap()
                            .apply(&xi, &xj)
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

                let xi: Vec<_> = self.sv[idx_1]
                    .x
                    .iter()
                    .map(|e| e.to_f64().unwrap())
                    .collect::<Vec<_>>();

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
                                    &xi,
                                    &self.sv[idx_2]
                                        .x
                                        .iter()
                                        .map(|e| e.to_f64().unwrap())
                                        .collect::<Vec<_>>(),
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

                let xi: Vec<_> = sv2.x.iter().map(|e| e.to_f64().unwrap()).collect();
                for i in 0..self.sv.len() {
                    let v = &self.sv[i];
                    let xj: Vec<_> = v.x.iter().map(|e| e.to_f64().unwrap()).collect();
                    let z = gm - v.grad;
                    let k = cache.get(
                        sv2,
                        v,
                        self.parameters
                            .kernel
                            .as_ref()
                            .unwrap()
                            .apply(&xi, &xj)
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

                let xj: Vec<_> = self.sv[idx_2]
                    .x
                    .iter()
                    .map(|e| e.to_f64().unwrap())
                    .collect();

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
                                        .collect::<Vec<_>>(),
                                    &xj,
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
                            .collect::<Vec<_>>(),
                        &self.sv[idx_2]
                            .x
                            .iter()
                            .map(|e| e.to_f64().unwrap())
                            .collect::<Vec<_>>(),
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

        let xi_v1: Vec<_> = self.sv[v1].x.iter().map(|e| e.to_f64().unwrap()).collect();
        let xi_v2: Vec<_> = self.sv[v2].x.iter().map(|e| e.to_f64().unwrap()).collect();
        for i in 0..self.sv.len() {
            let xj: Vec<_> = self.sv[i].x.iter().map(|e| e.to_f64().unwrap()).collect();
            let k2 = cache.get(
                &self.sv[v2],
                &self.sv[i],
                self.parameters
                    .kernel
                    .as_ref()
                    .unwrap()
                    .apply(&xi_v2, &xj)
                    .unwrap(),
            );
            let k1 = cache.get(
                &self.sv[v1],
                &self.sv[i],
                self.parameters
                    .kernel
                    .as_ref()
                    .unwrap()
                    .apply(&xi_v1, &xj)
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
    use crate::svm::Kernels;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
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
        ])
        .unwrap();

        let y: Vec<i32> = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let knl = Kernels::linear();
        let parameters = SVCParameters::default()
            .with_c(200.0)
            .with_kernel(knl)
            .with_seed(Some(100));

        let y_hat = SVC::fit(&x, &y, &parameters)
            .and_then(|lr| lr.predict(&x))
            .unwrap();
        let acc = accuracy(&y, &(y_hat.iter().map(|e| e.to_i32().unwrap()).collect()));

        assert!(acc >= 0.9, "accuracy ({acc}) is not larger or equal to 0.9");
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn svc_fit_decision_function() {
        let x = DenseMatrix::from_2d_array(&[&[4.0, 0.0], &[0.0, 4.0], &[8.0, 0.0], &[0.0, 8.0]])
            .unwrap();

        let x2 = DenseMatrix::from_2d_array(&[
            &[3.0, 3.0],
            &[4.0, 4.0],
            &[6.0, 6.0],
            &[10.0, 10.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
        ])
        .unwrap();

        let y: Vec<i32> = vec![-1, -1, 1, 1];

        let y_hat = SVC::fit(
            &x,
            &y,
            &SVCParameters::default()
                .with_c(200.0)
                .with_kernel(Kernels::linear()),
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

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
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
        ])
        .unwrap();

        let y: Vec<i32> = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let y_hat = SVC::fit(
            &x,
            &y,
            &SVCParameters::default()
                .with_c(1.0)
                .with_kernel(Kernels::rbf().with_gamma(0.7)),
        )
        .and_then(|lr| lr.predict(&x))
        .unwrap();

        let acc = accuracy(&y, &(y_hat.iter().map(|e| e.to_i32().unwrap()).collect()));

        assert!(acc >= 0.9, "accuracy ({acc}) is not larger or equal to 0.9");
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn svc_multiclass_fit_predict() {
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
        ])
        .unwrap();

        let y: Vec<i32> = vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2];

        let knl = Kernels::linear();
        let parameters = SVCParameters::default()
            .with_c(200.0)
            .with_kernel(knl)
            .with_seed(Some(100));

        let y_hat = MultiClassSVC::fit(&x, &y, &parameters)
            .and_then(|lr| lr.predict(&x))
            .unwrap();

        let acc = accuracy(&y, &(y_hat.iter().map(|e| e.to_i32().unwrap()).collect()));

        assert!(
            acc >= 0.9,
            "Multiclass accuracy ({acc}) is not larger or equal to 0.9"
        );
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
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
        ])
        .unwrap();

        let y: Vec<i32> = vec![
            -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];

        let knl = Kernels::linear();
        let parameters = SVCParameters::default().with_kernel(knl);
        let svc = SVC::fit(&x, &y, &parameters).unwrap();

        // serialization
        let deserialized_svc: SVC<'_, f64, i32, _, _> =
            serde_json::from_str(&serde_json::to_string(&svc).unwrap()).unwrap();

        assert_eq!(svc, deserialized_svc);
    }
}
