//! # Support Vector Machines
//!
//! Support Vector Machines (SVM) is one of the most performant off-the-shelf machine learning algorithms.
//! SVM is based on the [Vapnik–Chervonenkiy theory](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) that was developed during 1960–1990 by Vladimir Vapnik and Alexey Chervonenkiy.
//!
//! SVM splits data into two sets using a maximal-margin decision boundary, \\(f(x)\\). For regression, the algorithm uses a value of the function \\(f(x)\\) to predict a target value.
//! To classify a new point, algorithm calculates a sign of the decision function to see where the new point is relative to the boundary.
//!
//! SVM is memory efficient since it uses only a subset of training data to find a decision boundary. This subset is called support vectors.
//!
//! In SVM distance between a data point and the support vectors is defined by the kernel function.
//! `smartcore` supports multiple kernel functions but you can always define a new kernel function by implementing the `Kernel` trait. Not all functions can be a kernel.
//! Building a new kernel requires a good mathematical understanding of the [Mercer theorem](https://en.wikipedia.org/wiki/Mercer%27s_theorem)
//! that gives necessary and sufficient condition for a function to be a kernel function.
//!
//! Pre-defined kernel functions:
//!
//! * *Linear*, \\( K(x, x') = \langle x, x' \rangle\\)
//! * *Polynomial*, \\( K(x, x') = (\gamma\langle x, x' \rangle + r)^d\\), where \\(d\\) is polynomial degree, \\(\gamma\\) is a kernel coefficient and \\(r\\) is an independent term in the kernel function.
//! * *RBF (Gaussian)*, \\( K(x, x') = e^{-\gamma \lVert x - x' \rVert ^2} \\), where \\(\gamma\\) is kernel coefficient
//! * *Sigmoid (hyperbolic tangent)*, \\( K(x, x') = \tanh ( \gamma \langle x, x' \rangle + r ) \\), where \\(\gamma\\) is kernel coefficient and \\(r\\) is an independent term in the kernel function.
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
/// search parameters
pub mod svc;
pub mod svr;
// search parameters space
pub mod search;

use core::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Only import typetag if not compiling for wasm32 and serde is enabled
#[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
use typetag;

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, ArrayView1};

/// Defines a kernel function.
/// This is a object-safe trait.
#[cfg_attr(
    all(feature = "serde", not(target_arch = "wasm32")),
    typetag::serde(tag = "type")
)]
pub trait Kernel: Debug {
    #[allow(clippy::ptr_arg)]
    /// Apply kernel function to x_i and x_j
    fn apply(&self, x_i: &Vec<f64>, x_j: &Vec<f64>) -> Result<f64, Failed>;
}

/// A enumerator for all the kernels type to support.
/// This allows kernel selection and parameterization ergonomic, type-safe, and ready for use in parameter structs like SVRParameters.
/// You can construct kernels using the provided variants and builder-style methods.
///
/// # Examples
///
/// ```
/// use smartcore::svm::Kernels;
///
/// let linear = Kernels::linear();
/// let rbf = Kernels::rbf().with_gamma(0.5);
/// let poly = Kernels::polynomial().with_degree(3.0).with_gamma(0.5).with_coef0(1.0);
/// let sigmoid = Kernels::sigmoid().with_gamma(0.2).with_coef0(0.0);
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Kernels {
    /// Linear kernel (default).
    ///
    /// Computes the standard dot product between vectors.
    Linear,

    /// Radial Basis Function (RBF) kernel.
    ///
    /// Formula: K(x, y) = exp(-gamma * ||x-y||²)
    RBF {
        /// Controls the width of the Gaussian RBF kernel.
        ///
        /// Larger values of gamma lead to higher bias and lower variance.
        /// This parameter is inversely proportional to the radius of influence
        /// of samples selected by the model as support vectors.
        gamma: Option<f64>,
    },

    /// Polynomial kernel.
    ///
    /// Formula: K(x, y) = (gamma * <x, y> + coef0)^degree
    Polynomial {
        /// The degree of the polynomial kernel.
        ///
        /// Integer values are typical (2 = quadratic, 3 = cubic), but any positive real value is valid.
        /// Higher degree values create decision boundaries with higher complexity.
        degree: Option<f64>,

        /// Kernel coefficient for the dot product.
        ///
        /// Controls the influence of higher-degree versus lower-degree terms in the polynomial.
        /// If None, a default value will be used.
        gamma: Option<f64>,

        /// Independent term in the polynomial kernel.
        ///
        /// Controls the influence of higher-degree versus lower-degree terms.
        /// If None, a default value of 1.0 will be used.
        coef0: Option<f64>,
    },

    /// Sigmoid kernel.
    ///
    /// Formula: K(x, y) = tanh(gamma * <x, y> + coef0)
    Sigmoid {
        /// Kernel coefficient for the dot product.
        ///
        /// Controls the scaling of the dot product in the sigmoid function.
        /// If None, a default value will be used.
        gamma: Option<f64>,

        /// Independent term in the sigmoid kernel.
        ///
        /// Acts as a threshold/bias term in the sigmoid function.
        /// If None, a default value of 1.0 will be used.
        coef0: Option<f64>,
    },
}

impl Kernels {
    /// Create a linear kernel.
    ///
    /// The linear kernel computes the dot product between two vectors:
    /// K(x, y) = <x, y>
    pub fn linear() -> Self {
        Kernels::Linear
    }

    /// Create an RBF kernel with unspecified gamma.
    ///
    /// The RBF kernel is defined as:
    /// K(x, y) = exp(-gamma * ||x-y||²)
    ///
    /// You should specify gamma using `with_gamma()` before using this kernel.
    pub fn rbf() -> Self {
        Kernels::RBF { gamma: None }
    }

    /// Create a polynomial kernel with default parameters.
    ///
    /// The polynomial kernel is defined as:
    /// K(x, y) = (gamma * <x, y> + coef0)^degree
    ///
    /// Default values:
    /// - gamma: None (must be specified)
    /// - degree: None (must be specified)
    /// - coef0: 1.0
    pub fn polynomial() -> Self {
        Kernels::Polynomial {
            gamma: None,
            degree: None,
            coef0: Some(1.0),
        }
    }

    /// Create a sigmoid kernel with default parameters.
    ///
    /// The sigmoid kernel is defined as:
    /// K(x, y) = tanh(gamma * <x, y> + coef0)
    ///
    /// Default values:
    /// - gamma: None (must be specified)
    /// - coef0: 1.0
    ///
    pub fn sigmoid() -> Self {
        Kernels::Sigmoid {
            gamma: None,
            coef0: Some(1.0),
        }
    }

    /// Set the `gamma` parameter for RBF, polynomial, or sigmoid kernels.
    ///
    /// The gamma parameter has different interpretations depending on the kernel:
    /// - For RBF: Controls the width of the Gaussian. Larger values mean tighter fit.
    /// - For Polynomial: Scaling factor for the dot product.
    /// - For Sigmoid: Scaling factor for the dot product.
    ///
    pub fn with_gamma(self, gamma: f64) -> Self {
        match self {
            Kernels::RBF { .. } => Kernels::RBF { gamma: Some(gamma) },
            Kernels::Polynomial { degree, coef0, .. } => Kernels::Polynomial {
                gamma: Some(gamma),
                degree,
                coef0,
            },
            Kernels::Sigmoid { coef0, .. } => Kernels::Sigmoid {
                gamma: Some(gamma),
                coef0,
            },
            other => other,
        }
    }

    /// Set the `degree` parameter for the polynomial kernel.
    ///
    /// The degree parameter controls the flexibility of the decision boundary.
    /// Higher degrees create more complex boundaries but may lead to overfitting.
    ///
    pub fn with_degree(self, degree: f64) -> Self {
        match self {
            Kernels::Polynomial { gamma, coef0, .. } => Kernels::Polynomial {
                degree: Some(degree),
                gamma,
                coef0,
            },
            other => other,
        }
    }

    /// Set the `coef0` parameter for polynomial or sigmoid kernels.
    ///
    /// The coef0 parameter is the independent term in the kernel function:
    /// - For Polynomial: Controls the influence of higher-degree vs. lower-degree terms.
    /// - For Sigmoid: Acts as a threshold/bias term.
    ///
    pub fn with_coef0(self, coef0: f64) -> Self {
        match self {
            Kernels::Polynomial { degree, gamma, .. } => Kernels::Polynomial {
                degree,
                gamma,
                coef0: Some(coef0),
            },
            Kernels::Sigmoid { gamma, .. } => Kernels::Sigmoid {
                gamma,
                coef0: Some(coef0),
            },
            other => other,
        }
    }
}

/// Implementation of the [`Kernel`] trait for the [`Kernels`] enum in smartcore.
///
/// This method computes the value of the kernel function between two feature vectors `x_i` and `x_j`,
/// according to the variant and parameters of the [`Kernels`] enum. This enables flexible and type-safe
/// selection of kernel functions for SVM and SVR models in smartcore.
///
/// # Supported Kernels
///
/// - [`Kernels::Linear`]: Computes the standard dot product between `x_i` and `x_j`.
/// - [`Kernels::RBF`]: Computes the Radial Basis Function (Gaussian) kernel. Requires `gamma`.
/// - [`Kernels::Polynomial`]: Computes the polynomial kernel. Requires `degree`, `gamma`, and `coef0`.
/// - [`Kernels::Sigmoid`]: Computes the sigmoid kernel. Requires `gamma` and `coef0`.
///
/// # Parameters
///
/// - `x_i`: First input vector (feature vector).
/// - `x_j`: Second input vector (feature vector).
///
/// # Returns
///
/// - `Ok(f64)`: The computed kernel value.
/// - `Err(Failed)`: If any required kernel parameter is missing.
///
/// # Errors
///
/// Returns `Err(Failed)` if a required parameter (such as `gamma`, `degree`, or `coef0`)
/// is `None` for the selected kernel variant.
///
/// # Example
///
/// ```
/// use smartcore::svm::Kernels;
/// use smartcore::svm::Kernel;
///
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![4.0, 5.0, 6.0];
/// let kernel = Kernels::rbf().with_gamma(0.5);
/// let value = kernel.apply(&x, &y).unwrap();
/// ```
///
/// # Notes
///
/// - This implementation follows smartcore's philosophy: pure Rust, no macros, no unsafe code,
///   and an accessible, pythonic API surface for both ML practitioners and Rust beginners.
/// - All kernel parameters must be set before calling `apply`; missing parameters will result in an error.
///
/// See the [`Kernels`] enum documentation for more details on each kernel type and its parameters.
#[cfg_attr(all(feature = "serde", not(target_arch = "wasm32")), typetag::serde)]
impl Kernel for Kernels {
    fn apply(&self, x_i: &Vec<f64>, x_j: &Vec<f64>) -> Result<f64, Failed> {
        match self {
            Kernels::Linear => Ok(x_i.dot(x_j)),
            Kernels::RBF { gamma } => {
                let gamma = gamma.ok_or_else(|| {
                    Failed::because(FailedError::ParametersError, "gamma not set")
                })?;
                let v_diff = x_i.sub(x_j);
                Ok((-gamma * v_diff.mul(&v_diff).sum()).exp())
            }
            Kernels::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                let degree = degree.ok_or_else(|| {
                    Failed::because(FailedError::ParametersError, "degree not set")
                })?;
                let gamma = gamma.ok_or_else(|| {
                    Failed::because(FailedError::ParametersError, "gamma not set")
                })?;
                let coef0 = coef0.ok_or_else(|| {
                    Failed::because(FailedError::ParametersError, "coef0 not set")
                })?;
                let dot = x_i.dot(x_j);
                Ok((gamma * dot + coef0).powf(degree))
            }
            Kernels::Sigmoid { gamma, coef0 } => {
                let gamma = gamma.ok_or_else(|| {
                    Failed::because(FailedError::ParametersError, "gamma not set")
                })?;
                let coef0 = coef0.ok_or_else(|| {
                    Failed::because(FailedError::ParametersError, "coef0 not set")
                })?;
                let dot = x_i.dot(x_j);
                Ok((gamma * dot + coef0).tanh())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::Kernels;

    #[test]
    fn rbf_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];
        let result = Kernels::rbf()
            .with_gamma(0.055)
            .apply(&v1, &v2)
            .unwrap()
            .abs();
        assert!((0.2265f64 - result) < 1e-4);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn linear_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert_eq!(32f64, Kernels::linear().apply(&v1, &v2).unwrap());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_rbf_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        let result = Kernels::rbf()
            .with_gamma(0.055)
            .apply(&v1, &v2)
            .unwrap()
            .abs();

        assert!((0.2265f64 - result) < 1e-4);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn polynomial_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        let result = Kernels::polynomial()
            .with_gamma(0.5)
            .with_degree(3.0)
            .with_coef0(1.0)
            //.with_params(3.0, 0.5, 1.0)
            .apply(&v1, &v2)
            .unwrap()
            .abs();

        assert!((4913f64 - result).abs() < f64::EPSILON);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn sigmoid_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        let result = Kernels::sigmoid()
            .with_gamma(0.01)
            .with_coef0(0.1)
            .apply(&v1, &v2)
            .unwrap()
            .abs();

        assert!((0.3969f64 - result) < 1e-4);
    }
}
