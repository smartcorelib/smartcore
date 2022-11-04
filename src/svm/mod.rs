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
//! SmartCore supports multiple kernel functions but you can always define a new kernel function by implementing the `Kernel` trait. Not all functions can be a kernel.
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
// /// search parameters space
// pub mod search;

use core::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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

/// Pre-defined kernel functions
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Kernels;

impl Kernels {
    /// Return a default linear
    pub fn linear() -> LinearKernel {
        LinearKernel::default()
    }
    /// Return a default RBF
    pub fn rbf() -> RBFKernel {
        RBFKernel::default()
    }
    /// Return a default polynomial
    pub fn polynomial() -> PolynomialKernel {
        PolynomialKernel::default()
    }
    /// Return a default sigmoid
    pub fn sigmoid() -> SigmoidKernel {
        SigmoidKernel::default()
    }
}

/// Linear Kernel
#[allow(clippy::derive_partial_eq_without_eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LinearKernel;

/// Radial basis function (Gaussian) kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RBFKernel {
    /// kernel coefficient
    pub gamma: Option<f64>,
}

#[allow(dead_code)]
impl RBFKernel {
    /// assign gamma parameter to kernel (required)
    /// ```rust
    /// use smartcore::svm::RBFKernel;
    /// let knl = RBFKernel::default().with_gamma(0.7);
    /// ```
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }
}

/// Polynomial kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialKernel {
    /// degree of the polynomial
    pub degree: Option<f64>,
    /// kernel coefficient
    pub gamma: Option<f64>,
    /// independent term in kernel function
    pub coef0: Option<f64>,
}

impl Default for PolynomialKernel {
    fn default() -> Self {
        Self {
            gamma: Option::None,
            degree: Option::None,
            coef0: Some(1f64),
        }
    }
}

impl PolynomialKernel {
    /// set parameters for kernel
    /// ```rust
    /// use smartcore::svm::PolynomialKernel;
    /// let knl = PolynomialKernel::default().with_params(3.0, 0.7, 1.0);
    /// ```
    pub fn with_params(mut self, degree: f64, gamma: f64, coef0: f64) -> Self {
        self.degree = Some(degree);
        self.gamma = Some(gamma);
        self.coef0 = Some(coef0);
        self
    }
    /// set gamma parameter for kernel
    /// ```rust
    /// use smartcore::svm::PolynomialKernel;
    /// let knl = PolynomialKernel::default().with_gamma(0.7);
    /// ```
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }
    /// set degree parameter for kernel
    /// ```rust
    /// use smartcore::svm::PolynomialKernel;
    /// let knl = PolynomialKernel::default().with_degree(3.0, 100);
    /// ```
    pub fn with_degree(self, degree: f64, n_features: usize) -> Self {
        self.with_params(degree, 1f64, 1f64 / n_features as f64)
    }
}

/// Sigmoid (hyperbolic tangent) kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SigmoidKernel {
    /// kernel coefficient
    pub gamma: Option<f64>,
    /// independent term in kernel function
    pub coef0: Option<f64>,
}

impl Default for SigmoidKernel {
    fn default() -> Self {
        Self {
            gamma: Option::None,
            coef0: Some(1f64),
        }
    }
}

impl SigmoidKernel {
    /// set parameters for kernel
    /// ```rust
    /// use smartcore::svm::SigmoidKernel;
    /// let knl = SigmoidKernel::default().with_params(0.7, 1.0);
    /// ```
    pub fn with_params(mut self, gamma: f64, coef0: f64) -> Self {
        self.gamma = Some(gamma);
        self.coef0 = Some(coef0);
        self
    }
    /// set gamma parameter for kernel
    /// ```rust
    /// use smartcore::svm::SigmoidKernel;
    /// let knl = SigmoidKernel::default().with_gamma(0.7);
    /// ```
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }
}

#[cfg_attr(all(feature = "serde", not(target_arch = "wasm32")), typetag::serde)]
impl Kernel for LinearKernel {
    fn apply(&self, x_i: &Vec<f64>, x_j: &Vec<f64>) -> Result<f64, Failed> {
        Ok(x_i.dot(x_j))
    }
}

#[cfg_attr(all(feature = "serde", not(target_arch = "wasm32")), typetag::serde)]
impl Kernel for RBFKernel {
    fn apply(&self, x_i: &Vec<f64>, x_j: &Vec<f64>) -> Result<f64, Failed> {
        if self.gamma.is_none() {
            return Err(Failed::because(
                FailedError::ParametersError,
                "gamma should be set, use {Kernel}::default().with_gamma(..)",
            ));
        }
        let v_diff = x_i.sub(x_j);
        Ok((-self.gamma.unwrap() * v_diff.mul(&v_diff).sum()).exp())
    }
}

#[cfg_attr(all(feature = "serde", not(target_arch = "wasm32")), typetag::serde)]
impl Kernel for PolynomialKernel {
    fn apply(&self, x_i: &Vec<f64>, x_j: &Vec<f64>) -> Result<f64, Failed> {
        if self.gamma.is_none() || self.coef0.is_none() || self.degree.is_none() {
            return Err(Failed::because(
                FailedError::ParametersError, "gamma, coef0, degree should be set, 
                                                        use {Kernel}::default().with_{parameter}(..)")
            );
        }
        let dot = x_i.dot(x_j);
        Ok((self.gamma.unwrap() * dot + self.coef0.unwrap()).powf(self.degree.unwrap()))
    }
}

#[cfg_attr(all(feature = "serde", not(target_arch = "wasm32")), typetag::serde)]
impl Kernel for SigmoidKernel {
    fn apply(&self, x_i: &Vec<f64>, x_j: &Vec<f64>) -> Result<f64, Failed> {
        if self.gamma.is_none() || self.coef0.is_none() {
            return Err(Failed::because(
                FailedError::ParametersError, "gamma, coef0, degree should be set, 
                                                        use {Kernel}::default().with_{parameter}(..)")
            );
        }
        let dot = x_i.dot(x_j);
        Ok(self.gamma.unwrap() * dot + self.coef0.unwrap().tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::Kernels;

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
    fn polynomial_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        let result = Kernels::polynomial()
            .with_params(3.0, 0.5, 1.0)
            .apply(&v1, &v2)
            .unwrap()
            .abs();

        assert!((4913f64 - result) < std::f64::EPSILON);
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
            .with_params(0.01, 0.1)
            .apply(&v1, &v2)
            .unwrap()
            .abs();

        assert!((0.3969f64 - result) < 1e-4);
    }
}
