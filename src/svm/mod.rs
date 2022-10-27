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

pub mod svc;
// pub mod svr;

use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, ArrayView1};
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;

/// Defines a kernel function
pub trait Kernel<T: Number + RealNumber>: Clone {
    /// Apply kernel function to x_i and x_j
    fn apply(&self, x_i: &Vec<T>, x_j: &Vec<T>) -> Result<T, Failed>;
}

/// Pre-defined kernel functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kernels<T> {
    phantom: PhantomData<T>
}

impl<T: Number + RealNumber> Kernels<T> {
    fn linear() -> LinearKernel {
        LinearKernel::default()
    }

    fn rbf() -> RBFKernel<T> {
        RBFKernel::<T>::default()
    }

    fn polynomial() -> PolynomialKernel<T> {
        PolynomialKernel::<T>::default()
    }

    fn sigmoid() -> SigmoidKernel<T> {
        SigmoidKernel::<T>::default()
    }
}

/// Linear Kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearKernel {}

impl Default for LinearKernel {
    fn default() -> Self {
        Self {}
    }
}

/// Radial basis function (Gaussian) kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RBFKernel<T: Number + RealNumber> {
    /// kernel coefficient
    pub gamma: Option<T>,
}

impl<T: Number + RealNumber> Default for RBFKernel<T> {
    fn default() -> Self {
        Self {
            gamma: Option::None,
        }
    }
}

impl<T: Number + RealNumber> RBFKernel<T> {
    fn with_gamma(mut self, gamma: T) -> Self {
        self.gamma = Some(gamma);
        self
    }
}

/// Polynomial kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialKernel<T: Number + RealNumber> {
    /// degree of the polynomial
    pub degree: Option<T>,
    /// kernel coefficient
    pub gamma: Option<T>,
    /// independent term in kernel function
    pub coef0: Option<T>,
}

impl<T: Number + RealNumber> Default for PolynomialKernel<T> {
    fn default() -> Self {
        Self {
            gamma: Option::None,
            degree: Option::None,
            coef0: Some(T::one()),
        }
    }
}

impl<T: Number + RealNumber> PolynomialKernel<T> {
    fn with_params(mut self, degree: T, gamma: T, coef0: T) -> Self {
        self.degree = Some(degree);
        self.gamma = Some(gamma);
        self.coef0 = Some(coef0);
        self
    }

    fn with_gamma(mut self, gamma: T) -> Self {
        self.gamma = Some(gamma);
        self
    }

    fn with_degree(mut self, degree: T, n_features: usize) -> Self {
        self.with_params(
            degree,
            T::one(),
            T::one() / T::from_usize(n_features).unwrap(),
        )
    }
}

/// Sigmoid (hyperbolic tangent) kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SigmoidKernel<T: Number + RealNumber> {
    /// kernel coefficient
    pub gamma: Option<T>,
    /// independent term in kernel function
    pub coef0: Option<T>,
}

impl<T: Number + RealNumber> Default for SigmoidKernel<T> {
    fn default() -> Self {
        Self {
            gamma: Option::None,
            coef0: Some(T::one()),
        }
    }
}

impl<T: Number + RealNumber> SigmoidKernel<T> {
    fn with_params(mut self, gamma: T, coef0: T) -> Self {
        self.gamma = Some(gamma);
        self.coef0 = Some(coef0);
        self
    }
    fn with_gamma(mut self, gamma: T) -> Self {
        self.gamma = Some(gamma);
        self
    }
}

impl<T: Number + RealNumber> Kernel<T> for LinearKernel {
    fn apply(&self, x_i: &Vec<T>, x_j: &Vec<T>) -> Result<T, Failed> {
        Ok(x_i.dot(x_j))
    }
}

impl<T: Number + RealNumber> Kernel<T> for RBFKernel<T> {
    fn apply(&self, x_i: &Vec<T>, x_j: &Vec<T>) -> Result<T, Failed> {
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

impl<T: Number + RealNumber> Kernel<T> for PolynomialKernel<T> {
    fn apply(&self, x_i: &Vec<T>, x_j: &Vec<T>) -> Result<T, Failed> {
        if self.gamma.is_none() || self.coef0.is_none() || self.degree.is_none() {
            return Err(Failed::because(
                FailedError::ParametersError, "gamma, coef0, degree should be set, 
                                                        use {Kernel}::default().with_{parameter}(..)")
            );
        }
        let dot = x_i.dot(x_j);
        Ok(
            (self.gamma.unwrap() * dot + self.coef0.unwrap())
            .powf(self.degree.unwrap())
        )
    }
}

impl<T: Number + RealNumber> Kernel<T> for SigmoidKernel<T> {
    fn apply(&self, x_i: &Vec<T>, x_j: &Vec<T>) -> Result<T, Failed> {
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
    use crate::svm::LinearKernel;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linear_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert_eq!(32f64, Kernels::<f64>::linear().apply(&v1, &v2).unwrap());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rbf_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        let result = Kernels::<f64>::rbf().with_gamma(0.055).apply(&v1, &v2).unwrap().abs();

        assert!((0.2265f64 - result)  < 1e-4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polynomial_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        let result = Kernels::<f64>::polynomial()
            .with_params(3.0, 0.5, 1.0)
            .apply(&v1, &v2).unwrap()  
            .abs();

        assert!(
            (4913f64 - result) < std::f64::EPSILON
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sigmoid_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        let result = Kernels::<f64>::sigmoid().with_params(0.01, 0.1).apply(&v1, &v2).unwrap().abs();

        assert!(
            (0.3969f64 - result)  < 1e-4
        );
    }
}
