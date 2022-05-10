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
pub mod svr;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::BaseVector;
use crate::math::num::RealNumber;

/// Defines a kernel function
pub trait Kernel<T: RealNumber, V: BaseVector<T>> {
    /// Apply kernel function to x_i and x_j
    fn apply(&self, x_i: &V, x_j: &V) -> T;
}

/// Pre-defined kernel functions
pub struct Kernels {}

impl Kernels {
    /// Linear kernel
    pub fn linear() -> LinearKernel {
        LinearKernel {}
    }

    /// Radial basis function kernel (Gaussian)
    pub fn rbf<T: RealNumber>(gamma: T) -> RBFKernel<T> {
        RBFKernel { gamma }
    }

    /// Polynomial kernel
    /// * `degree` - degree of the polynomial
    /// * `gamma` - kernel coefficient
    /// * `coef0` - independent term in kernel function
    pub fn polynomial<T: RealNumber>(degree: T, gamma: T, coef0: T) -> PolynomialKernel<T> {
        PolynomialKernel {
            degree,
            gamma,
            coef0,
        }
    }

    /// Polynomial kernel
    /// * `degree` - degree of the polynomial
    /// * `n_features` - number of features in vector
    pub fn polynomial_with_degree<T: RealNumber>(
        degree: T,
        n_features: usize,
    ) -> PolynomialKernel<T> {
        let coef0 = T::one();
        let gamma = T::one() / T::from_usize(n_features).unwrap();
        Kernels::polynomial(degree, gamma, coef0)
    }

    /// Sigmoid kernel    
    /// * `gamma` - kernel coefficient
    /// * `coef0` - independent term in kernel function
    pub fn sigmoid<T: RealNumber>(gamma: T, coef0: T) -> SigmoidKernel<T> {
        SigmoidKernel { gamma, coef0 }
    }

    /// Sigmoid kernel    
    /// * `gamma` - kernel coefficient    
    pub fn sigmoid_with_gamma<T: RealNumber>(gamma: T) -> SigmoidKernel<T> {
        SigmoidKernel {
            gamma,
            coef0: T::one(),
        }
    }
}

/// Linear Kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LinearKernel {}

/// Radial basis function (Gaussian) kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct RBFKernel<T: RealNumber> {
    /// kernel coefficient
    pub gamma: T,
}

/// Polynomial kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct PolynomialKernel<T: RealNumber> {
    /// degree of the polynomial
    pub degree: T,
    /// kernel coefficient
    pub gamma: T,
    /// independent term in kernel function
    pub coef0: T,
}

/// Sigmoid (hyperbolic tangent) kernel
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct SigmoidKernel<T: RealNumber> {
    /// kernel coefficient
    pub gamma: T,
    /// independent term in kernel function
    pub coef0: T,
}

impl<T: RealNumber, V: BaseVector<T>> Kernel<T, V> for LinearKernel {
    fn apply(&self, x_i: &V, x_j: &V) -> T {
        x_i.dot(x_j)
    }
}

impl<T: RealNumber, V: BaseVector<T>> Kernel<T, V> for RBFKernel<T> {
    fn apply(&self, x_i: &V, x_j: &V) -> T {
        let v_diff = x_i.sub(x_j);
        (-self.gamma * v_diff.mul(&v_diff).sum()).exp()
    }
}

impl<T: RealNumber, V: BaseVector<T>> Kernel<T, V> for PolynomialKernel<T> {
    fn apply(&self, x_i: &V, x_j: &V) -> T {
        let dot = x_i.dot(x_j);
        (self.gamma * dot + self.coef0).powf(self.degree)
    }
}

impl<T: RealNumber, V: BaseVector<T>> Kernel<T, V> for SigmoidKernel<T> {
    fn apply(&self, x_i: &V, x_j: &V) -> T {
        let dot = x_i.dot(x_j);
        (self.gamma * dot + self.coef0).tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linear_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert_eq!(32f64, Kernels::linear().apply(&v1, &v2));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rbf_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert!((0.2265f64 - Kernels::rbf(0.055).apply(&v1, &v2)).abs() < 1e-4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polynomial_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert!(
            (4913f64 - Kernels::polynomial(3.0, 0.5, 1.0).apply(&v1, &v2)).abs()
                < std::f64::EPSILON
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sigmoid_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert!((0.3969f64 - Kernels::sigmoid(0.01, 0.1).apply(&v1, &v2)).abs() < 1e-4);
    }
}
