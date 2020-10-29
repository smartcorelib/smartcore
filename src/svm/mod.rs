//! # Support Vector Machines
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

pub mod svc;
pub mod svr;

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
        RBFKernel { gamma: gamma }
    }

    /// Polynomial kernel
    /// * `degree` - degree of the polynomial
    /// * `gamma` - kernel coefficient
    /// * `coef0` - independent term in kernel function
    pub fn polynomial<T: RealNumber>(degree: T, gamma: T, coef0: T) -> PolynomialKernel<T> {
        PolynomialKernel {
            degree: degree,
            gamma: gamma,
            coef0: coef0,
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
        SigmoidKernel {
            gamma: gamma,
            coef0: coef0,
        }
    }

    /// Sigmoid kernel    
    /// * `gamma` - kernel coefficient    
    pub fn sigmoid_with_gamma<T: RealNumber>(gamma: T) -> SigmoidKernel<T> {
        SigmoidKernel {
            gamma: gamma,
            coef0: T::one(),
        }
    }
}

/// Linear Kernel
#[derive(Serialize, Deserialize, Debug)]
pub struct LinearKernel {}

/// Radial basis function (Gaussian) kernel 
pub struct RBFKernel<T: RealNumber> {
    /// kernel coefficient
    pub gamma: T,
}

/// Polynomial kernel
pub struct PolynomialKernel<T: RealNumber> {
    /// degree of the polynomial
    pub degree: T,
    /// kernel coefficient
    pub gamma: T,
    /// independent term in kernel function
    pub coef0: T,
}

/// Sigmoid (hyperbolic tangent) kernel
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

    #[test]
    fn linear_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert_eq!(32f64, Kernels::linear().apply(&v1, &v2));
    }

    #[test]
    fn rbf_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert!((0.2265f64 - Kernels::rbf(0.055).apply(&v1, &v2)).abs() < 1e-4);
    }

    #[test]
    fn polynomial_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert!(
            (4913f64 - Kernels::polynomial(3.0, 0.5, 1.0).apply(&v1, &v2)).abs()
                < std::f64::EPSILON
        );
    }

    #[test]
    fn sigmoid_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert!((0.3969f64 - Kernels::sigmoid(0.01, 0.1).apply(&v1, &v2)).abs() < 1e-4);
    }
}
