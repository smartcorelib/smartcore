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

/// Kernel functions
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Kernel<T: RealNumber> {
    /// Linear kernel
    Linear,
    /// Radial basis function kernel (Gaussian)
    RBF {
        /// kernel coefficient
        gamma: T,
    },
    /// Sigmoid kernel    
    Sigmoid {
        /// kernel coefficient
        gamma: T,
        /// independent term in kernel function
        coef0: T,
    },
    /// Polynomial kernel
    Polynomial {
        /// kernel coefficient
        gamma: T,
        /// independent term in kernel function
        coef0: T,
        /// degree of the polynomial
        degree: T,
    },
}

impl<T: RealNumber> Default for Kernel<T> {
    fn default() -> Self {
        Kernel::Linear
    }
}

fn apply<T: RealNumber, V: BaseVector<T>>(kernel: &Kernel<T>, x_i: &V, x_j: &V) -> T {
    match kernel {
        Kernel::Polynomial {
            degree,
            gamma,
            coef0,
        } => {
            let dot = x_i.dot(x_j);
            (*gamma * dot + *coef0).powf(*degree)
        }
        Kernel::Sigmoid { gamma, coef0 } => {
            let dot = x_i.dot(x_j);
            (*gamma * dot + *coef0).tanh()
        }
        Kernel::RBF { gamma } => {
            let v_diff = x_i.sub(x_j);
            (-*gamma * v_diff.mul(&v_diff).sum()).exp()
        }
        Kernel::Linear => x_i.dot(x_j),
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

        assert_eq!(32f64, apply(&Kernel::Linear, &v1, &v2));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rbf_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert!((0.2265f64 - apply(&Kernel::RBF { gamma: 0.055 }, &v1, &v2)).abs() < 1e-4);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn polynomial_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert!(
            (4913f64
                - apply(
                    &Kernel::Polynomial {
                        gamma: 0.5,
                        coef0: 1.0,
                        degree: 3.0
                    },
                    &v1,
                    &v2
                ))
            .abs()
                < std::f64::EPSILON
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sigmoid_kernel() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![4., 5., 6.];

        assert!(
            (0.3969f64
                - apply(
                    &Kernel::Sigmoid {
                        gamma: 0.01,
                        coef0: 0.1
                    },
                    &v1,
                    &v2
                ))
            .abs()
                < 1e-4
        );
    }
}
