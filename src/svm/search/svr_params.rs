//! # SVR Grid Search Parameters
//!
//! This module provides utilities for defining and iterating over grid search parameter spaces
//! for Support Vector Regression (SVR) models in [smartcore](https://github.com/smartcorelib/smartcore).
//!
//! The main struct, [`SVRSearchParameters`], allows users to specify multiple values for each
//! SVR hyperparameter (epsilon, regularization parameter C, tolerance, and kernel function).
//! The provided iterator yields all possible combinations (the Cartesian product) of these parameters,
//! enabling exhaustive grid search for hyperparameter tuning.
//!
//!
//! ## Example
//! ```
//! use smartcore::svm::Kernels;
//! use smartcore::svm::search::svr_params::SVRSearchParameters;
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//!
//! let params = SVRSearchParameters::<f64, DenseMatrix<f64>> {
//!     eps: vec![0.1, 0.2],
//!     c: vec![1.0, 10.0],
//!     tol: vec![1e-3],
//!     kernel: vec![Kernels::linear(), Kernels::rbf().with_gamma(0.5)],
//!     m: std::marker::PhantomData,
//! };
//!
//! // for param_set in params.into_iter() {
//!     // Use param_set (of type svr::SVRParameters) to fit and evaluate your SVR model.
//! // }
//! ```
//!
//!
//! ## Note
//! This module is intended for use with smartcore version 0.4 or later. The API is not compatible with older versions[1].
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::basic::arrays::Array2;
use crate::numbers::basenum::Number;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;
use crate::svm::{svr, Kernels};
use std::marker::PhantomData;

/// ## SVR grid search parameters
/// A struct representing a grid of hyperparameters for SVR grid search in smartcore.
///
/// Each field is a vector of possible values for the corresponding SVR hyperparameter.
/// The [`IntoIterator`] implementation yields every possible combination of these parameters
/// as an `svr::SVRParameters` struct, suitable for use in model selection routines.
///
/// # Type Parameters
/// - `T`: Numeric type for parameters (e.g., `f64`)
/// - `M`: Matrix type implementing [`Array2<T>`]
///
/// # Fields
/// - `eps`: Vector of epsilon values for the epsilon-insensitive loss in SVR.
/// - `c`: Vector of regularization parameters (C) for SVR.
/// - `tol`: Vector of tolerance values for the stopping criterion.
/// - `kernel`: Vector of kernel function variants (see [`Kernels`]).
/// - `m`: Phantom data for the matrix type parameter.
///
/// # Example
/// ```
/// use smartcore::svm::Kernels;
/// use smartcore::svm::search::svr_params::SVRSearchParameters;
/// use smartcore::linalg::basic::matrix::DenseMatrix;
///
/// let params = SVRSearchParameters::<f64, DenseMatrix<f64>> {
///     eps: vec![0.1, 0.2],
///     c: vec![1.0, 10.0],
///     tol: vec![1e-3],
///     kernel: vec![Kernels::linear(), Kernels::rbf().with_gamma(0.5)],
///     m: std::marker::PhantomData,
/// };
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct SVRSearchParameters<T: Number + RealNumber, M: Array2<T>> {
    /// Epsilon in the epsilon-SVR model.
    pub eps: Vec<T>,
    /// Regularization parameter.
    pub c: Vec<T>,
    /// Tolerance for stopping eps.
    pub tol: Vec<T>,
    /// The kernel function.
    pub kernel: Vec<Kernels>,
    /// Unused parameter.
    pub m: PhantomData<M>,
}

/// SVR grid search iterator
pub struct SVRSearchParametersIterator<T: Number + RealNumber, M: Array2<T>> {
    svr_search_parameters: SVRSearchParameters<T, M>,
    current_eps: usize,
    current_c: usize,
    current_tol: usize,
    current_kernel: usize,
}

impl<T: Number + FloatNumber + RealNumber, M: Array2<T>> IntoIterator
    for SVRSearchParameters<T, M>
{
    type Item = svr::SVRParameters<T>;
    type IntoIter = SVRSearchParametersIterator<T, M>;

    fn into_iter(self) -> Self::IntoIter {
        SVRSearchParametersIterator {
            svr_search_parameters: self,
            current_eps: 0,
            current_c: 0,
            current_tol: 0,
            current_kernel: 0,
        }
    }
}

impl<T: Number + FloatNumber + RealNumber, M: Array2<T>> Iterator
    for SVRSearchParametersIterator<T, M>
{
    type Item = svr::SVRParameters<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_eps == self.svr_search_parameters.eps.len()
            && self.current_c == self.svr_search_parameters.c.len()
            && self.current_tol == self.svr_search_parameters.tol.len()
            && self.current_kernel == self.svr_search_parameters.kernel.len()
        {
            return None;
        }

        let next = svr::SVRParameters::<T> {
            eps: self.svr_search_parameters.eps[self.current_eps],
            c: self.svr_search_parameters.c[self.current_c],
            tol: self.svr_search_parameters.tol[self.current_tol],
            kernel: Some(self.svr_search_parameters.kernel[self.current_kernel].clone()),
        };

        if self.current_eps + 1 < self.svr_search_parameters.eps.len() {
            self.current_eps += 1;
        } else if self.current_c + 1 < self.svr_search_parameters.c.len() {
            self.current_eps = 0;
            self.current_c += 1;
        } else if self.current_tol + 1 < self.svr_search_parameters.tol.len() {
            self.current_eps = 0;
            self.current_c = 0;
            self.current_tol += 1;
        } else if self.current_kernel + 1 < self.svr_search_parameters.kernel.len() {
            self.current_eps = 0;
            self.current_c = 0;
            self.current_tol = 0;
            self.current_kernel += 1;
        } else {
            self.current_eps += 1;
            self.current_c += 1;
            self.current_tol += 1;
            self.current_kernel += 1;
        }

        Some(next)
    }
}

impl<T: Number + FloatNumber + RealNumber, M: Array2<T>> Default for SVRSearchParameters<T, M> {
    fn default() -> Self {
        let default_params: svr::SVRParameters<T> = svr::SVRParameters::default();

        SVRSearchParameters {
            eps: vec![default_params.eps],
            c: vec![default_params.c],
            tol: vec![default_params.tol],
            kernel: vec![default_params.kernel.unwrap_or_else(Kernels::linear)],
            m: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::svm::Kernels;

    type T = f64;
    type M = DenseMatrix<T>;

    #[test]
    fn test_default_parameters() {
        let params = SVRSearchParameters::<T, M>::default();
        assert_eq!(params.eps.len(), 1);
        assert_eq!(params.c.len(), 1);
        assert_eq!(params.tol.len(), 1);
        assert_eq!(params.kernel.len(), 1);
        // Check that the default kernel is linear
        assert_eq!(params.kernel[0], Kernels::linear());
    }

    #[test]
    fn test_single_grid_iteration() {
        let params = SVRSearchParameters::<T, M> {
            eps: vec![0.1],
            c: vec![1.0],
            tol: vec![1e-3],
            kernel: vec![Kernels::rbf().with_gamma(0.5)],
            m: PhantomData,
        };
        let mut iter = params.into_iter();
        let param = iter.next().unwrap();
        assert_eq!(param.eps, 0.1);
        assert_eq!(param.c, 1.0);
        assert_eq!(param.tol, 1e-3);
        assert_eq!(param.kernel, Some(Kernels::rbf().with_gamma(0.5)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_cartesian_grid_iteration() {
        let params = SVRSearchParameters::<T, M> {
            eps: vec![0.1, 0.2],
            c: vec![1.0, 2.0],
            tol: vec![1e-3],
            kernel: vec![Kernels::linear(), Kernels::rbf().with_gamma(0.5)],
            m: PhantomData,
        };
        let expected_count =
            params.eps.len() * params.c.len() * params.tol.len() * params.kernel.len();
        let results: Vec<_> = params.into_iter().collect();
        assert_eq!(results.len(), expected_count);

        // Check that all parameter combinations are present
        let mut seen = vec![];
        for p in &results {
            seen.push((p.eps, p.c, p.tol, p.kernel.clone().unwrap()));
        }
        for &eps in &[0.1, 0.2] {
            for &c in &[1.0, 2.0] {
                for &tol in &[1e-3] {
                    for kernel in &[Kernels::linear(), Kernels::rbf().with_gamma(0.5)] {
                        assert!(seen.contains(&(eps, c, tol, kernel.clone())));
                    }
                }
            }
        }
    }

    #[test]
    fn test_empty_grid() {
        let params = SVRSearchParameters::<T, M> {
            eps: vec![],
            c: vec![],
            tol: vec![],
            kernel: vec![],
            m: PhantomData,
        };
        let mut iter = params.into_iter();
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_kernel_enum_variants() {
        let lin = Kernels::linear();
        let rbf = Kernels::rbf().with_gamma(0.2);
        let poly = Kernels::polynomial()
            .with_degree(2.0)
            .with_gamma(1.0)
            .with_coef0(0.5);
        let sig = Kernels::sigmoid().with_gamma(0.3).with_coef0(0.1);

        assert_eq!(lin, Kernels::Linear);
        match rbf {
            Kernels::RBF { gamma } => assert_eq!(gamma, Some(0.2)),
            _ => panic!("Not RBF"),
        }
        match poly {
            Kernels::Polynomial {
                degree,
                gamma,
                coef0,
            } => {
                assert_eq!(degree, Some(2.0));
                assert_eq!(gamma, Some(1.0));
                assert_eq!(coef0, Some(0.5));
            }
            _ => panic!("Not Polynomial"),
        }
        match sig {
            Kernels::Sigmoid { gamma, coef0 } => {
                assert_eq!(gamma, Some(0.3));
                assert_eq!(coef0, Some(0.1));
            }
            _ => panic!("Not Sigmoid"),
        }
    }
}
