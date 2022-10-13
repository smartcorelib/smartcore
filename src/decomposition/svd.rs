//! # Dimensionality reduction using SVD
//!
//! Similar to [`PCA`](../pca/index.html), SVD is a technique that can be used to reduce the number of input variables _p_ to a smaller number _k_, while preserving
//! the most important structure or relationships between the variables observed in the data.
//!
//! Contrary to PCA, SVD does not center the data before computing the singular value decomposition.
//!
//! Example:
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::decomposition::svd::*;
//!
//! // Iris data
//! let iris = DenseMatrix::from_2d_array(&[
//!                     &[5.1, 3.5, 1.4, 0.2],
//!                     &[4.9, 3.0, 1.4, 0.2],
//!                     &[4.7, 3.2, 1.3, 0.2],
//!                     &[4.6, 3.1, 1.5, 0.2],
//!                     &[5.0, 3.6, 1.4, 0.2],
//!                     &[5.4, 3.9, 1.7, 0.4],
//!                     &[4.6, 3.4, 1.4, 0.3],
//!                     &[5.0, 3.4, 1.5, 0.2],
//!                     &[4.4, 2.9, 1.4, 0.2],
//!                     &[4.9, 3.1, 1.5, 0.1],
//!                     &[7.0, 3.2, 4.7, 1.4],
//!                     &[6.4, 3.2, 4.5, 1.5],
//!                     &[6.9, 3.1, 4.9, 1.5],
//!                     &[5.5, 2.3, 4.0, 1.3],
//!                     &[6.5, 2.8, 4.6, 1.5],
//!                     &[5.7, 2.8, 4.5, 1.3],
//!                     &[6.3, 3.3, 4.7, 1.6],
//!                     &[4.9, 2.4, 3.3, 1.0],
//!                     &[6.6, 2.9, 4.6, 1.3],
//!                     &[5.2, 2.7, 3.9, 1.4],
//!                     ]);
//!
//! let svd = SVD::fit(&iris, SVDParameters::default().
//!         with_n_components(2)).unwrap(); // Reduce number of features to 2
//!
//! let iris_reduced = svd.transform(&iris).unwrap();
//!
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::fmt::Debug;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Transformer, UnsupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::Array2;
use crate::linalg::traits::evd::EVDDecomposable;
use crate::linalg::traits::svd::SVDDecomposable;
use crate::numbers::floatnum::FloatNumber;

/// SVD
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct SVD<T: FloatNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>> {
    components: X,
    phantom: PhantomData<T>,
}

impl<T: FloatNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>> PartialEq
    for SVD<T, X>
{
    fn eq(&self, other: &Self) -> bool {
        self.components
            .iterator(0)
            .zip(other.components.iterator(0))
            .all(|(&a, &b)| (a - b).abs() <= T::epsilon())
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// SVD parameters
pub struct SVDParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of components to keep.
    pub n_components: usize,
}

impl Default for SVDParameters {
    fn default() -> Self {
        SVDParameters { n_components: 2 }
    }
}

impl SVDParameters {
    /// Number of components to keep.
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }
}

/// SVD grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct SVDSearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Maximum number of iterations of the k-means algorithm for a single run.
    pub n_components: Vec<usize>,
}

/// SVD grid search iterator
pub struct SVDSearchParametersIterator {
    svd_search_parameters: SVDSearchParameters,
    current_n_components: usize,
}

impl IntoIterator for SVDSearchParameters {
    type Item = SVDParameters;
    type IntoIter = SVDSearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        SVDSearchParametersIterator {
            svd_search_parameters: self,
            current_n_components: 0,
        }
    }
}

impl Iterator for SVDSearchParametersIterator {
    type Item = SVDParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_n_components == self.svd_search_parameters.n_components.len() {
            return None;
        }

        let next = SVDParameters {
            n_components: self.svd_search_parameters.n_components[self.current_n_components],
        };

        self.current_n_components += 1;

        Some(next)
    }
}

impl Default for SVDSearchParameters {
    fn default() -> Self {
        let default_params = SVDParameters::default();

        SVDSearchParameters {
            n_components: vec![default_params.n_components],
        }
    }
}

impl<T: FloatNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>>
    UnsupervisedEstimator<X, SVDParameters> for SVD<T, X>
{
    fn fit(x: &X, parameters: SVDParameters) -> Result<Self, Failed> {
        SVD::fit(x, parameters)
    }
}

impl<T: FloatNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>> Transformer<X>
    for SVD<T, X>
{
    fn transform(&self, x: &X) -> Result<X, Failed> {
        self.transform(x)
    }
}

impl<T: FloatNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>> SVD<T, X> {
    /// Fits SVD to your data.
    /// * `data` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `n_components` - number of components to keep.
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(x: &X, parameters: SVDParameters) -> Result<SVD<T, X>, Failed> {
        let (_, p) = x.shape();

        if parameters.n_components >= p {
            return Err(Failed::fit(&format!(
                "Number of components, n_components should be < number of attributes ({})",
                p
            )));
        }

        let svd = x.svd()?;

        let components = X::from_slice(svd.V.slice(0..p, 0..parameters.n_components).as_ref());

        Ok(SVD {
            components,
            phantom: PhantomData,
        })
    }

    /// Run dimensionality reduction for `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn transform(&self, x: &X) -> Result<X, Failed> {
        let (n, p) = x.shape();
        let (p_c, k) = self.components.shape();
        if p_c != p {
            return Err(Failed::transform(&format!(
                "Can not transform a {}x{} matrix into {}x{} matrix, incorrect input dimentions",
                n, p, n, k
            )));
        }

        Ok(x.matmul(&self.components))
    }

    /// Get a projection matrix
    pub fn components(&self) -> &X {
        &self.components
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;
    use approx::relative_eq;

    #[test]
    fn search_parameters() {
        let parameters = SVDSearchParameters {
            n_components: vec![10, 100],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.n_components, 10);
        let next = iter.next().unwrap();
        assert_eq!(next.n_components, 100);
        assert!(iter.next().is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn svd_decompose() {
        // https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/USArrests.html
        let x = DenseMatrix::from_2d_array(&[
            &[13.2, 236.0, 58.0, 21.2],
            &[10.0, 263.0, 48.0, 44.5],
            &[8.1, 294.0, 80.0, 31.0],
            &[8.8, 190.0, 50.0, 19.5],
            &[9.0, 276.0, 91.0, 40.6],
            &[7.9, 204.0, 78.0, 38.7],
            &[3.3, 110.0, 77.0, 11.1],
            &[5.9, 238.0, 72.0, 15.8],
            &[15.4, 335.0, 80.0, 31.9],
            &[17.4, 211.0, 60.0, 25.8],
            &[5.3, 46.0, 83.0, 20.2],
            &[2.6, 120.0, 54.0, 14.2],
            &[10.4, 249.0, 83.0, 24.0],
            &[7.2, 113.0, 65.0, 21.0],
            &[2.2, 56.0, 57.0, 11.3],
            &[6.0, 115.0, 66.0, 18.0],
            &[9.7, 109.0, 52.0, 16.3],
            &[15.4, 249.0, 66.0, 22.2],
            &[2.1, 83.0, 51.0, 7.8],
            &[11.3, 300.0, 67.0, 27.8],
            &[4.4, 149.0, 85.0, 16.3],
            &[12.1, 255.0, 74.0, 35.1],
            &[2.7, 72.0, 66.0, 14.9],
            &[16.1, 259.0, 44.0, 17.1],
            &[9.0, 178.0, 70.0, 28.2],
            &[6.0, 109.0, 53.0, 16.4],
            &[4.3, 102.0, 62.0, 16.5],
            &[12.2, 252.0, 81.0, 46.0],
            &[2.1, 57.0, 56.0, 9.5],
            &[7.4, 159.0, 89.0, 18.8],
            &[11.4, 285.0, 70.0, 32.1],
            &[11.1, 254.0, 86.0, 26.1],
            &[13.0, 337.0, 45.0, 16.1],
            &[0.8, 45.0, 44.0, 7.3],
            &[7.3, 120.0, 75.0, 21.4],
            &[6.6, 151.0, 68.0, 20.0],
            &[4.9, 159.0, 67.0, 29.3],
            &[6.3, 106.0, 72.0, 14.9],
            &[3.4, 174.0, 87.0, 8.3],
            &[14.4, 279.0, 48.0, 22.5],
            &[3.8, 86.0, 45.0, 12.8],
            &[13.2, 188.0, 59.0, 26.9],
            &[12.7, 201.0, 80.0, 25.5],
            &[3.2, 120.0, 80.0, 22.9],
            &[2.2, 48.0, 32.0, 11.2],
            &[8.5, 156.0, 63.0, 20.7],
            &[4.0, 145.0, 73.0, 26.2],
            &[5.7, 81.0, 39.0, 9.3],
            &[2.6, 53.0, 66.0, 10.8],
            &[6.8, 161.0, 60.0, 15.6],
        ]);

        let expected = DenseMatrix::from_2d_array(&[
            &[243.54655757, -18.76673788],
            &[268.36802004, -33.79304302],
            &[305.93972467, -15.39087376],
            &[197.28420365, -11.66808306],
            &[293.43187394, 1.91163633],
        ]);
        let svd = SVD::fit(&x, Default::default()).unwrap();

        let x_transformed = svd.transform(&x).unwrap();

        assert_eq!(svd.components.shape(), (x.shape().1, 2));

        assert!(relative_eq!(
            DenseMatrix::from_slice(x_transformed.slice(0..5, 0..2).as_ref()),
            &expected,
            epsilon = 1e-4
        ));
    }

    // Disable this test for now
    // TODO: implement deserialization for new DenseMatrix
    // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    // #[test]
    // #[cfg(feature = "serde")]
    // fn serde() {
    //     let iris = DenseMatrix::from_2d_array(&[
    //         &[5.1, 3.5, 1.4, 0.2],
    //         &[4.9, 3.0, 1.4, 0.2],
    //         &[4.7, 3.2, 1.3, 0.2],
    //         &[4.6, 3.1, 1.5, 0.2],
    //         &[5.0, 3.6, 1.4, 0.2],
    //         &[5.4, 3.9, 1.7, 0.4],
    //         &[4.6, 3.4, 1.4, 0.3],
    //         &[5.0, 3.4, 1.5, 0.2],
    //         &[4.4, 2.9, 1.4, 0.2],
    //         &[4.9, 3.1, 1.5, 0.1],
    //         &[7.0, 3.2, 4.7, 1.4],
    //         &[6.4, 3.2, 4.5, 1.5],
    //         &[6.9, 3.1, 4.9, 1.5],
    //         &[5.5, 2.3, 4.0, 1.3],
    //         &[6.5, 2.8, 4.6, 1.5],
    //         &[5.7, 2.8, 4.5, 1.3],
    //         &[6.3, 3.3, 4.7, 1.6],
    //         &[4.9, 2.4, 3.3, 1.0],
    //         &[6.6, 2.9, 4.6, 1.3],
    //         &[5.2, 2.7, 3.9, 1.4],
    //     ]);

    //     let svd = SVD::fit(&iris, Default::default()).unwrap();

    //     let deserialized_svd: SVD<f32, DenseMatrix<f32>> =
    //         serde_json::from_str(&serde_json::to_string(&svd).unwrap()).unwrap();

    //     assert_eq!(svd, deserialized_svd);
    // }
}
