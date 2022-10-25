//! # PCA
//!
//! Principal components analysis (PCA) is a method that is used to select several linear combinations that capture most of the variation in your data.
//! PCA is an unsupervised approach, since it involves only a set of features \\(X1, X2, . . . , Xn\\), and no associated response \\(Y\\).
//! Apart from producing derived variables for use in supervised learning problems, PCA also serves as a tool for data visualization.
//!
//! PCA is scale sensitive. Before PCA is performed, the variables should be centered to have mean zero.
//! Furthermore, the results obtained also depend on whether the variables have been individually scaled.
//! Use `use_correlation_matrix` parameter to standardize your variables (to mean 0 and standard deviation 1).
//!
//! Example:
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::decomposition::pca::*;
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
//! let pca = PCA::fit(&iris, PCAParameters::default().with_n_components(2)).unwrap(); // Reduce number of features to 2
//!
//! let iris_reduced = pca.transform(&iris).unwrap();
//!
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::{Transformer, UnsupervisedEstimator};
use crate::error::Failed;
use crate::linalg::basic::arrays::Array2;
use crate::linalg::traits::evd::EVDDecomposable;
use crate::linalg::traits::svd::SVDDecomposable;
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;

/// Principal components analysis algorithm
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct PCA<T: Number + RealNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>> {
    eigenvectors: X,
    eigenvalues: Vec<T>,
    projection: X,
    mu: Vec<T>,
    pmu: Vec<T>,
}

impl<T: Number + RealNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>> PartialEq
    for PCA<T, X>
{
    fn eq(&self, other: &Self) -> bool {
        if self.eigenvalues.len() != other.eigenvalues.len()
            || self
                .eigenvectors
                .iterator(0)
                .zip(other.eigenvectors.iterator(0))
                .any(|(&a, &b)| (a - b).abs() > T::epsilon())
        {
            false
        } else {
            for i in 0..self.eigenvalues.len() {
                if (self.eigenvalues[i] - other.eigenvalues[i]).abs() > T::epsilon() {
                    return false;
                }
            }
            true
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// PCA parameters
pub struct PCAParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of components to keep.
    pub n_components: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    /// By default, covariance matrix is used to compute principal components.
    /// Enable this flag if you want to use correlation matrix instead.
    pub use_correlation_matrix: bool,
}

impl PCAParameters {
    /// Number of components to keep.
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }
    /// By default, covariance matrix is used to compute principal components.
    /// Enable this flag if you want to use correlation matrix instead.
    pub fn with_use_correlation_matrix(mut self, use_correlation_matrix: bool) -> Self {
        self.use_correlation_matrix = use_correlation_matrix;
        self
    }
}

impl Default for PCAParameters {
    fn default() -> Self {
        PCAParameters {
            n_components: 2,
            use_correlation_matrix: false,
        }
    }
}

/// PCA grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct PCASearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of components to keep.
    pub n_components: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// By default, covariance matrix is used to compute principal components.
    /// Enable this flag if you want to use correlation matrix instead.
    pub use_correlation_matrix: Vec<bool>,
}

/// PCA grid search iterator
pub struct PCASearchParametersIterator {
    pca_search_parameters: PCASearchParameters,
    current_k: usize,
    current_use_correlation_matrix: usize,
}

impl IntoIterator for PCASearchParameters {
    type Item = PCAParameters;
    type IntoIter = PCASearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        PCASearchParametersIterator {
            pca_search_parameters: self,
            current_k: 0,
            current_use_correlation_matrix: 0,
        }
    }
}

impl Iterator for PCASearchParametersIterator {
    type Item = PCAParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_k == self.pca_search_parameters.n_components.len()
            && self.current_use_correlation_matrix
                == self.pca_search_parameters.use_correlation_matrix.len()
        {
            return None;
        }

        let next = PCAParameters {
            n_components: self.pca_search_parameters.n_components[self.current_k],
            use_correlation_matrix: self.pca_search_parameters.use_correlation_matrix
                [self.current_use_correlation_matrix],
        };

        if self.current_k + 1 < self.pca_search_parameters.n_components.len() {
            self.current_k += 1;
        } else if self.current_use_correlation_matrix + 1
            < self.pca_search_parameters.use_correlation_matrix.len()
        {
            self.current_k = 0;
            self.current_use_correlation_matrix += 1;
        } else {
            self.current_k += 1;
            self.current_use_correlation_matrix += 1;
        }

        Some(next)
    }
}

impl Default for PCASearchParameters {
    fn default() -> Self {
        let default_params = PCAParameters::default();

        PCASearchParameters {
            n_components: vec![default_params.n_components],
            use_correlation_matrix: vec![default_params.use_correlation_matrix],
        }
    }
}

impl<T: Number + RealNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>>
    UnsupervisedEstimator<X, PCAParameters> for PCA<T, X>
{
    fn fit(x: &X, parameters: PCAParameters) -> Result<Self, Failed> {
        PCA::fit(x, parameters)
    }
}

impl<T: Number + RealNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>> Transformer<X>
    for PCA<T, X>
{
    fn transform(&self, x: &X) -> Result<X, Failed> {
        self.transform(x)
    }
}

impl<T: Number + RealNumber, X: Array2<T> + SVDDecomposable<T> + EVDDecomposable<T>> PCA<T, X> {
    /// Fits PCA to your data.
    /// * `data` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `n_components` - number of components to keep.
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.
    pub fn fit(data: &X, parameters: PCAParameters) -> Result<PCA<T, X>, Failed> {
        let (m, n) = data.shape();

        if parameters.n_components > n {
            return Err(Failed::fit(&format!(
                "Number of components, n_components should be <= number of attributes ({})",
                n
            )));
        }

        let mu: Vec<T> = data
            .mean(0)
            .iter()
            .map(|&v| T::from_f64(v).unwrap())
            .collect();

        let mut x = data.clone();

        for (c, &mu_c) in mu.iter().enumerate().take(n) {
            for r in 0..m {
                x.sub_element_mut((r, c), mu_c);
            }
        }

        let mut eigenvalues;
        let mut eigenvectors;

        if m > n && !parameters.use_correlation_matrix {
            let svd = x.svd()?;
            eigenvalues = svd.s;
            for eigenvalue in &mut eigenvalues {
                *eigenvalue = *eigenvalue * (*eigenvalue);
            }

            eigenvectors = svd.V;
        } else {
            let mut cov = X::zeros(n, n);

            for k in 0..m {
                for i in 0..n {
                    for j in 0..=i {
                        cov.add_element_mut((i, j), *x.get((k, i)) * *x.get((k, j)));
                    }
                }
            }

            for i in 0..n {
                for j in 0..=i {
                    cov.div_element_mut((i, j), T::from(m).unwrap());
                    cov.set((j, i), *cov.get((i, j)));
                }
            }

            if parameters.use_correlation_matrix {
                let mut sd = vec![T::zero(); n];
                for (i, sd_i) in sd.iter_mut().enumerate().take(n) {
                    *sd_i = cov.get((i, i)).sqrt();
                }

                for i in 0..n {
                    for j in 0..=i {
                        cov.div_element_mut((i, j), sd[i] * sd[j]);
                        cov.set((j, i), *cov.get((i, j)));
                    }
                }

                let evd = cov.evd(true)?;

                eigenvalues = evd.d;

                eigenvectors = evd.V;

                for (i, sd_i) in sd.iter().enumerate().take(n) {
                    for j in 0..n {
                        eigenvectors.div_element_mut((i, j), *sd_i);
                    }
                }
            } else {
                let evd = cov.evd(true)?;

                eigenvalues = evd.d;

                eigenvectors = evd.V;
            }
        }

        let mut projection = X::zeros(parameters.n_components, n);
        for i in 0..n {
            for j in 0..parameters.n_components {
                projection.set((j, i), *eigenvectors.get((i, j)));
            }
        }

        let mut pmu = vec![T::zero(); parameters.n_components];
        for (k, mu_k) in mu.iter().enumerate().take(n) {
            for (i, pmu_i) in pmu.iter_mut().enumerate().take(parameters.n_components) {
                *pmu_i += *projection.get((i, k)) * (*mu_k);
            }
        }

        Ok(PCA {
            eigenvectors,
            eigenvalues,
            projection: projection.transpose(),
            mu,
            pmu,
        })
    }

    /// Run dimensionality reduction for `x`
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn transform(&self, x: &X) -> Result<X, Failed> {
        let (nrows, ncols) = x.shape();
        let (_, n_components) = self.projection.shape();
        if ncols != self.mu.len() {
            return Err(Failed::transform(&format!(
                "Invalid input vector size: {}, expected: {}",
                ncols,
                self.mu.len()
            )));
        }

        let mut x_transformed = x.matmul(&self.projection);
        for r in 0..nrows {
            for c in 0..n_components {
                x_transformed.sub_element_mut((r, c), self.pmu[c]);
            }
        }
        Ok(x_transformed)
    }

    /// Get a projection matrix
    pub fn components(&self) -> &X {
        &self.projection
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use approx::relative_eq;

    #[test]
    fn search_parameters() {
        let parameters = PCASearchParameters {
            n_components: vec![2, 4],
            use_correlation_matrix: vec![true, false],
            ..Default::default()
        };
        let mut iter = parameters.into_iter();
        let next = iter.next().unwrap();
        assert_eq!(next.n_components, 2);
        assert_eq!(next.use_correlation_matrix, true);
        let next = iter.next().unwrap();
        assert_eq!(next.n_components, 4);
        assert_eq!(next.use_correlation_matrix, true);
        let next = iter.next().unwrap();
        assert_eq!(next.n_components, 2);
        assert_eq!(next.use_correlation_matrix, false);
        let next = iter.next().unwrap();
        assert_eq!(next.n_components, 4);
        assert_eq!(next.use_correlation_matrix, false);
        assert!(iter.next().is_none());
    }

    fn us_arrests_data() -> DenseMatrix<f64> {
        DenseMatrix::from_2d_array(&[
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
        ])
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pca_components() {
        let us_arrests = us_arrests_data();

        let expected = DenseMatrix::from_2d_array(&[
            &[0.0417, 0.0448],
            &[0.9952, 0.0588],
            &[0.0463, 0.9769],
            &[0.0752, 0.2007],
        ]);

        let pca = PCA::fit(&us_arrests, Default::default()).unwrap();

        assert!(relative_eq!(
            expected,
            pca.components().abs(),
            epsilon = 1e-3
        ));
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn decompose_covariance() {
        let us_arrests = us_arrests_data();

        let expected_eigenvectors = DenseMatrix::from_2d_array(&[
            &[
                -0.0417043206282872,
                -0.0448216562696701,
                -0.0798906594208108,
                -0.994921731246978,
            ],
            &[
                -0.995221281426497,
                -0.058760027857223,
                0.0675697350838043,
                0.0389382976351601,
            ],
            &[
                -0.0463357461197108,
                0.97685747990989,
                0.200546287353866,
                -0.0581691430589319,
            ],
            &[
                -0.075155500585547,
                0.200718066450337,
                -0.974080592182491,
                0.0723250196376097,
            ],
        ]);

        let expected_projection = DenseMatrix::from_2d_array(&[
            &[-64.8022, -11.448, 2.4949, -2.4079],
            &[-92.8275, -17.9829, -20.1266, 4.094],
            &[-124.0682, 8.8304, 1.6874, 4.3537],
            &[-18.34, -16.7039, -0.2102, 0.521],
            &[-107.423, 22.5201, -6.7459, 2.8118],
            &[-34.976, 13.7196, -12.2794, 1.7215],
            &[60.8873, 12.9325, 8.4207, 0.6999],
            &[-66.731, 1.3538, 11.281, 3.728],
            &[-165.2444, 6.2747, 2.9979, -1.2477],
            &[-40.5352, -7.2902, -3.6095, -7.3437],
            &[123.5361, 24.2912, -3.7244, -3.4728],
            &[51.797, -9.4692, 1.5201, 3.3478],
            &[-78.9921, 12.8971, 5.8833, -0.3676],
            &[57.551, 2.8463, -3.7382, -1.6494],
            &[115.5868, -3.3421, 0.654, 0.8695],
            &[55.7897, 3.1572, -0.3844, -0.6528],
            &[62.3832, -10.6733, -2.2371, -3.8762],
            &[-78.2776, -4.2949, 3.8279, -4.4836],
            &[89.261, -11.4878, 4.6924, 2.1162],
            &[-129.3301, -5.007, 2.3472, 1.9283],
            &[21.2663, 19.4502, 7.5071, 1.0348],
            &[-85.4515, 5.9046, -6.4643, -0.499],
            &[98.9548, 5.2096, -0.0066, 0.7319],
            &[-86.8564, -27.4284, 5.0034, -3.8798],
            &[-7.9863, 5.2756, -5.5006, -0.6794],
            &[62.4836, -9.5105, -1.8384, -0.2459],
            &[69.0965, -0.2112, -0.468, 0.6566],
            &[-83.6136, 15.1022, -15.8887, -0.3342],
            &[114.7774, -4.7346, 2.2824, 0.9359],
            &[10.8157, 23.1373, 6.3102, -1.6124],
            &[-114.8682, -0.3365, -2.2613, 1.3812],
            &[-84.2942, 15.924, 4.7213, -0.892],
            &[-164.3255, -31.0966, 11.6962, 2.1112],
            &[127.4956, -16.135, 1.3118, 2.301],
            &[50.0868, 12.2793, -1.6573, -2.0291],
            &[19.6937, 3.3701, 0.4531, 0.1803],
            &[11.1502, 3.8661, -8.13, 2.914],
            &[64.6891, 8.9115, 3.2065, -1.8749],
            &[-3.064, 18.374, 17.47, 2.3083],
            &[-107.2811, -23.5361, 2.0328, -1.2517],
            &[86.1067, -16.5979, -1.3144, 1.2523],
            &[-17.5063, -6.5066, -6.1001, -3.9229],
            &[-31.2911, 12.985, 0.3934, -4.242],
            &[49.9134, 17.6485, -1.7882, 1.8677],
            &[124.7145, -27.3136, -4.8028, 2.005],
            &[14.8174, -1.7526, -1.0454, -1.1738],
            &[25.0758, 9.968, -4.7811, 2.6911],
            &[91.5446, -22.9529, 0.402, -0.7369],
            &[118.1763, 5.5076, 2.7113, -0.205],
            &[10.4345, -5.9245, 3.7944, 0.5179],
        ]);

        let expected_eigenvalues: Vec<f64> = vec![
            343544.6277001563,
            9897.625949808047,
            2063.519887011604,
            302.04806302399646,
        ];

        let pca = PCA::fit(&us_arrests, PCAParameters::default().with_n_components(4)).unwrap();

        assert!(relative_eq!(
            pca.eigenvectors.abs(),
            &expected_eigenvectors.abs(),
            epsilon = 1e-4
        ));

        for i in 0..pca.eigenvalues.len() {
            assert!((pca.eigenvalues[i].abs() - expected_eigenvalues[i].abs()).abs() < 1e-8);
        }

        let us_arrests_t = pca.transform(&us_arrests).unwrap();

        assert!(relative_eq!(
            us_arrests_t.abs(),
            &expected_projection.abs(),
            epsilon = 1e-4
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn decompose_correlation() {
        let us_arrests = us_arrests_data();

        let expected_eigenvectors = DenseMatrix::from_2d_array(&[
            &[
                0.124288601688222,
                -0.0969866877028367,
                0.0791404742697482,
                -0.150572299008293,
            ],
            &[
                0.00706888610512014,
                -0.00227861130898090,
                0.00325028101296307,
                0.00901099154845273,
            ],
            &[
                0.0194141494466002,
                0.060910660326921,
                0.0263806464184195,
                -0.0093429458365566,
            ],
            &[
                0.0586084532558777,
                0.0180450999787168,
                -0.0881962972508558,
                -0.0096011588898465,
            ],
        ]);

        let expected_projection = DenseMatrix::from_2d_array(&[
            &[0.9856, -1.1334, 0.4443, -0.1563],
            &[1.9501, -1.0732, -2.04, 0.4386],
            &[1.7632, 0.746, -0.0548, 0.8347],
            &[-0.1414, -1.1198, -0.1146, 0.1828],
            &[2.524, 1.5429, -0.5986, 0.342],
            &[1.5146, 0.9876, -1.095, -0.0015],
            &[-1.3586, 1.0889, 0.6433, 0.1185],
            &[0.0477, 0.3254, 0.7186, 0.882],
            &[3.013, -0.0392, 0.5768, 0.0963],
            &[1.6393, -1.2789, 0.3425, -1.0768],
            &[-0.9127, 1.5705, -0.0508, -0.9028],
            &[-1.6398, -0.211, -0.2598, 0.4991],
            &[1.3789, 0.6818, 0.6775, 0.122],
            &[-0.5055, 0.1516, -0.2281, -0.4247],
            &[-2.2536, 0.1041, -0.1646, -0.0176],
            &[-0.7969, 0.2702, -0.0256, -0.2065],
            &[-0.7509, -0.9584, 0.0284, -0.6706],
            &[1.5648, -0.8711, 0.7835, -0.4547],
            &[-2.3968, -0.3764, 0.0657, 0.3305],
            &[1.7634, -0.4277, 0.1573, 0.5591],
            &[-0.4862, 1.4745, 0.6095, 0.1796],
            &[2.1084, 0.1554, -0.3849, -0.1024],
            &[-1.6927, 0.6323, -0.1531, -0.0673],
            &[0.9965, -2.3938, 0.7408, -0.2155],
            &[0.6968, 0.2634, -0.3774, -0.2258],
            &[-1.1855, -0.5369, -0.2469, -0.1237],
            &[-1.2656, 0.194, -0.1756, -0.0159],
            &[2.8744, 0.7756, -1.1634, -0.3145],
            &[-2.3839, 0.0181, -0.0369, 0.0331],
            &[0.1816, 1.4495, 0.7645, -0.2434],
            &[1.98, -0.1428, -0.1837, 0.3395],
            &[1.6826, 0.8232, 0.6431, 0.0135],
            &[1.1234, -2.228, 0.8636, 0.9544],
            &[-2.9922, -0.5991, -0.3013, 0.254],
            &[-0.226, 0.7422, 0.0311, -0.4739],
            &[-0.3118, 0.2879, 0.0153, -0.0103],
            &[0.0591, 0.5414, -0.9398, 0.2378],
            &[-0.8884, 0.5711, 0.4006, -0.3591],
            &[-0.8638, 1.492, 1.3699, 0.6136],
            &[1.3207, -1.9334, 0.3005, 0.1315],
            &[-1.9878, -0.8233, -0.3893, 0.1096],
            &[0.9997, -0.8603, -0.1881, -0.6529],
            &[1.3551, 0.4125, 0.4921, -0.6432],
            &[-0.5506, 1.4715, -0.2937, 0.0823],
            &[-2.8014, -1.4023, -0.8413, 0.1449],
            &[-0.0963, -0.1997, -0.0117, -0.2114],
            &[-0.2169, 0.9701, -0.6249, 0.2208],
            &[-2.1086, -1.4248, -0.1048, -0.1319],
            &[-2.0797, 0.6113, 0.1389, -0.1841],
            &[-0.6294, -0.321, 0.2407, 0.1667],
        ]);

        let expected_eigenvalues: Vec<f64> = vec![
            2.480241579149493,
            0.9897651525398419,
            0.35656318058083064,
            0.1734300877298357,
        ];

        let pca = PCA::fit(
            &us_arrests,
            PCAParameters::default()
                .with_n_components(4)
                .with_use_correlation_matrix(true),
        )
        .unwrap();

        assert!(relative_eq!(
            pca.eigenvectors.abs(),
            &expected_eigenvectors.abs(),
            epsilon = 1e-4
        ));

        for i in 0..pca.eigenvalues.len() {
            assert!((pca.eigenvalues[i].abs() - expected_eigenvalues[i].abs()).abs() < 1e-8);
        }

        let us_arrests_t = pca.transform(&us_arrests).unwrap();

        assert!(relative_eq!(
            us_arrests_t.abs(),
            &expected_projection.abs(),
            epsilon = 1e-4
        ));
    }

    // Disable this test for now
    // TODO: implement deserialization for new DenseMatrix
    // #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    // #[test]
    // #[cfg(feature = "serde")]
    // fn pca_serde() {
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

    //     let pca = PCA::fit(&iris, Default::default()).unwrap();

    //     let deserialized_pca: PCA<f64, DenseMatrix<f64>> =
    //         serde_json::from_str(&serde_json::to_string(&pca).unwrap()).unwrap();

    //     assert_eq!(pca, deserialized_pca);
    // }
}
