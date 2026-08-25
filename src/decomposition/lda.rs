//! # Linear Discriminant Analysis
//!
//! Linear discriminant analysis (LDA) finds the linear combinations of the features that best separate two or more classes.
//! Unlike [PCA](../pca/index.html), which is unsupervised and keeps the directions of largest variance, LDA is supervised: it uses the class
//! labels \\(y\\) to look for the directions that maximise the ratio of between-class scatter to within-class scatter (the Rayleigh quotient).
//! The result is a projection of the data onto at most \\(C - 1\\) axes, where \\(C\\) is the number of classes, that pulls the classes apart.
//!
//! LDA is often used to reduce the number of features before a classification step and for data visualization.
//! Each discriminant direction \\(w\\) is a generalized eigenvector of the pair \\((S_B, S_W)\\), the between-class and within-class scatter matrices:
//!
//! \\[S_B w = \lambda S_W w\\]
//!
//! The directions are kept in descending order of \\(\lambda\\), so the first component separates the classes the most.
//!
//! LDA assumes the classes share the same covariance. It needs the within-class scatter matrix to be invertible, so keep the number of
//! samples per class larger than the number of features.
//!
//! Example:
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::decomposition::lda::*;
//!
//! // Three well separated classes, three samples each
//! let x = DenseMatrix::from_2d_array(&[
//!                     &[4.0, 2.0, 0.6],
//!                     &[4.2, 2.1, 0.5],
//!                     &[3.9, 1.9, 0.7],
//!                     &[6.0, 3.0, 4.5],
//!                     &[6.2, 2.9, 4.6],
//!                     &[5.8, 3.1, 4.4],
//!                     &[7.5, 3.6, 6.1],
//!                     &[7.7, 3.5, 6.0],
//!                     &[7.3, 3.7, 6.2],
//!                     ]).unwrap();
//! let y = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
//!
//! let lda = LDA::fit(&x, &y, LDAParameters::default()).unwrap(); // keep C - 1 = 2 components
//!
//! let projected = lda.transform(&x).unwrap();
//! ```
//!
//! ## References:
//! * ["An Introduction to Statistical Learning", James G., Witten D., Hastie T., Tibshirani R., 4.4 Linear Discriminant Analysis](http://faculty.marshall.usc.edu/gareth-james/ISL/)
//! * ["Pattern Classification", Duda R.O., Hart P.E., Stork D.G., 2nd ed., 3.8.3 Multiple Discriminant Analysis](https://www.wiley.com/en-us/Pattern+Classification%2C+2nd+Edition-p-9780471056690)
//!
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::cmp::Ordering;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::api::Transformer;
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::linalg::traits::evd::EVDDecomposable;
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;

/// Linear discriminant analysis
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct LDA<T: Number + RealNumber, X: Array2<T> + EVDDecomposable<T>> {
    // Projection matrix, one column per kept discriminant direction (n_features x n_components).
    projection_matrix: X,
    // Generalized eigenvalue of every kept direction, largest first.
    eigenvalues: Vec<T>,
    // Number of features expected by `transform`.
    n_features: usize,
}

impl<T: Number + RealNumber, X: Array2<T> + EVDDecomposable<T>> PartialEq for LDA<T, X> {
    fn eq(&self, other: &Self) -> bool {
        let tol = T::from(1e-6).unwrap();
        if self.n_features != other.n_features
            || self.eigenvalues.len() != other.eigenvalues.len()
            || self
                .projection_matrix
                .iterator(0)
                .zip(other.projection_matrix.iterator(0))
                .any(|(&a, &b)| (a - b).abs() > tol)
        {
            return false;
        }
        for i in 0..self.eigenvalues.len() {
            if (self.eigenvalues[i] - other.eigenvalues[i]).abs() > tol {
                return false;
            }
        }
        true
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
/// LDA parameters
#[must_use]
pub struct LDAParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of discriminant directions to keep. When `None` the largest useful number,
    /// `min(n_classes - 1, n_features)`, is used.
    pub n_components: Option<usize>,
}

impl LDAParameters {
    /// Number of discriminant directions to keep.
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = Some(n_components);
        self
    }
}

/// LDA grid search parameters
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
#[must_use]
pub struct LDASearchParameters {
    #[cfg_attr(feature = "serde", serde(default))]
    /// Number of discriminant directions to keep.
    pub n_components: Vec<Option<usize>>,
}

/// LDA grid search iterator
pub struct LDASearchParametersIterator {
    lda_search_parameters: LDASearchParameters,
    current_n_components: usize,
}

impl IntoIterator for LDASearchParameters {
    type Item = LDAParameters;
    type IntoIter = LDASearchParametersIterator;

    fn into_iter(self) -> Self::IntoIter {
        LDASearchParametersIterator {
            lda_search_parameters: self,
            current_n_components: 0,
        }
    }
}

impl Iterator for LDASearchParametersIterator {
    type Item = LDAParameters;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_n_components == self.lda_search_parameters.n_components.len() {
            return None;
        }

        let next = LDAParameters {
            n_components: self.lda_search_parameters.n_components[self.current_n_components],
        };

        self.current_n_components += 1;

        Some(next)
    }
}

impl Default for LDASearchParameters {
    fn default() -> Self {
        let default_params = LDAParameters::default();

        LDASearchParameters {
            n_components: vec![default_params.n_components],
        }
    }
}

impl<T: Number + RealNumber, X: Array2<T> + EVDDecomposable<T>> Transformer<X> for LDA<T, X> {
    fn transform(&self, x: &X) -> Result<X, Failed> {
        self.transform(x)
    }
}

impl<T: Number + RealNumber, X: Array2<T> + EVDDecomposable<T>> LDA<T, X> {
    /// Fits LDA to your data.
    /// * `x` - _NxM_ matrix with _N_ observations and _M_ features in each observation.
    /// * `y` - class labels, one per observation.
    /// * `parameters` - other parameters, use `Default::default()` to set parameters to default values.
    pub fn fit<TY: Number + Ord, Y: Array1<TY>>(
        x: &X,
        y: &Y,
        parameters: LDAParameters,
    ) -> Result<LDA<T, X>, Failed> {
        let (n, m) = x.shape();

        if n != y.shape() {
            return Err(Failed::fit(
                "Number of rows of X does not match the length of y",
            ));
        }

        let classes = y.unique();
        let k = classes.len();
        if k < 2 {
            return Err(Failed::fit(&format!(
                "Number of classes should be at least 2, got {k}"
            )));
        }

        let max_components = (k - 1).min(m);
        let n_components = match parameters.n_components {
            Some(c) => {
                if c < 1 || c > max_components {
                    return Err(Failed::fit(&format!(
                        "n_components should be between 1 and {max_components}, got {c}"
                    )));
                }
                c
            }
            None => max_components,
        };

        let class_of: Vec<usize> = (0..n)
            .map(|i| {
                let yi = y.get(i);
                classes.iter().position(|c| yi == c).unwrap()
            })
            .collect();

        // Class means and overall mean.
        let mut means = vec![vec![T::zero(); m]; k];
        let mut counts = vec![0usize; k];
        let mut mu = vec![T::zero(); m];
        for (i, &ci) in class_of.iter().enumerate() {
            counts[ci] += 1;
            for j in 0..m {
                let v = *x.get((i, j));
                means[ci][j] += v;
                mu[j] += v;
            }
        }
        for (c, count) in counts.iter().enumerate() {
            let denom = T::from(*count).unwrap();
            for mean_cj in means[c].iter_mut().take(m) {
                *mean_cj /= denom;
            }
        }
        let n_t = T::from(n).unwrap();
        for mu_j in mu.iter_mut().take(m) {
            *mu_j /= n_t;
        }

        // Within-class scatter Sw and total scatter St, both divided by N to match the
        // prior weighted covariance formulation. Sb (between-class scatter) is St - Sw.
        let mut sw = X::zeros(m, m);
        let mut st = X::zeros(m, m);
        for (i, &ci) in class_of.iter().enumerate() {
            for a in 0..m {
                let wa = *x.get((i, a)) - means[ci][a];
                let ta = *x.get((i, a)) - mu[a];
                for b in 0..m {
                    let wb = *x.get((i, b)) - means[ci][b];
                    let tb = *x.get((i, b)) - mu[b];
                    sw.add_element_mut((a, b), wa * wb);
                    st.add_element_mut((a, b), ta * tb);
                }
            }
        }
        for a in 0..m {
            for b in 0..m {
                sw.div_element_mut((a, b), n_t);
                st.div_element_mut((a, b), n_t);
            }
        }
        let mut sb = X::zeros(m, m);
        for a in 0..m {
            for b in 0..m {
                sb.set((a, b), *st.get((a, b)) - *sw.get((a, b)));
            }
        }

        // Solve the generalized eigenproblem Sb w = lambda Sw w by whitening with Sw.
        // Sw is symmetric positive definite when it is not singular, so its eigen
        // decomposition gives Sw^(-1/2) = U diag(1/sqrt(l)) U^T.
        let sw_evd = sw.evd(true)?;
        let u = sw_evd.V;
        let l = sw_evd.d;
        let l_max = l
            .iter()
            .cloned()
            .fold(T::zero(), |a, b| if b > a { b } else { a });
        let tol = T::from(1e-10).unwrap().max(T::from(1e-4).unwrap() * l_max);
        for &li in &l {
            if li <= tol {
                return Err(Failed::fit(
                    "Within-class scatter matrix is singular, provide more samples per class or fewer features",
                ));
            }
        }

        let mut sqrt_inv = X::zeros(m, m);
        for (j, &lj) in l.iter().enumerate() {
            sqrt_inv.set((j, j), T::one() / lj.sqrt());
        }
        let whitening = u.matmul(&sqrt_inv).matmul(&u.transpose());

        // A = Sw^(-1/2) Sb Sw^(-1/2) is symmetric, so a symmetric eigen decomposition is
        // enough and the generalized eigenvectors are whitening * W.
        let a = whitening.matmul(&sb).matmul(&whitening);
        let a_evd = a.evd(true)?;
        let w = a_evd.V;
        let g = a_evd.d;
        let directions = whitening.matmul(&w);

        // Keep the directions with the largest eigenvalues.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&i, &j| {
            if T::gt(&g[i], &g[j]) {
                Ordering::Less
            } else if T::gt(&g[j], &g[i]) {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });

        let mut projection_matrix = X::zeros(m, n_components);
        let mut eigenvalues = vec![T::zero(); n_components];
        for (col, &src) in order.iter().take(n_components).enumerate() {
            eigenvalues[col] = g[src];
            for row in 0..m {
                projection_matrix.set((row, col), *directions.get((row, src)));
            }
        }

        Ok(LDA {
            projection_matrix,
            eigenvalues,
            n_features: m,
        })
    }

    /// Project `x` onto the discriminant directions.
    /// * `x` - _KxM_ data where _K_ is number of observations and _M_ is number of features.
    pub fn transform(&self, x: &X) -> Result<X, Failed> {
        let (_, ncols) = x.shape();
        if ncols != self.n_features {
            return Err(Failed::transform(&format!(
                "Invalid input vector size: {}, expected: {}",
                ncols, self.n_features
            )));
        }
        Ok(x.matmul(&self.projection_matrix))
    }

    /// Get the projection matrix, one column per discriminant direction.
    pub fn scalings(&self) -> &X {
        &self.projection_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::Array;
    use crate::linalg::basic::matrix::DenseMatrix;
    use approx::relative_eq;

    fn three_class_data() -> (DenseMatrix<f64>, Vec<i32>) {
        let x = DenseMatrix::from_2d_array(&[
            &[4.0, 2.0, 0.6, 1.1],
            &[4.2, 2.1, 0.5, 1.0],
            &[3.9, 1.9, 0.7, 1.2],
            &[4.1, 2.2, 0.6, 0.9],
            &[4.3, 2.0, 0.4, 1.1],
            &[6.0, 3.0, 4.5, 1.5],
            &[6.2, 2.9, 4.6, 1.4],
            &[5.8, 3.1, 4.4, 1.6],
            &[6.1, 3.0, 4.7, 1.5],
            &[5.9, 2.8, 4.5, 1.3],
            &[7.5, 3.6, 6.1, 2.5],
            &[7.7, 3.5, 6.0, 2.3],
            &[7.3, 3.7, 6.2, 2.6],
            &[7.6, 3.4, 5.9, 2.4],
            &[7.4, 3.8, 6.3, 2.5],
        ])
        .unwrap();
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
        (x, y)
    }

    #[test]
    fn search_parameters() {
        let parameters = LDASearchParameters {
            n_components: vec![Some(1), Some(2)],
        };
        let mut iter = parameters.into_iter();
        assert_eq!(iter.next().unwrap().n_components, Some(1));
        assert_eq!(iter.next().unwrap().n_components, Some(2));
        assert!(iter.next().is_none());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn decompose_three_classes() {
        let (x, y) = three_class_data();

        // Reference values from scikit-learn 1.9.0 LinearDiscriminantAnalysis(solver="eigen").
        let expected_scalings = DenseMatrix::from_2d_array(&[
            &[6.389573900932, 4.224739978721],
            &[0.102734909256, 1.555841828755],
            &[7.696103067204, 5.276934744897],
            &[4.284298828330, 9.934345360635],
        ])
        .unwrap();

        let expected_projection = DenseMatrix::from_2d_array(&[
            &[34.683216336699, 27.772262622154],
            &[34.752817436407, 28.307053739200],
            &[35.252572627085, 27.659945502981],
            &[34.444766979275, 26.519035913650],
            &[35.060867893538, 30.095071564750],
            &[79.088150722732, 21.171277047506],
            &[80.717519417732, 20.339512849822],
            &[77.458782027733, 22.003041245191],
            &[81.266328726266, 20.538364096399],
            &[77.612880548824, 18.450765611756],
            &[105.208934364432, 29.933141881640],
            &[104.870652563157, 29.163330096872],
            &[105.118786282873, 29.709519130345],
            &[103.900788240102, 30.106399926678],
            &[106.088650605928, 28.766449300540],
        ])
        .unwrap();

        let lda = LDA::fit(&x, &y, LDAParameters::default()).unwrap();

        // Directions are only defined up to a sign, so compare magnitudes like the PCA tests do.
        assert!(relative_eq!(
            lda.scalings().abs(),
            &expected_scalings.abs(),
            epsilon = 1e-4
        ));

        let projected = lda.transform(&x).unwrap();
        assert!(relative_eq!(
            projected.abs(),
            &expected_projection.abs(),
            epsilon = 1e-4
        ));

        // Two well separated classes plus one give C - 1 = 2 directions with positive eigenvalues.
        assert_eq!(lda.eigenvalues.len(), 2);
        assert!((lda.eigenvalues[0] - 840.4907762345).abs() / 840.4907762345 < 1e-3);
        assert!((lda.eigenvalues[1] - 15.67633793586).abs() / 15.67633793586 < 1e-3);
        assert!(lda.eigenvalues[0] > lda.eigenvalues[1]);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn two_classes_separate_on_first_axis() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0, 2.0],
            &[1.2, 1.8],
            &[0.8, 2.2],
            &[1.1, 2.1],
            &[5.0, 6.0],
            &[5.2, 5.8],
            &[4.8, 6.2],
            &[5.1, 6.1],
        ])
        .unwrap();
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let lda = LDA::fit(&x, &y, LDAParameters::default()).unwrap();
        // Two classes give a single discriminant direction.
        let projected = lda.transform(&x).unwrap();
        assert_eq!(projected.shape(), (8, 1));

        // A nearest centroid rule on the single axis separates the two classes perfectly.
        let mean0 = (0..4).map(|i| *projected.get((i, 0))).sum::<f64>() / 4.0;
        let mean1 = (4..8).map(|i| *projected.get((i, 0))).sum::<f64>() / 4.0;
        for i in 0..8 {
            let v = *projected.get((i, 0));
            let predicted = if (v - mean0).abs() <= (v - mean1).abs() {
                0
            } else {
                1
            };
            assert_eq!(predicted, y[i]);
        }
    }

    #[test]
    fn too_many_components_is_rejected() {
        let (x, y) = three_class_data();
        // Only C - 1 = 2 directions exist.
        let result = LDA::fit(&x, &y, LDAParameters::default().with_n_components(3));
        assert!(result.is_err());
    }

    #[test]
    fn single_class_is_rejected() {
        let x = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[1.1, 2.1], &[0.9, 1.9]]).unwrap();
        let y = vec![0, 0, 0];
        let result = LDA::fit(&x, &y, LDAParameters::default());
        assert!(result.is_err());
    }

    #[test]
    fn mismatched_x_y_is_rejected() {
        let (x, _) = three_class_data();
        let y_short = vec![0i32; x.shape().0 - 1];
        assert!(LDA::fit(&x, &y_short, LDAParameters::default()).is_err());
    }

    #[test]
    fn zero_n_components_is_rejected() {
        let (x, y) = three_class_data();
        assert!(LDA::fit(&x, &y, LDAParameters::default().with_n_components(0)).is_err());
    }

    #[test]
    fn transform_wrong_features_is_rejected() {
        let (x, y) = three_class_data();
        let lda = LDA::fit(&x, &y, LDAParameters::default()).unwrap();
        let bad = DenseMatrix::<f64>::zeros(3, 2);
        assert!(lda.transform(&bad).is_err());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    #[cfg(feature = "serde")]
    fn serde() {
        let (x, y) = three_class_data();

        let lda = LDA::fit(&x, &y, LDAParameters::default()).unwrap();

        let deserialized_lda: LDA<f64, DenseMatrix<f64>> =
            postcard::from_bytes(&postcard::to_allocvec(&lda).unwrap()).unwrap();

        assert_eq!(lda, deserialized_lda);
    }
}
