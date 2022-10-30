//! # Standard-Scaling For [RealNumber](../../math/num/trait.RealNumber.html) Matricies
//! Transform a data [Matrix](../../linalg/trait.BaseMatrix.html) by removing the mean and scaling to unit variance.
//!
//! ### Usage Example
//! ```
//! use smartcore::api::{Transformer, UnsupervisedEstimator};
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::preprocessing::numerical;
//! let data = DenseMatrix::from_2d_vec(&vec![
//!     vec![0.0, 0.0],
//!     vec![0.0, 0.0],
//!     vec![1.0, 1.0],
//!     vec![1.0, 1.0],
//! ]);
//!
//! let standard_scaler =
//! numerical::StandardScaler::fit(&data, numerical::StandardScalerParameters::default())
//!    .unwrap();
//! let transformed_data = standard_scaler.transform(&data).unwrap();
//! assert_eq!(
//!     transformed_data,
//!     DenseMatrix::from_2d_vec(&vec![
//!         vec![-1.0, -1.0],
//!         vec![-1.0, -1.0],
//!         vec![1.0, 1.0],
//!         vec![1.0, 1.0],
//!     ])
//! );
//! ```
use std::marker::PhantomData;

use crate::api::{Transformer, UnsupervisedEstimator};
use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::Array2;
use crate::numbers::basenum::Number;
use crate::numbers::realnum::RealNumber;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configure Behaviour of `StandardScaler`.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Copy, Eq, PartialEq)]
pub struct StandardScalerParameters {
    /// Optionaly adjust mean to be zero.
    with_mean: bool,
    /// Optionally adjust standard-deviation to be one.
    with_std: bool,
}
impl Default for StandardScalerParameters {
    fn default() -> Self {
        StandardScalerParameters {
            with_mean: true,
            with_std: true,
        }
    }
}

/// With the `StandardScaler` data can be adjusted so
/// that every column has a mean of zero and a standard
/// deviation of one. This can improve model training for
/// scaling sensitive models like neural network or nearest
/// neighbors based models.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StandardScaler<T: Number + RealNumber> {
    means: Vec<f64>,
    stds: Vec<f64>,
    parameters: StandardScalerParameters,
    _phantom: PhantomData<T>,
}

#[allow(dead_code)]
impl<T: Number + RealNumber> StandardScaler<T> {
    fn new(parameters: StandardScalerParameters) -> Self
    where
        T: Number + RealNumber,
    {
        Self {
            means: vec![],
            stds: vec![],
            parameters: StandardScalerParameters {
                with_mean: parameters.with_mean,
                with_std: parameters.with_std,
            },
            _phantom: PhantomData,
        }
    }
    /// When the mean should be adjusted, the column mean
    /// should be kept. Otherwise, replace it by zero.
    fn adjust_column_mean(&self, mean: f64) -> f64 {
        if self.parameters.with_mean {
            mean
        } else {
            0f64
        }
    }
    /// When the standard-deviation should be adjusted, the column
    /// standard-deviation should be kept. Otherwise, replace it by one.
    fn adjust_column_std(&self, std: f64) -> f64 {
        if self.parameters.with_std {
            ensure_std_valid(std)
        } else {
            1f64
        }
    }
}

/// Make sure the standard deviation is valid. If it is
/// negative or zero, it should replaced by the smallest
/// positive value the type can have. That way we can savely
/// divide the columns with the resulting scalar.
fn ensure_std_valid<T: Number + RealNumber>(value: T) -> T {
    value.max(T::min_positive_value())
}

/// During `fit` the `StandardScaler` computes the column means and standard deviation.
impl<T: Number + RealNumber, M: Array2<T>> UnsupervisedEstimator<M, StandardScalerParameters>
    for StandardScaler<T>
{
    fn fit(x: &M, parameters: StandardScalerParameters) -> Result<Self, Failed>
    where
        T: Number + RealNumber,
        M: Array2<T>,
    {
        Ok(Self {
            means: x.column_mean(),
            stds: x.std_dev(0),
            parameters,
            _phantom: Default::default(),
        })
    }
}

/// During `transform` the `StandardScaler` applies the summary statistics
/// computed during `fit` to set the mean of each column to zero and the
/// standard deviation to one.
impl<T: Number + RealNumber, M: Array2<T>> Transformer<M> for StandardScaler<T> {
    fn transform(&self, x: &M) -> Result<M, Failed> {
        let (_, n_cols) = x.shape();
        if n_cols != self.means.len() {
            return Err(Failed::because(
                FailedError::TransformFailed,
                &format!(
                    "Expected {} columns, but got {} columns instead.",
                    self.means.len(),
                    n_cols,
                ),
            ));
        }

        Ok(build_matrix_from_columns(
            self.means
                .iter()
                .zip(self.stds.iter())
                .enumerate()
                .map(|(column_index, (column_mean, column_std))| {
                    x.take_column(column_index)
                        .sub_scalar(T::from(self.adjust_column_mean(*column_mean)).unwrap())
                        .div_scalar(T::from(self.adjust_column_std(*column_std)).unwrap())
                })
                .collect(),
        )
        .unwrap())
    }
}

/// From a collection of matrices, that contain columns, construct
/// a matrix by stacking the columns horizontally.
fn build_matrix_from_columns<T, M>(columns: Vec<M>) -> Option<M>
where
    T: Number + RealNumber,
    M: Array2<T>,
{
    if let Some(output_matrix) = columns.first().cloned() {
        return Some(
            columns
                .iter()
                .skip(1)
                .fold(output_matrix, |current_matrix, new_colum| {
                    current_matrix.h_stack(new_colum)
                }),
        );
    } else {
        None
    }
}

#[cfg(test)]
mod tests {

    mod helper_functionality {
        use super::super::{build_matrix_from_columns, ensure_std_valid};
        use crate::linalg::basic::matrix::DenseMatrix;

        #[test]
        fn combine_three_columns() {
            assert_eq!(
                build_matrix_from_columns(vec![
                    DenseMatrix::from_2d_vec(&vec![vec![1.0], vec![1.0], vec![1.0],]),
                    DenseMatrix::from_2d_vec(&vec![vec![2.0], vec![2.0], vec![2.0],]),
                    DenseMatrix::from_2d_vec(&vec![vec![3.0], vec![3.0], vec![3.0],])
                ]),
                Some(DenseMatrix::from_2d_vec(&vec![
                    vec![1.0, 2.0, 3.0],
                    vec![1.0, 2.0, 3.0],
                    vec![1.0, 2.0, 3.0]
                ]))
            )
        }

        #[test]
        fn negative_value_should_be_replace_with_minimal_positive_value() {
            assert_eq!(ensure_std_valid(-1.0), f64::MIN_POSITIVE)
        }

        #[test]
        fn zero_should_be_replace_with_minimal_positive_value() {
            assert_eq!(ensure_std_valid(0.0), f64::MIN_POSITIVE)
        }
    }
    mod standard_scaler {
        use super::super::{StandardScaler, StandardScalerParameters};
        use crate::api::{Transformer, UnsupervisedEstimator};
        use crate::linalg::basic::arrays::Array2;
        use crate::linalg::basic::matrix::DenseMatrix;

        #[test]
        fn dont_adjust_mean_if_used() {
            assert_eq!(
                (StandardScaler::<f64>::new(StandardScalerParameters {
                    with_mean: true,
                    with_std: true
                }))
                .adjust_column_mean(1.0),
                1.0
            )
        }
        #[test]
        fn replace_mean_with_zero_if_not_used() {
            assert_eq!(
                (StandardScaler::<f64>::new(StandardScalerParameters {
                    with_mean: false,
                    with_std: true
                }))
                .adjust_column_mean(1.0),
                0.0
            )
        }
        #[test]
        fn dont_adjust_std_if_used() {
            assert_eq!(
                (StandardScaler::<f64>::new(StandardScalerParameters {
                    with_mean: true,
                    with_std: true
                }))
                .adjust_column_std(10.0),
                10.0
            )
        }
        #[test]
        fn replace_std_with_one_if_not_used() {
            assert_eq!(
                (StandardScaler::<f64>::new(StandardScalerParameters {
                    with_mean: true,
                    with_std: false
                }))
                .adjust_column_std(10.0),
                1.0
            )
        }

        /// Helper function to apply fit as well as transform at the same time.
        fn fit_transform_with_default_standard_scaler(
            values_to_be_transformed: &DenseMatrix<f64>,
        ) -> DenseMatrix<f64> {
            StandardScaler::fit(
                values_to_be_transformed,
                StandardScalerParameters::default(),
            )
            .unwrap()
            .transform(values_to_be_transformed)
            .unwrap()
        }

        /// Fit transform with random generated values, expected values taken from
        /// sklearn.
        #[test]
        fn fit_transform_random_values() {
            let transformed_values =
                fit_transform_with_default_standard_scaler(&DenseMatrix::from_2d_array(&[
                    &[0.1004222429, 0.2194113576, 0.9310663354, 0.3313593793],
                    &[0.2045493861, 0.1683865411, 0.5071506765, 0.7257355264],
                    &[0.5708488802, 0.1846414616, 0.9590802982, 0.5591871046],
                    &[0.8387612750, 0.5754861361, 0.5537109852, 0.1077646442],
                ]));
            println!("{}", transformed_values);
            assert!(transformed_values.approximate_eq(
                &DenseMatrix::from_2d_array(&[
                    &[-1.1154020653, -0.4031985330, 0.9284605204, -0.4271473866],
                    &[-0.7615464283, -0.7076698384, -1.1075452562, 1.2632979631],
                    &[0.4832504303, -0.6106747444, 1.0630075435, 0.5494084257],
                    &[1.3936980634, 1.7215431158, -0.8839228078, -1.3855590021],
                ]),
                1.0
            ))
        }

        /// Test `fit` and `transform` for a column with zero variance.
        #[test]
        fn fit_transform_with_zero_variance() {
            assert_eq!(
                fit_transform_with_default_standard_scaler(&DenseMatrix::from_2d_array(&[
                    &[1.0],
                    &[1.0],
                    &[1.0],
                    &[1.0]
                ])),
                DenseMatrix::from_2d_array(&[&[0.0], &[0.0], &[0.0], &[0.0]]),
                "When scaling values with zero variance, zero is expected as return value"
            )
        }

        /// Test `fit` for columns with nice summary statistics.
        #[test]
        fn fit_for_simple_values() {
            assert_eq!(
                StandardScaler::fit(
                    &DenseMatrix::from_2d_array(&[
                        &[1.0, 1.0, 1.0],
                        &[1.0, 2.0, 5.0],
                        &[1.0, 1.0, 1.0],
                        &[1.0, 2.0, 5.0]
                    ]),
                    StandardScalerParameters::default(),
                ),
                Ok(StandardScaler {
                    means: vec![1.0, 1.5, 3.0],
                    stds: vec![0.0, 0.5, 2.0],
                    parameters: StandardScalerParameters {
                        with_mean: true,
                        with_std: true
                    },
                    _phantom: Default::default(),
                })
            )
        }
        /// Test `fit` for random generated values.
        #[test]
        fn fit_for_random_values() {
            let fitted_scaler = StandardScaler::fit(
                &DenseMatrix::from_2d_array(&[
                    &[0.1004222429, 0.2194113576, 0.9310663354, 0.3313593793],
                    &[0.2045493861, 0.1683865411, 0.5071506765, 0.7257355264],
                    &[0.5708488802, 0.1846414616, 0.9590802982, 0.5591871046],
                    &[0.8387612750, 0.5754861361, 0.5537109852, 0.1077646442],
                ]),
                StandardScalerParameters::default(),
            )
            .unwrap();

            assert_eq!(
                fitted_scaler.means,
                vec![0.42864544605, 0.2869813741, 0.737752073825, 0.431011663625],
            );

            assert!(
                &DenseMatrix::<f64>::from_2d_vec(&vec![fitted_scaler.stds]).approximate_eq(
                    &DenseMatrix::from_2d_array(&[&[
                        0.29426447500954,
                        0.16758497615485,
                        0.20820945786863,
                        0.23329718831165
                    ],]),
                    0.00000000000001
                )
            )
        }

        /// If `with_std` is set to `false` the values should not be
        /// adjusted to have a std of one.
        #[test]
        fn transform_without_std() {
            let standard_scaler = StandardScaler {
                means: vec![1.0, 3.0],
                stds: vec![1.0, 2.0],
                parameters: StandardScalerParameters {
                    with_mean: true,
                    with_std: false,
                },
                _phantom: Default::default(),
            };

            assert_eq!(
                standard_scaler.transform(&DenseMatrix::from_2d_array(&[&[0.0, 2.0], &[2.0, 4.0]])),
                Ok(DenseMatrix::from_2d_array(&[&[-1.0, -1.0], &[1.0, 1.0]]))
            )
        }

        /// If `with_mean` is set to `false` the values should not be adjusted
        /// to have a mean of zero.
        #[test]
        fn transform_without_mean() {
            let standard_scaler = StandardScaler {
                means: vec![1.0, 2.0],
                stds: vec![2.0, 3.0],
                parameters: StandardScalerParameters {
                    with_mean: false,
                    with_std: true,
                },
                _phantom: Default::default(),
            };

            assert_eq!(
                standard_scaler
                    .transform(&DenseMatrix::from_2d_array(&[&[0.0, 9.0], &[4.0, 12.0]])),
                Ok(DenseMatrix::from_2d_array(&[&[0.0, 3.0], &[2.0, 4.0]]))
            )
        }

        /// Same as `fit_for_random_values` test, but using a `StandardScaler` that has been
        /// serialized and deserialized.
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
        #[test]
        #[cfg(feature = "serde")]
        fn serde_fit_for_random_values() {
            let fitted_scaler = StandardScaler::fit(
                &DenseMatrix::from_2d_array(&[
                    &[0.1004222429, 0.2194113576, 0.9310663354, 0.3313593793],
                    &[0.2045493861, 0.1683865411, 0.5071506765, 0.7257355264],
                    &[0.5708488802, 0.1846414616, 0.9590802982, 0.5591871046],
                    &[0.8387612750, 0.5754861361, 0.5537109852, 0.1077646442],
                ]),
                StandardScalerParameters::default(),
            )
            .unwrap();

            let deserialized_scaler: StandardScaler<f64> =
                serde_json::from_str(&serde_json::to_string(&fitted_scaler).unwrap()).unwrap();

            assert_eq!(
                deserialized_scaler.means,
                vec![0.42864544605, 0.2869813741, 0.737752073825, 0.431011663625],
            );

            assert!(
                &DenseMatrix::from_2d_vec(&vec![deserialized_scaler.stds]).approximate_eq(
                    &DenseMatrix::from_2d_array(&[&[
                        0.29426447500954,
                        0.16758497615485,
                        0.20820945786863,
                        0.23329718831165
                    ],]),
                    0.00000000000001
                )
            )
        }
    }
}
