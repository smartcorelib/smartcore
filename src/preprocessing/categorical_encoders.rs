//! # One-hot Encoding For [RealNumber](../../math/num/trait.RealNumber.html) Matricies
//! Transform a data [Matrix](../../linalg/trait.BaseMatrix.html) by replacing all categorical variables with their one-hot equivalents
//!
//! ### Usage Example
//! ```
//! use smartcore::linalg::naive::dense_matrix::DenseMatrix;
//! use smartcore::preprocessing::categorical_encoder::{OneHotEncoder, OneHotEncoderParams};
//! let data = DenseMatrix::from_2d_array(&[
//!         &[1.5, 1.0, 1.5, 3.0],
//!         &[1.5, 2.0, 1.5, 4.0],
//!         &[1.5, 1.0, 1.5, 5.0],
//!         &[1.5, 2.0, 1.5, 6.0],
//!   ]);
//! let encoder_params = OneHotEncoderParams::from_cat_idx(&[1, 3]);
//! // Infer number of categories from data and return a reusable encoder
//! let encoder = OneHotEncoder::fit(&data, encoder_params).unwrap();
//! // Transform categorical to one-hot encoded (can transform similar)
//! let oh_data = encoder.transform(&data).unwrap();
//! // Produces the following:
//! //    &[1.5, 1.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0]
//! //    &[1.5, 0.0, 1.0, 1.5, 0.0, 1.0, 0.0, 0.0]
//! //    &[1.5, 1.0, 0.0, 1.5, 0.0, 0.0, 1.0, 0.0]
//! //    &[1.5, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1.0]
//! ```

use crate::error::Failed;
use crate::linalg::{BaseVector, Matrix};
use crate::math::num::RealNumber;

use crate::preprocessing::series_encoder::SeriesOneHotEncoder;

pub type HashableReal = u32;

fn hashable_num<T: RealNumber>(v: &T) -> HashableReal {
    // gaxler: If first 32 bits are the same, assume numbers are the same for the categorical coercion
    v.to_f32_bits()
}

#[derive(Debug, Clone)]
pub struct OneHotEncoderParams {
    pub categorical_param_idxs: Option<Vec<usize>>,
    pub infer_categorical: bool,
}
/// Encode Categorical variavbles of data matrix to one-hot
pub struct OneHotEncoder {
    series_encoders: Vec<SeriesOneHotEncoder<HashableReal>>,
    categorical_param_idxs: Vec<usize>,
}

impl<T: RealNumber, M: Matrix<T>> OneHotEncoder {
    /// PlaceHolder

    pub fn fit(data: &M, params: OneHotEncoderParams) -> Result<OneHotEncoder, Failed> {
        match (params.categorical_param_idxs, params.infer_categorical) {
            (None, false) => Err(Failed::fit(
                "Must pass categorical series ids or infer flag",
            )),

            (Some(idxs), true) => Err(Failed::fit(
                "Ambigous parameters, got both infer and categroy ids",
            )),

            (Some(idxs), false) => Ok(Self {
                series_encoders: Self::build_series_encoders::<T, M>(data, &idxs[..]),
                categorical_param_idxs: idxs,
            }),

            (None, true) => {
                todo!("implement categorical auto-inference")
            }
        }
    }

    fn build_series_encoders(data: &M, idxs: &[usize]) -> Vec<SeriesOneHotEncoder<HashableReal>> {
        let (nrows, _) = data.shape();
        // let mut res: Vec<SeriesOneHotEncoder<HashableReal>> = Vec::with_capacity(idxs.len());
        let mut tmp_col: Vec<T> = Vec::with_capacity(nrows);

        let res: Vec<SeriesOneHotEncoder<HashableReal>> = idxs
            .iter()
            .map(|&idx| {
                data.copy_col_as_vec(idx, &mut tmp_col);
                let hashable_col = tmp_col.iter().map(|v| hashable_num::<T>(v));
                SeriesOneHotEncoder::fit_to_iter(hashable_col)
            })
            .collect();
        res
    }


}