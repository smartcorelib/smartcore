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
/// Calculate the offset to parameters to due introduction of one-hot encoding
fn find_new_idxs(num_params: usize, cat_sizes: &[usize], encoded_idxs: &[usize]) -> Vec<usize> {
    // This functions uses iterators and returns a vector.
    // In case we get a huge amount of paramenters this might be a problem
    // todo: Change this such that it will return an iterator

    let cat_idx = encoded_idxs.iter().copied().chain((num_params..).take(1));

    // Offset is constant between two categorical values, here we calculate the number of steps
    // that remain constant
    let repeats = cat_idx.scan(0, |a, v| {
        let im = v + 1 - *a;
        *a = v;
        Some(im)
    });

    // Calculate the offset to parameter idx due to newly intorduced one-hot vectors
    let offset_ = cat_sizes.iter().scan(0, |a, &v| {
        *a = *a + v - 1;
        Some(*a)
    });
    let offset = (0..1).chain(offset_);

    let new_param_idxs: Vec<usize> = (0..num_params)
        .zip(
            repeats
                .zip(offset)
                .map(|(r, o)| iter::repeat(o).take(r))
                .flatten(),
        )
        .map(|(idx, ofst)| idx + ofst)
        .collect();
    new_param_idxs
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