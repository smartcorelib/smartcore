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
fn validate_col_is_categorical<T:  Categorizable>(data: &Vec<T>) -> bool {
    for v in data {
        if !v.is_valid() { return false}
    }
    true
}
/// Encode Categorical variavbles of data matrix to one-hot
pub struct OneHotEncoder {
    series_encoders: Vec<SeriesOneHotEncoder<CategoricalFloat>>,
    col_idx_categorical: Vec<usize>,
}

impl OneHotEncoder {
    /// PlaceHolder

    pub fn fit<T:  Categorizable, M: Matrix<T>>(
        data: &M,
        params: OneHotEncoderParams,
    ) -> Result<OneHotEncoder, Failed> {
        match (params.col_idx_categorical, params.infer_categorical) {
            (None, false) => Err(Failed::fit(
                "Must pass categorical series ids or infer flag",
            )),

            (Some(_idxs), true) => Err(Failed::fit(
                "Ambigous parameters, got both infer and categroy ids",
            )),

            (Some(mut idxs), false) => {
                // make sure categories have same order as data columns
                idxs.sort();

                let (nrows, _) = data.shape();

                // col buffer to avoid allocations
                let mut col_buf: Vec<T> = iter::repeat(T::zero()).take(nrows).collect();
        
                let mut res: Vec<SeriesOneHotEncoder<CategoricalFloat>> = Vec::with_capacity(idxs.len());
                
                for &idx in &idxs {
                    data.copy_col_as_vec(idx, &mut col_buf);
                    if !validate_col_is_categorical(&col_buf) {
                        let msg = format!("Column {} of data matrix containts non categorizable (integer) values", idx);
                        return Err(Failed::fit(&msg[..]))
                    }
                    let hashable_col = col_buf.iter().map(|v| v.to_category());
                    res.push(SeriesOneHotEncoder::fit_to_iter(hashable_col));
                }

                Ok(Self {
                    series_encoders: res, //Self::build_series_encoders::<T, M>(data, &idxs[..]),
                    col_idx_categorical: idxs,
                })
            }

            (None, true) => {
                todo!("Auto-Inference for Categorical Variables not yet implemented")
            }
        }
    }

    /// Transform categorical variables to one-hot encoded and return a new matrix
    pub fn transform<T:  Categorizable, M: Matrix<T>>(&self, x: &M) -> Option<M> {
        let (nrows, p) = x.shape();
        let additional_params: Vec<usize> = self
            .series_encoders
            .iter()
            .map(|enc| enc.num_categories)
            .collect();

        let new_param_num: usize = p + additional_params.iter().fold(0, |cs, &v| cs + v - 1);
        let new_col_idx = find_new_idxs(p, &additional_params[..], &self.col_idx_categorical[..]);
        let mut res = M::zeros(nrows, new_param_num);
        // copy old data in x to their new location
        for (old_p, &new_p) in new_col_idx.iter().enumerate() {
            for r in 0..nrows {
                let val = x.get(r, old_p);
                res.set(r, new_p, val);
            }
        }
        for (pidx, &old_cidx) in self.col_idx_categorical.iter().enumerate() {
            let cidx = new_col_idx[old_cidx];
            let col_iter = (0..nrows).map(|r| res.get(r, cidx).to_category());
            let sencoder = &self.series_encoders[pidx];
            let oh_series: Vec<Option<Vec<T>>> = sencoder.transform_iter(col_iter);

            for (row, oh_vec) in oh_series.iter().enumerate() {
                match oh_vec {
                    None => {
                        // Bad value in a series causes in to be invalid
                        // todo: proper error handling, so user can know where the bad value is
                        return None;
            }
                    Some(v) => {
                        // copy one hot vectors to their place in the data matrix;
                        for (col_ofst, &val) in v.iter().enumerate() {
                            res.set(row, cidx + col_ofst, val);
        }
    }
                }
            }
        }
        Some(res)
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