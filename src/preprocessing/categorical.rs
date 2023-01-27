//! # One-hot Encoding For [RealNumber](../../math/num/trait.RealNumber.html) Matricies
//! Transform a data [Matrix](../../linalg/trait.BaseMatrix.html) by replacing all categorical variables with their one-hot equivalents
//!
//! Internally OneHotEncoder treats every categorical column as a series and transforms it using [CategoryMapper](../series_encoder/struct.CategoryMapper.html)
//!
//! ### Usage Example
//! ```
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::preprocessing::categorical::{OneHotEncoder, OneHotEncoderParams};
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
use std::iter;

use crate::error::Failed;
use crate::linalg::basic::arrays::Array2;

use crate::preprocessing::series_encoder::CategoryMapper;
use crate::preprocessing::traits::{CategoricalFloat, Categorizable};

/// OneHotEncoder Parameters
#[derive(Debug, Clone)]
pub struct OneHotEncoderParams {
    /// Column number that contain categorical variable
    pub col_idx_categorical: Option<Vec<usize>>,
    /// (Currently not implemented) Try and infer which of the matrix columns are categorical variables
    infer_categorical: bool,
}

impl OneHotEncoderParams {
    /// Generate parameters from categorical variable column numbers
    pub fn from_cat_idx(categorical_params: &[usize]) -> Self {
        Self {
            col_idx_categorical: Some(categorical_params.to_vec()),
            infer_categorical: false,
        }
    }
}

/// Calculate the offset to parameters to due introduction of one-hot encoding
fn find_new_idxs(num_params: usize, cat_sizes: &[usize], cat_idxs: &[usize]) -> Vec<usize> {
    // This functions uses iterators and returns a vector.
    // In case we get a huge amount of paramenters this might be a problem
    // todo: Change this such that it will return an iterator

    let cat_idx = cat_idxs.iter().copied().chain((num_params..).take(1));

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
                .flat_map(|(r, o)| iter::repeat(o).take(r)),
        )
        .map(|(idx, ofst)| idx + ofst)
        .collect();
    new_param_idxs
}

fn validate_col_is_categorical<T: Categorizable>(data: &[T]) -> bool {
    for v in data {
        if !v.is_valid() {
            return false;
        }
    }
    true
}

/// Encode Categorical variavbles of data matrix to one-hot
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    category_mappers: Vec<CategoryMapper<CategoricalFloat>>,
    col_idx_categorical: Vec<usize>,
}

impl OneHotEncoder {
    /// Create an encoder instance with categories infered from data matrix
    pub fn fit<T, M>(data: &M, params: OneHotEncoderParams) -> Result<OneHotEncoder, Failed>
    where
        T: Categorizable,
        M: Array2<T>,
    {
        match (params.col_idx_categorical, params.infer_categorical) {
            (None, false) => Err(Failed::fit(
                "Must pass categorical series ids or infer flag",
            )),

            (Some(_idxs), true) => Err(Failed::fit(
                "Ambigous parameters, got both infer and categroy ids",
            )),

            (Some(mut idxs), false) => {
                // make sure categories have same order as data columns
                idxs.sort_unstable();

                let (nrows, _) = data.shape();

                // col buffer to avoid allocations
                let mut col_buf: Vec<T> = iter::repeat(T::zero()).take(nrows).collect();

                let mut res: Vec<CategoryMapper<CategoricalFloat>> = Vec::with_capacity(idxs.len());

                for &idx in &idxs {
                    data.copy_col_as_vec(idx, &mut col_buf);
                    if !validate_col_is_categorical(&col_buf) {
                        let msg = format!(
                            "Column {idx} of data matrix containts non categorizable (integer) values"
                        );
                        return Err(Failed::fit(&msg[..]));
                    }
                    let hashable_col = col_buf.iter().map(|v| v.to_category());
                    res.push(CategoryMapper::fit_to_iter(hashable_col));
                }

                Ok(Self {
                    category_mappers: res,
                    col_idx_categorical: idxs,
                })
            }

            (None, true) => {
                todo!("Auto-Inference for Categorical Variables not yet implemented")
            }
        }
    }

    /// Transform categorical variables to one-hot encoded and return a new matrix
    pub fn transform<T, M>(&self, x: &M) -> Result<M, Failed>
    where
        T: Categorizable,
        M: Array2<T>,
    {
        let (nrows, p) = x.shape();
        let additional_params: Vec<usize> = self
            .category_mappers
            .iter()
            .map(|enc| enc.num_categories())
            .collect();

        // Eac category of size v adds v-1 params
        let expandws_p: usize = p + additional_params.iter().fold(0, |cs, &v| cs + v - 1);

        let new_col_idx = find_new_idxs(p, &additional_params[..], &self.col_idx_categorical[..]);
        let mut res = M::zeros(nrows, expandws_p);

        for (pidx, &old_cidx) in self.col_idx_categorical.iter().enumerate() {
            let cidx = new_col_idx[old_cidx];
            let col_iter = (0..nrows).map(|r| x.get((r, old_cidx)).to_category());
            let sencoder = &self.category_mappers[pidx];
            let oh_series = col_iter.map(|c| sencoder.get_one_hot::<T, Vec<T>>(&c));

            for (row, oh_vec) in oh_series.enumerate() {
                match oh_vec {
                    None => {
                        // Since we support T types, bad value in a series causes in to be invalid
                        let msg = format!("At least one value in column {old_cidx} doesn't conform to category definition");
                        return Err(Failed::transform(&msg[..]));
                    }
                    Some(v) => {
                        // copy one hot vectors to their place in the data matrix;
                        for (col_ofst, &val) in v.iter().enumerate() {
                            res.set((row, cidx + col_ofst), val);
                        }
                    }
                }
            }
        }

        // copy old data in x to their new location while skipping catergorical vars (already treated)
        let mut skip_idx_iter = self.col_idx_categorical.iter();
        let mut cur_skip = skip_idx_iter.next();

        for (old_p, &new_p) in new_col_idx.iter().enumerate() {
            // if found treated varible, skip it
            if let Some(&v) = cur_skip {
                if v == old_p {
                    cur_skip = skip_idx_iter.next();
                    continue;
                }
            }

            for r in 0..nrows {
                let val = x.get((r, old_p));
                res.set((r, new_p), *val);
            }
        }

        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::preprocessing::series_encoder::CategoryMapper;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn adjust_idxs() {
        assert_eq!(find_new_idxs(0, &[], &[]), Vec::<usize>::new());
        // [0,1,2] -> [0, 1, 1, 1, 2]
        assert_eq!(find_new_idxs(3, &[3], &[1]), vec![0, 1, 4]);
    }

    fn build_cat_first_and_last() -> (DenseMatrix<f64>, DenseMatrix<f64>) {
        let orig = DenseMatrix::from_2d_array(&[
            &[1.0, 1.5, 3.0],
            &[2.0, 1.5, 4.0],
            &[1.0, 1.5, 5.0],
            &[2.0, 1.5, 6.0],
        ]);

        let oh_enc = DenseMatrix::from_2d_array(&[
            &[1.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 1.5, 0.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 1.5, 0.0, 0.0, 1.0, 0.0],
            &[0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1.0],
        ]);

        (orig, oh_enc)
    }

    fn build_fake_matrix() -> (DenseMatrix<f64>, DenseMatrix<f64>) {
        // Categorical first and last
        let orig = DenseMatrix::from_2d_array(&[
            &[1.5, 1.0, 1.5, 3.0],
            &[1.5, 2.0, 1.5, 4.0],
            &[1.5, 1.0, 1.5, 5.0],
            &[1.5, 2.0, 1.5, 6.0],
        ]);

        let oh_enc = DenseMatrix::from_2d_array(&[
            &[1.5, 1.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0],
            &[1.5, 0.0, 1.0, 1.5, 0.0, 1.0, 0.0, 0.0],
            &[1.5, 1.0, 0.0, 1.5, 0.0, 0.0, 1.0, 0.0],
            &[1.5, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1.0],
        ]);

        (orig, oh_enc)
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn hash_encode_f64_series() {
        let series = vec![3.0, 1.0, 2.0, 1.0];
        let hashable_series: Vec<CategoricalFloat> =
            series.iter().map(|v| v.to_category()).collect();
        let enc = CategoryMapper::from_positional_category_vec(hashable_series);
        let inv = enc.invert_one_hot(vec![0.0, 0.0, 1.0]);
        let orig_val: f64 = inv.unwrap().into();
        assert_eq!(orig_val, 2.0);
    }
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn test_fit() {
        let (x, _) = build_fake_matrix();
        let params = OneHotEncoderParams::from_cat_idx(&[1, 3]);
        let oh_enc = OneHotEncoder::fit(&x, params).unwrap();
        assert_eq!(oh_enc.category_mappers.len(), 2);

        let num_cat: Vec<usize> = oh_enc
            .category_mappers
            .iter()
            .map(|a| a.num_categories())
            .collect();
        assert_eq!(num_cat, vec![2, 4]);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn matrix_transform_test() {
        let (x, expected_x) = build_fake_matrix();
        let params = OneHotEncoderParams::from_cat_idx(&[1, 3]);
        let oh_enc = OneHotEncoder::fit(&x, params).unwrap();
        let nm = oh_enc.transform(&x).unwrap();
        assert_eq!(nm, expected_x);

        let (x, expected_x) = build_cat_first_and_last();
        let params = OneHotEncoderParams::from_cat_idx(&[0, 2]);
        let oh_enc = OneHotEncoder::fit(&x, params).unwrap();
        let nm = oh_enc.transform(&x).unwrap();
        assert_eq!(nm, expected_x);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn fail_on_bad_category() {
        let m = DenseMatrix::from_2d_array(&[
            &[1.0, 1.5, 3.0],
            &[2.0, 1.5, 4.0],
            &[1.0, 1.5, 5.0],
            &[2.0, 1.5, 6.0],
        ]);

        let params = OneHotEncoderParams::from_cat_idx(&[1]);
        let result = OneHotEncoder::fit(&m, params);
        assert!(result.is_err());
    }
}
