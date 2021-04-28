#![allow(clippy::ptr_arg)]
//! # Series Encoder
//! Encode a series of categorical features as a one-hot numeric array.

use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::math::num::RealNumber;
use std::collections::HashMap;
use std::hash::Hash;

/// ## Bi-directional map category <-> label num.
/// Turn Hashable objects into a one-hot vectors or ordinal values.
/// This struct encodes single class per exmample
///
/// You can fit_to_iter a category enumeration by passing an iterator of categories.
/// category numbers will be assigned in the order they are encountered
///
/// Example:
/// ```
/// use std::collections::HashMap;
/// use smartcore::preprocessing::series_encoder::CategoryMapper;
///
/// let fake_categories: Vec<usize> = vec![1, 2, 3, 4, 5, 3, 5, 3, 1, 2, 4];
/// let it = fake_categories.iter().map(|&a| a);
/// let enc = CategoryMapper::<usize>::fit_to_iter(it);
/// let oh_vec: Vec<f64> = enc.get_one_hot(&1).unwrap();
/// // notice that 1 is actually a zero-th positional category
/// assert_eq!(oh_vec, vec![1.0, 0.0, 0.0, 0.0, 0.0]);
/// ```
///
/// You can also pass a predefined category enumeration such as a hashmap `HashMap<C, usize>` or a vector `Vec<C>`
///
///
/// ```
/// use std::collections::HashMap;
/// use smartcore::preprocessing::series_encoder::CategoryMapper;
///
/// let category_map: HashMap<&str, usize> =
/// vec![("cat", 2), ("background",0), ("dog", 1)]
/// .into_iter()
/// .collect();
/// let category_vec = vec!["background", "dog", "cat"];
///
/// let enc_lv  = CategoryMapper::<&str>::from_positional_category_vec(category_vec);
/// let enc_lm  = CategoryMapper::<&str>::from_category_map(category_map);
///
/// // ["background", "dog", "cat"]
/// println!("{:?}", enc_lv.get_categories());
/// let lv: Vec<f64> = enc_lv.get_one_hot(&"dog").unwrap();
/// let lm: Vec<f64> = enc_lm.get_one_hot(&"dog").unwrap();
/// assert_eq!(lv, lm);
/// ```
#[derive(Debug, Clone)]
pub struct CategoryMapper<C> {
    category_map: HashMap<C, usize>,
    categories: Vec<C>,
    num_categories: usize,
}

impl<C> CategoryMapper<C>
where
    C: Hash + Eq + Clone,
{
    /// Get the number of categories in the mapper
    pub fn num_categories(&self) -> usize {
        self.num_categories
    }

    /// Fit an encoder to a lable iterator
    pub fn fit_to_iter(categories: impl Iterator<Item = C>) -> Self {
        let mut category_map: HashMap<C, usize> = HashMap::new();
        let mut category_num = 0usize;
        let mut unique_lables: Vec<C> = Vec::new();

        for l in categories {
            if !category_map.contains_key(&l) {
                category_map.insert(l.clone(), category_num);
                unique_lables.push(l.clone());
                category_num += 1;
            }
        }
        Self {
            category_map,
            num_categories: category_num,
            categories: unique_lables,
        }
    }

    /// Build an encoder from a predefined (category -> class number) map
    pub fn from_category_map(category_map: HashMap<C, usize>) -> Self {
        let mut _unique_cat: Vec<(C, usize)> =
            category_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
        _unique_cat.sort_by(|a, b| a.1.cmp(&b.1));
        let categories: Vec<C> = _unique_cat.into_iter().map(|a| a.0).collect();
        Self {
            num_categories: categories.len(),
            categories,
            category_map,
        }
    }

    /// Build an encoder from a predefined positional category-class num vector
    pub fn from_positional_category_vec(categories: Vec<C>) -> Self {
        let category_map: HashMap<C, usize> = categories
            .iter()
            .enumerate()
            .map(|(v, k)| (k.clone(), v))
            .collect();
        Self {
            num_categories: categories.len(),
            category_map,
            categories,
        }
    }

    /// Get label num of a category
    pub fn get_num(&self, category: &C) -> Option<&usize> {
        self.category_map.get(category)
    }

    /// Return category corresponding to label num
    pub fn get_cat(&self, num: usize) -> &C {
        &self.categories[num]
    }

    /// List all categories (position = category number)
    pub fn get_categories(&self) -> &[C] {
        &self.categories[..]
    }

    /// Get one-hot encoding of the category
    pub fn get_one_hot<U, V>(&self, category: &C) -> Option<V>
    where
        U: RealNumber,
        V: BaseVector<U>,
    {
        match self.get_num(category) {
            None => None,
            Some(&idx) => Some(make_one_hot::<U, V>(idx, self.num_categories)),
        }
    }

    /// Invert one-hot vector, back to the category
    pub fn invert_one_hot<U, V>(&self, one_hot: V) -> Result<C, Failed>
    where
        U: RealNumber,
        V: BaseVector<U>,
    {
        let pos = U::one();

        let oh_it = (0..one_hot.len()).map(|idx| one_hot.get(idx));

        let s: Vec<usize> = oh_it
            .enumerate()
            .filter_map(|(idx, v)| if v == pos { Some(idx) } else { None })
            .collect();

        if s.len() == 1 {
            let idx = s[0];
            return Ok(self.get_cat(idx).clone());
        }
        let pos_entries = format!(
            "Expected a single positive entry, {} entires found",
            s.len()
        );
        Err(Failed::transform(&pos_entries[..]))
    }

    /// Get ordinal encoding of the catergory
    pub fn get_ordinal<U>(&self, category: &C) -> Option<U>
    where
        U: RealNumber,
    {
        match self.get_num(category) {
            None => None,
            Some(&idx) => U::from_usize(idx),
        }
    }
}

/// Make a one-hot encoded vector from a categorical variable
///
/// Example:
/// ```
/// use smartcore::preprocessing::series_encoder::make_one_hot;
/// let one_hot: Vec<f64> = make_one_hot(2, 3);
/// assert_eq!(one_hot, vec![0.0, 0.0, 1.0]);
/// ```
pub fn make_one_hot<T, V>(category_idx: usize, num_categories: usize) -> V
where
    T: RealNumber,
    V: BaseVector<T>,
{
    let pos = T::one();
    let mut z = V::zeros(num_categories);
    z.set(category_idx, pos);
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn from_categories() {
        let fake_categories: Vec<usize> = vec![1, 2, 3, 4, 5, 3, 5, 3, 1, 2, 4];
        let it = fake_categories.iter().map(|&a| a);
        let enc = CategoryMapper::<usize>::fit_to_iter(it);
        let oh_vec: Vec<f64> = match enc.get_one_hot(&1) {
            None => panic!("Wrong categories"),
            Some(v) => v,
        };
        let res: Vec<f64> = vec![1f64, 0f64, 0f64, 0f64, 0f64];
        assert_eq!(oh_vec, res);
    }

    fn build_fake_str_enc<'a>() -> CategoryMapper<&'a str> {
        let fake_category_pos = vec!["background", "dog", "cat"];
        let enc = CategoryMapper::<&str>::from_positional_category_vec(fake_category_pos);
        enc
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ordinal_encoding() {
        let enc = build_fake_str_enc();
        assert_eq!(1f64, enc.get_ordinal::<f64>(&"dog").unwrap())
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn category_map_and_vec() {
        let category_map: HashMap<&str, usize> = vec![("background", 0), ("dog", 1), ("cat", 2)]
            .into_iter()
            .collect();
        let enc = CategoryMapper::<&str>::from_category_map(category_map);
        let oh_vec: Vec<f64> = match enc.get_one_hot(&"dog") {
            None => panic!("Wrong categories"),
            Some(v) => v,
        };
        let res: Vec<f64> = vec![0f64, 1f64, 0f64];
        assert_eq!(oh_vec, res);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn positional_categories_vec() {
        let enc = build_fake_str_enc();
        let oh_vec: Vec<f64> = match enc.get_one_hot(&"dog") {
            None => panic!("Wrong categories"),
            Some(v) => v,
        };
        let res: Vec<f64> = vec![0.0, 1.0, 0.0];
        assert_eq!(oh_vec, res);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invert_label_test() {
        let enc = build_fake_str_enc();
        let res: Vec<f64> = vec![0.0, 1.0, 0.0];
        let lab = enc.invert_one_hot(res).unwrap();
        assert_eq!(lab, "dog");
        if let Err(e) = enc.invert_one_hot(vec![0.0, 0.0, 0.0]) {
            let pos_entries = format!("Expected a single positive entry, 0 entires found");
            assert_eq!(e, Failed::transform(&pos_entries[..]));
        };
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn test_many_categorys() {
        let enc = build_fake_str_enc();
        let cat_it = ["dog", "cat", "fish", "background"].iter().cloned();
        let res: Vec<Option<Vec<f64>>> = cat_it.map(|v| enc.get_one_hot(&v)).collect();
        let v = vec![
            Some(vec![0.0, 1.0, 0.0]),
            Some(vec![0.0, 0.0, 1.0]),
            None,
            Some(vec![1.0, 0.0, 0.0]),
        ];
        assert_eq!(res, v)
    }
}
