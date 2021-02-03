#![allow(clippy::ptr_arg)]
//! # Series Encoder
//! Encode a series of categorical features as a one-hot numeric array.

use crate::error::Failed;
use crate::linalg::BaseVector;
use crate::math::num::RealNumber;
use std::collections::HashMap;
use std::hash::Hash;

/// Bi-directional map category <-> label num.
#[derive(Debug, Clone)]
pub struct CategoryMapper<C> {
    category_map: HashMap<C, usize>,
    categories: Vec<C>,
    num_categories: usize,
}

impl<'a, C> CategoryMapper<C> 
where
    C: 'a + Hash + Eq + Clone
{
    
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
}

/// Defines common behavior for series encoders(e.g. OneHot, Ordinal)
pub trait SeriesEncoder<C>:
    where 
        C: Hash + Eq + Clone
{
    /// Fit an encoder to a lable iterator
    fn fit_to_iter(categories: impl Iterator<Item = C>) -> Self;
    
    /// Number of categories for categorical variable
    fn num_categories(&self) -> usize;
     
    /// Transform a single category type into a one-hot vector
    fn transform_one<U: RealNumber, V: BaseVector<U>>(&self, category: &C) -> Option<V>;

    /// Invert one-hot vector, back to the category
    fn invert_one<U: RealNumber, V: BaseVector<U>>(&self,  one_hot: V) -> Result<C, Failed>;

    /// Get categories ordered by encoder's category enumeration
    fn get_categories(&self) -> &[C];

    /// Take an iterator as a series to transform
    /// None is returned if unknown category is encountered
    fn transform_iter<U: RealNumber, V: BaseVector<U>>(
        &self,
        cat_it: impl Iterator<Item = C>,
    ) -> Vec<Option<V>> {
        cat_it.map(|l| self.transform_one(&l)).collect()
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
pub fn make_one_hot<T: RealNumber, V: BaseVector<T>>(
    category_idx: usize,
    num_categories: usize,
) -> V {
    let pos = T::from_f64(1f64).unwrap();
    let mut z = V::zeros(num_categories);
    z.set(category_idx, pos);
    z
}

/// Turn a collection of Hashable objects into a one-hot vectors.
/// This struct encodes single class per exmample
///
/// You can fit_to_iter a category enumeration by passing an iterator of categories.
/// category numbers will be assigned in the order they are encountered
///
/// Example:
/// ```
/// use std::collections::HashMap;
/// use smartcore::preprocessing::series_encoder::{SeriesOneHotEncoder, SeriesEncoder};
///
/// let fake_categories: Vec<usize> = vec![1, 2, 3, 4, 5, 3, 5, 3, 1, 2, 4];
/// let it = fake_categories.iter().map(|&a| a);
/// let enc: SeriesOneHotEncoder::<usize> = SeriesEncoder::fit_to_iter(it);
/// let oh_vec: Vec<f64> = enc.transform_one(&1).unwrap();
/// // notice that 1 is actually a zero-th positional category
/// assert_eq!(oh_vec, vec![1.0, 0.0, 0.0, 0.0, 0.0]);
/// ```
///
/// You can also pass a predefined category enumeration such as a hashmap `HashMap<C, usize>` or a vector `Vec<C>`
///
///
/// ```
/// use std::collections::HashMap;
/// use smartcore::preprocessing::series_encoder::{SeriesOneHotEncoder, SeriesEncoder, CategoryMapper};
///
/// let category_map: HashMap<&str, usize> =
/// vec![("cat", 2), ("background",0), ("dog", 1)]
/// .into_iter()
/// .collect();
/// let category_vec = vec!["background", "dog", "cat"];
///
/// let enc_lv  = SeriesOneHotEncoder::<&str>::new(CategoryMapper::from_positional_category_vec(category_vec));
/// let enc_lm  = SeriesOneHotEncoder::<&str>::new(CategoryMapper::from_category_map(category_map));
///
/// // ["background", "dog", "cat"]
/// println!("{:?}", enc_lv.get_categories());
/// let lv: Vec<f64> = enc_lv.transform_one(&"dog").unwrap();
/// let lm: Vec<f64> = enc_lm.transform_one(&"dog").unwrap();
/// assert_eq!(lv, lm);
/// ```
#[derive(Debug, Clone)]
pub struct SeriesOneHotEncoder<C> {
    mapper: CategoryMapper<C>,
}

impl<C> SeriesOneHotEncoder<C> 
where 
    C: Hash + Eq + Clone
{
    /// Create SeriesEncoder form existing mapper
    pub fn new(mapper: CategoryMapper<C>) -> Self {
        Self {mapper}
    }
}

impl<C> SeriesEncoder<C> for SeriesOneHotEncoder<C> 
where 
    C: Hash + Eq + Clone
{
    

    fn fit_to_iter(categories: impl Iterator<Item = C>) -> Self {
        Self {mapper:CategoryMapper::fit_to_iter(categories)}
    }

    fn num_categories(&self) -> usize {
        self.mapper.num_categories
    }

    fn get_categories(&self) -> &[C] {
        self.mapper.get_categories()
    }

    fn invert_one<U, V>(&self, one_hot: V) -> Result<C, Failed>
    where
        U: RealNumber,
        V: BaseVector<U>

        {
            let pos = U::from_f64(1f64).unwrap();
    
            let oh_it = (0..one_hot.len()).map(|idx| one_hot.get(idx));
    
            let s: Vec<usize> = oh_it
                .enumerate()
                .filter_map(|(idx, v)| if v == pos { Some(idx) } else { None })
                .collect();
    
            if s.len() == 1 {
                let idx = s[0];
                return Ok(self.mapper.get_cat(idx).clone());
            }
            let pos_entries = format!(
                "Expected a single positive entry, {} entires found",
                s.len()
            );
            Err(Failed::transform(&pos_entries[..]))
        }

    fn transform_one<U, V>(&self, category: &C) -> Option<V> 
    where
        U: RealNumber,
        V: BaseVector<U>
    {
        match self.mapper.get_num(category) {
            None => None,
            Some(&idx) => Some(make_one_hot(idx, self.num_categories())),
        }
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_categories() {
        let fake_categories: Vec<usize> = vec![1, 2, 3, 4, 5, 3, 5, 3, 1, 2, 4];
        let it = fake_categories.iter().map(|&a| a);
        let enc = SeriesOneHotEncoder::<usize>::fit_to_iter(it);
        let oh_vec: Vec<f64> = match enc.transform_one(&1) {
            None => panic!("Wrong categories"),
            Some(v) => v,
        };
        let res: Vec<f64> = vec![1f64, 0f64, 0f64, 0f64, 0f64];
        assert_eq!(oh_vec, res);
    }

    fn build_fake_str_enc<'a>() -> SeriesOneHotEncoder<&'a str> {
        let fake_category_pos = vec!["background", "dog", "cat"];
        let enc = SeriesOneHotEncoder::<&str>::new( CategoryMapper::from_positional_category_vec(fake_category_pos));
        enc
    }

    #[test]
    fn category_map_and_vec() {
        let category_map: HashMap<&str, usize> = vec![("background", 0), ("dog", 1), ("cat", 2)]
            .into_iter()
            .collect();
        let enc = SeriesOneHotEncoder::<&str>::new( CategoryMapper::from_category_map(category_map));
        let oh_vec: Vec<f64> = match enc.transform_one(&"dog") {
            None => panic!("Wrong categories"),
            Some(v) => v,
        };
        let res: Vec<f64> = vec![0f64, 1f64, 0f64];
        assert_eq!(oh_vec, res);
    }

    #[test]
    fn positional_categories_vec() {
        let enc = build_fake_str_enc();
        let oh_vec: Vec<f64> = match enc.transform_one(&"dog") {
            None => panic!("Wrong categories"),
            Some(v) => v,
        };
        let res: Vec<f64> = vec![0.0, 1.0, 0.0];
        assert_eq!(oh_vec, res);
    }

    #[test]
    fn invert_label_test() {
        let enc = build_fake_str_enc();
        let res: Vec<f64> = vec![0.0, 1.0, 0.0];
        let lab = enc.invert_one(res).unwrap();
        assert_eq!(lab, "dog");
        if let Err(e) = enc.invert_one(vec![0.0, 0.0, 0.0]) {
            let pos_entries = format!("Expected a single positive entry, 0 entires found");
            assert_eq!(e, Failed::transform(&pos_entries[..]));
        };
    }

    #[test]
    fn test_many_categorys() {
        let enc = build_fake_str_enc();
        let cat_it = ["dog", "cat", "fish", "background"].iter().cloned();
        let res: Vec<Option<Vec<f64>>> = enc.transform_iter(cat_it);
        let v = vec![
            Some(vec![0.0, 1.0, 0.0]),
            Some(vec![0.0, 0.0, 1.0]),
            None,
            Some(vec![1.0, 0.0, 0.0]),
        ];
        assert_eq!(res, v)
    }
}
