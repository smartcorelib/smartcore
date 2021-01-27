#![allow(clippy::ptr_arg)]
//! # Encode categorical features as a one-hot or multi-class numeric array.

use crate::error::Failed;
use crate::math::num::RealNumber;
use std::collections::HashMap;
use std::hash::Hash;

/// Make a one-hot encoded vector from a categorical variable
pub fn make_one_hot<T: RealNumber, V: BaseVector<T>>(label_idx: usize, num_labels: usize) -> V {
    let pos = T::from_f64(1f64).unwrap();
    let mut z = V::zeros(num_labels);
    z.set(label_idx, pos);
    z
}
/// This struct encodes single class per exmample
///
/// You can fit a label enumeration by passing a collection of labels.
/// Label numbers will be assigned in the order they are encountered
///
/// Example:
/// ```
/// use std::collections::HashMap;
/// use smartcore::preprocessing::target_encoders::OneHotEncoder;
///
/// let fake_labels: Vec<usize> = vec![1,2,3,4,5,3,5,3,1,2,4];
/// let enc = OneHotEncoder::<usize>::fit(&fake_labels[..]);
/// let oh_vec: Vec<f64> = enc.transform_one(&1).unwrap();
/// // notice that 1 is actually a zero-th positional label
/// assert_eq!(oh_vec, vec![1.0, 0.0, 0.0, 0.0, 0.0]);
/// ```
///
/// You can also pass a predefined label enumeration such as a hashmap `HashMap<LabelType, usize>` or a vector `Vec<LabelType>`
///
///
/// ```
/// use std::collections::HashMap;
/// use smartcore::preprocessing::target_encoders::OneHotEncoder;
///
/// let label_map: HashMap<&str, usize> =
/// vec![("cat", 2), ("background",0), ("dog", 1)]
/// .into_iter()
/// .collect();
/// let label_vec = vec!["background", "dog", "cat"];
///
/// let enc_lv = OneHotEncoder::<&str>::from_positional_label_vec(label_vec);
/// let enc_lm = OneHotEncoder::<&str>::from_label_map(label_map);
///
/// // ["background", "dog", "cat"]
/// println!("{:?}", enc_lv.get_labels());
/// assert_eq!(enc_lv.transform_one::<f64>(&"dog"), enc_lm.transform_one::<f64>(&"dog"))
/// ```
pub struct OneHotEncoder<LabelType> {
    label_to_idx: HashMap<LabelType, usize>,
    labels: Vec<LabelType>,
    num_classes: usize,
}

enum LabelDefinition<T> {
    LabelToClsNumMap(HashMap<T, usize>),
    PositionalLabel(Vec<T>),
}

/// Crearte a vector of size num_labels with zeros everywhere and 1 at label_idx (one-hot vector)
pub fn make_one_hot<T: RealNumber>(label_idx: usize, num_labels: usize) -> Vec<T> {
    let (pos, neg) = (T::from_f64(1f64).unwrap(), T::from_f64(0f64).unwrap());
    (0..num_labels)
        .map(|idx| if idx == label_idx { pos } else { neg })
        .collect()
}

impl<'a, LabelType: Hash + Eq + Clone> OneHotEncoder<LabelType> {
    /// Fit an encoder to a lable list
    pub fn fit(labels: &[LabelType]) -> Self {
        let mut label_map: HashMap<LabelType, usize> = HashMap::new();
        let mut class_num = 0usize;
        let mut unique_lables: Vec<LabelType> = Vec::new();

        for l in labels {
            if !label_map.contains_key(&l) {
                label_map.insert(l.clone(), class_num);
                unique_lables.push(l.clone());
                class_num += 1;
            }
        }
        Self {
            label_to_idx: label_map,
            num_classes: class_num,
            labels: unique_lables,
        }
    }

    /// Build an encoder from a predefined (label -> class number) map
    pub fn from_label_map(category_map: HashMap<CategoryType, usize>) -> Self {
        let mut _unique_cat: Vec<(CategoryType, usize)> =
            category_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
        _unique_cat.sort_by(|a, b| a.1.cmp(&b.1));
        let categories: Vec<CategoryType> = _unique_cat.into_iter().map(|a| a.0).collect();
        Self {
            num_categories: categories.len(),
            categories,
            category_map,
    }
    }

    /// Build an encoder from a predefined positional label-class num vector
    pub fn from_positional_label_vec(categories: Vec<CategoryType>) -> Self {
        // Self::from_label_def(LabelDefinition::PositionalLabel(categories))
        let category_map: HashMap<CategoryType, usize> = categories
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

    /// Transform a slice of label types into one-hot vectors
    /// None is returned if unknown label is encountered
    pub fn transform<U: RealNumber>(&self, labels: &[LabelType]) -> Vec<Option<Vec<U>>> {
        labels.iter().map(|l| self.transform_one(l)).collect()
    }

    /// Transform a single label type into a one-hot vector
    pub fn transform_one<U: RealNumber>(&self, label: &LabelType) -> Option<Vec<U>> {
        match self.label_to_idx.get(label) {
            None => None,
            Some(&idx) => Some(make_one_hot(idx, self.num_classes)),
        }
    }

    /// Get labels ordered by encoder's label enumeration
    pub fn get_labels(&self) -> &Vec<LabelType> {
        &self.labels
    }

    /// Invert one-hot vector, back to the label
    pub fn invert_one<U: RealNumber>(&self, one_hot: Vec<U>) -> Result<LabelType, Failed> {
        let pos = U::from_f64(1f64).unwrap();

        let s: Vec<usize> = one_hot
            .into_iter()
            .enumerate()
            .filter_map(|(idx, v)| if v == pos { Some(idx) } else { None })
            .collect();

        if s.len() == 1 {
            let idx = s[0];
            return Ok(self.labels[idx].clone());
        }
        let pos_entries = format!(
            "Expected a single positive entry, {} entires found",
            s.len()
        );
        Err(Failed::transform(&pos_entries[..]))
    }

    fn from_label_def(labels: LabelDefinition<LabelType>) -> Self {
        let (label_map, class_num, unique_lables) = match labels {
            LabelDefinition::LabelToClsNumMap(h) => {
                let mut _unique_lab: Vec<(LabelType, usize)> =
                    h.iter().map(|(k, v)| (k.clone(), *v)).collect();
                _unique_lab.sort_by(|a, b| a.1.cmp(&b.1));
                let unique_lab: Vec<LabelType> = _unique_lab.into_iter().map(|a| a.0).collect();
                (h, unique_lab.len(), unique_lab)
            }
            LabelDefinition::PositionalLabel(unique_lab) => {
                let h: HashMap<LabelType, usize> = unique_lab
                    .iter()
                    .enumerate()
                    .map(|(v, k)| (k.clone(), v))
                    .collect();
                (h, unique_lab.len(), unique_lab)
            }
        };
        Self {
            label_to_idx: label_map,
            num_classes: class_num,
            labels: unique_lables,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_labels() {
        let fake_labels: Vec<usize> = vec![1, 2, 3, 4, 5, 3, 5, 3, 1, 2, 4];
        let enc = OneHotEncoder::<usize>::fit(&fake_labels[0..]);
        let oh_vec: Vec<f64> = match enc.transform_one(&1) {
            None => panic!("Wrong labels"),
            Some(v) => v,
        };
        let res: Vec<f64> = vec![1f64, 0f64, 0f64, 0f64, 0f64];
        assert_eq!(oh_vec, res);
    }

    fn build_fake_str_enc<'a>() -> OneHotEncoder<&'a str> {
        let fake_label_pos = vec!["background", "dog", "cat"];
        let enc = OneHotEncoder::<&str>::from_positional_label_vec(fake_label_pos);
        enc
    }

    #[test]
    fn label_map_and_vec() {
        let label_map: HashMap<&str, usize> = vec![("background", 0), ("dog", 1), ("cat", 2)]
            .into_iter()
            .collect();
        let enc = OneHotEncoder::<&str>::from_label_map(label_map);
        let oh_vec: Vec<f64> = match enc.transform_one(&"dog") {
            None => panic!("Wrong labels"),
            Some(v) => v,
        };
        let res: Vec<f64> = vec![0f64, 1f64, 0f64];
        assert_eq!(oh_vec, res);
    }

    #[test]
    fn positional_labels_vec() {
        let enc = build_fake_str_enc();
        let oh_vec: Vec<f64> = match enc.transform_one(&"dog") {
            None => panic!("Wrong labels"),
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
    fn test_many_labels() {
        let enc = build_fake_str_enc();
        let res: Vec<Option<Vec<f64>>> = enc.transform(&["dog", "cat", "fish", "background"]);
        let v = vec![
            Some(vec![0.0, 1.0, 0.0]),
            Some(vec![0.0, 0.0, 1.0]),
            None,
            Some(vec![1.0, 0.0, 0.0]),
        ];
        assert_eq!(res, v)
    }
}
