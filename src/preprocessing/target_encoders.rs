#![allow(clippy::ptr_arg)]
//! # Encode categorical features as a one-hot or multi-class numeric array.
//!

use crate::error::Failed;
use crate::math::num::RealNumber;
use std::collections::HashMap;
use std::hash::Hash;

/// Turn a collection of label types into a one-hot vectors.
/// This struct encodes single class per exmample
pub struct OneHotEncoder<T> {
    label_to_idx: HashMap<T, usize>,
    labels: Vec<T>,
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
        .map(|idx| {
            if idx == label_idx {
                pos
            } else {
                neg
            }
        })
        .collect()
}

impl<'a, T: Hash + Eq + Clone> OneHotEncoder<T> {
    /// Fit an encoder to a lable list
    ///
    /// Label numbers will be assigned in the order they are encountered
    /// Example:
    /// ```
    /// let fake_labels: Vec<usize> = vec![1,2,3,4,5,3,5,3,1,2,4];
    /// let enc = OneHotEncoder::<usize>::fit(&fake_labels[0..]);
    /// let oh_vec = enc.transform_one(&1); // notice that 1 is actually a zero-th positional label
    /// assert_eq!(oh_vec, vec![1f64,0f64,0f64,0f64,0f64]);
    /// ```
    pub fn fit(labels: &[T]) -> Self {
        let mut label_map: HashMap<T, usize> = HashMap::new();
        let mut class_num = 0usize;
        let mut unique_lables: Vec<T> = Vec::new();

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
    ///
    /// Definition example:
    /// ```
    /// let fake_label_map: HashMap<&str, u32> = vec![("background",0), ("dog", 1), ("cat", 2)]
    /// .into_iter()
    /// .collect();
    /// let enc = OneHotEncoder::<&str>::from_label_map(fake_label_map);
    /// ```
    pub fn from_label_map(labels: HashMap<T, usize>) -> Self {
        Self::from_label_def(LabelDefinition::LabelToClsNumMap(labels))
    }
    /// Build an encoder from a predefined positional label-class num vector
    ///
    /// Definition example:
    /// ```
    /// let fake_label_pos = vec!["background","dog", "cat"];
    /// let enc = OneHotEncoder::<&str>::from_positional_label_vec(fake_label_pos);
    /// ```
    pub fn from_positional_label_vec(labels: Vec<T>) -> Self {
        Self::from_label_def(LabelDefinition::PositionalLabel(labels))
    }

    /// Transform a slice of label types into one-hot vectors
    /// None is returned if unknown label is encountered
    pub fn transform(&self, labels: &[T]) -> Vec<Option<Vec<f64>>> {
        labels.iter().map(|l| self.transform_one(l)).collect()
    }

    /// Transform a single label type into a one-hot vector
    pub fn transform_one(&self, label: &T) -> Option<Vec<f64>> {
        match self.label_to_idx.get(label) {
            None => None,
            Some(&idx) => Some(make_one_hot(idx, self.num_classes)),
        }
    }

    /// Invert one-hot vector, back to the label
    ///```
    /// let lab = enc.invert_one(res)?; // e.g. res = [0,1,0,0...] "dog" == class 1
    /// assert_eq!(lab, "dog")
    /// ```
    pub fn invert_one<U: RealNumber>(&self, one_hot: Vec<U>) -> Result<T, Failed> {
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

    fn from_label_def(labels: LabelDefinition<T>) -> Self {
        let (label_map, class_num, unique_lables) = match labels {
            LabelDefinition::LabelToClsNumMap(h) => {
                let mut _unique_lab: Vec<(T, usize)> =
                    h.iter().map(|(k, v)| (k.clone(), *v)).collect();
                _unique_lab.sort_by(|a, b| a.1.cmp(&b.1));
                let unique_lab: Vec<T> = _unique_lab.into_iter().map(|a| a.0).collect();
                (h, unique_lab.len(), unique_lab)
            }
            LabelDefinition::PositionalLabel(unique_lab) => {
                let h: HashMap<T, usize> = unique_lab
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
        let oh_vec = match enc.transform_one(&1) {
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
        let fake_label_map: HashMap<&str, usize> = vec![("background", 0), ("dog", 1), ("cat", 2)]
            .into_iter()
            .collect();
        let enc = OneHotEncoder::<&str>::from_label_map(fake_label_map);
        let oh_vec = match enc.transform_one(&"dog") {
            None => panic!("Wrong labels"),
            Some(v) => v,
        };
        let res: Vec<f64> = vec![0f64, 1f64, 0f64];
        assert_eq!(oh_vec, res);
    }

    #[test]
    fn positional_labels_vec() {
        let enc = build_fake_str_enc();
        let oh_vec = match enc.transform_one(&"dog") {
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
}
