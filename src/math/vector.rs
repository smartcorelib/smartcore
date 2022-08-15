use crate::math::num::RealNumber;
use std::collections::HashMap;

use crate::linalg::BaseVector;
pub trait RealNumberVector<T: RealNumber> {
    fn unique_with_indices(&self) -> (Vec<T>, Vec<usize>);
}

impl<T: RealNumber, V: BaseVector<T>> RealNumberVector<T> for V {
    fn unique_with_indices(&self) -> (Vec<T>, Vec<usize>) {
        let mut unique = self.to_vec();
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup();

        let mut index = HashMap::with_capacity(unique.len());
        for (i, u) in unique.iter().enumerate() {
            index.insert(u.to_i64().unwrap(), i);
        }

        let mut unique_index = Vec::with_capacity(self.len());
        for idx in 0..self.len() {
            unique_index.push(index[&self.get(idx).to_i64().unwrap()]);
        }

        (unique, unique_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn unique_with_indices() {
        let v1 = vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 4.0];
        assert_eq!(
            (vec!(0.0, 1.0, 2.0, 4.0), vec!(0, 0, 1, 1, 2, 0, 3)),
            v1.unique_with_indices()
        );
    }
}
